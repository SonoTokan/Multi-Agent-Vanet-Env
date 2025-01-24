import sys

sys.path.append("./")

import os
import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer
from tianshou.env import DummyVectorEnv
from pettingzoo.utils.env import ParallelEnv
from vanet_env.gym_env_sumo import Env
from tianshou.utils import TensorboardLogger, WandbLogger


class HybridMAPPO(nn.Module):
    """处理混合动作空间的多智能体策略网络"""

    def __init__(self, obs_space, action_space, agent_num, device="cuda"):
        super().__init__()
        self.action_space = action_space
        self.agent_num = agent_num
        self.device = device

        # 共享特征提取层
        self.shared_encoder = nn.Sequential(
            nn.Linear(self._get_obs_dim(obs_space), 256), nn.ReLU(), nn.Linear(256, 256)
        )

        # 各智能体的独立策略头
        self.actor_heads = nn.ModuleList(
            [self._create_actor_head(action_space) for _ in range(agent_num)]
        )

        # 集中式critic网络
        self.critic = nn.Sequential(
            nn.Linear(agent_num * self._get_obs_dim(obs_space), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _get_obs_dim(self, space):
        """计算观察空间总维度"""
        if isinstance(space, spaces.Dict):
            return sum([np.prod(s.shape) for s in space.spaces.values()])
        return np.prod(space.shape)

    def _create_actor_head(self, action_space):
        """创建混合动作空间处理头"""
        heads = nn.ModuleDict()
        # 离散动作处理
        if "connection" in action_space.spaces:
            nvec = action_space["connection"].nvec
            heads["connection"] = nn.Linear(256, np.prod(nvec))

        # 连续动作处理
        for key in ["computing_power_alloc", "bw_alloc", "cp_usage"]:
            if key in action_space.spaces:
                space = action_space[key]
                heads[key] = nn.Linear(256, 2 * np.prod(space.shape))  # mu and sigma

        # 二进制动作处理
        for key in ["handling_jobs", "caching"]:
            if key in action_space.spaces:
                space = action_space[key]
                heads[key] = nn.Linear(256, np.prod(space.shape))
        return heads

    def forward(self, obs, state=None):
        # 假设obs是字典格式 {agent_id: observation}
        shared_features = {}
        for agent_id, ob in obs.items():
            # 展平字典观察
            if isinstance(ob, dict):
                flat_ob = torch.cat([v.flatten() for v in ob.values()], dim=-1)
            else:
                flat_ob = ob.flatten()
            shared_features[agent_id] = self.shared_encoder(flat_ob)

        # 生成各智能体动作参数
        actions = {}
        for agent_id in range(self.agent_num):
            agent_feat = shared_features[f"agent_{agent_id}"]
            action_params = {}
            for key, head in self.actor_heads[agent_id].items():
                if key == "connection":
                    logits = head(agent_feat)
                    action_params[key] = logits.view(
                        *self.action_space["connection"].shape
                    )
                elif key in ["computing_power_alloc", "bw_alloc", "cp_usage"]:
                    params = head(agent_feat)
                    mu, sigma = torch.chunk(params, 2, dim=-1)
                    action_params[key] = (torch.sigmoid(mu), F.softplus(sigma))
                else:
                    action_params[key] = torch.sigmoid(head(agent_feat))
            actions[f"agent_{agent_id}"] = action_params

        # 集中式价值估计
        global_obs = torch.cat([f for f in shared_features.values()], dim=-1)
        value = self.critic(global_obs)
        return actions, value, state


class HybridPPOPolicy(PPOPolicy):
    """扩展PPO策略处理混合动作"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_action(self, act, logits):
        """处理混合动作采样"""
        actions = {}
        log_probs = []
        for key in act:
            if key == "connection":
                dist = torch.distributions.Categorical(logits=logits[key])
                actions[key] = dist.sample()
                log_probs.append(dist.log_prob(actions[key]))
            elif key in ["computing_power_alloc", "bw_alloc", "cp_usage"]:
                mu, sigma = logits[key]
                dist = torch.distributions.Normal(mu, sigma)
                actions[key] = dist.sample()
                log_probs.append(dist.log_prob(actions[key]).sum(-1))
            else:
                dist = torch.distributions.Bernoulli(logits=logits[key])
                actions[key] = dist.sample()
                log_probs.append(dist.log_prob(actions[key]).sum(-1))
        return actions, sum(log_probs)


def create_tensorboard_logger():
    from torch.utils.tensorboard import SummaryWriter

    log_path = os.path.join("logs", "hybrid_mappo")
    writer = SummaryWriter(log_path)
    return TensorboardLogger(writer)


# 定义检查点保存路径
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(policy, buffer, epoch=0, step=0, filename="latest_checkpoint.pth"):
    """保存训练检查点"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

    # 保存策略状态（包含网络参数和优化器状态）
    policy_state = policy.state_dict()

    # 保存回放缓冲区
    buffer_path = os.path.join(CHECKPOINT_DIR, "buffer.hdf5")
    buffer.save_hdf5(buffer_path)

    # 保存元数据
    metadata = {"epoch": epoch, "step": step, "buffer_path": buffer_path}

    # 打包保存
    torch.save({"policy_state": policy_state, "metadata": metadata}, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(policy, buffer, filename="latest_checkpoint.pth"):
    """加载训练检查点"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # 加载检查点数据
    checkpoint = torch.load(checkpoint_path)

    # 加载策略状态
    policy.load_state_dict(checkpoint["policy_state"])

    # 加载回放缓冲区
    buffer.load_hdf5(checkpoint["metadata"]["buffer_path"])

    # 返回训练进度
    return checkpoint["metadata"]["epoch"], checkpoint["metadata"]["step"]


def train_with_resume(policy, train_collector, max_epoch=100, step_per_epoch=5000):
    logger = create_tensorboard_logger()
    start_epoch = 0
    start_step = 0

    # 尝试加载现有检查点
    try:
        start_epoch, start_step = load_checkpoint(policy, train_collector.buffer)
        print(f"Resuming training from epoch {start_epoch}, step {start_step}")
    except FileNotFoundError:
        print("No checkpoint found, starting new training")

    # 调整训练器参数
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector=None,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=5,
        episode_per_test=10,
        batch_size=1024,
        step_per_collect=1000,
        logger=logger,
        save_best_fn=lambda policy: save_checkpoint(
            policy, train_collector.buffer, start_epoch, start_step
        ),
        resume_from_log=True,
        start_epoch=start_epoch,
    )

    return result


# 训练流程
def train_hybrid_mappo():

    logger = create_tensorboard_logger()  # 选择TensorBoard方式
    # 环境初始化
    env = Env(None)
    agent_num = env.num_agents

    # 策略网络
    net = HybridMAPPO(env.observation_space(1), env.action_space(1), agent_num)
    actor_optim = torch.optim.Adam(net.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 多智能体策略管理器
    policies = MultiAgentPolicyManager(
        [
            HybridPPOPolicy(
                actor=net,
                critic=net.critic,
                actor_optim=actor_optim,
                critic_optim=critic_optim,
                action_space=env.action_space(1),
                dist_fn=lambda x: x,  # 自定义分布处理
            )
            for _ in range(agent_num)
        ]
    )

    # 数据收集
    train_collector = Collector(
        policies,
        DummyVectorEnv([lambda: env] * 10),
        VectorReplayBuffer(200000, 10),
    )

    # 训练参数
    result = onpolicy_trainer(
        policies,
        train_collector,
        test_collector=None,
        max_epoch=100,
        step_per_epoch=5000,
        repeat_per_collect=5,
        episode_per_test=10,
        batch_size=1024,
        step_per_collect=1000,
        save_best_fn=lambda policy: save_checkpoint(policy, train_collector.buffer),
        logger=logger,
    )

    return result


train_hybrid_mappo()
