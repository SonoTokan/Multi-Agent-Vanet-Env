# train.py
import sys

sys.path.append("./")

import os
import argparse
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.tune.registry import register_env
import torch
import torch.nn as nn
import torch.distributions as dist
from pathlib import Path

# 假设你的自定义环境在my_env.py中
from vanet_env.gym_env_sumo import Env


# # 自定义模型和分布实现 --------------------------------------------------------
# class CustomObsModel(TorchModelV2, nn.Module):
#     """实现观察空间处理的基模型"""

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)

#         # 观察空间参数
#         self.num_rsus = obs_space["cpu_usage_per_rsu"].shape[0]
#         self.num_cores = obs_space["self_jobs_qoe"].shape[0]
#         self.max_connections = obs_space["self_connection_queue"].shape[0]

#         # 定义编码器层
#         self._build_encoders()

#         # 共享层
#         self.shared_fc = nn.Sequential(
#             nn.Linear(72, 256),  # 根据实际总维度调整
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#         )

#     def _build_encoders(self):
#         """构建各观察项的编码器"""
#         # 连续值编码器
#         self.cpu_encoder = nn.Linear(self.num_rsus, 16)
#         self.qoe_encoder = nn.Linear(self.num_rsus, 16)
#         self.jobs_qoe_encoder = nn.Linear(self.num_cores, 16)

#         # 二进制编码器
#         self.arrival_encoder = nn.Linear(self.num_cores, 8)
#         self.handling_encoder = nn.Linear(self.num_cores, 8)
#         self.conn_queue_encoder = nn.Linear(self.max_connections, 8)
#         self.connected_encoder = nn.Linear(self.max_connections, 8)

#     def forward(self, input_dict, state, seq_lens):
#         obs = input_dict["obs"]

#         # 编码各特征
#         features = [
#             torch.relu(self.cpu_encoder(obs["cpu_usage_per_rsu"].float())),
#             torch.relu(self.qoe_encoder(obs["avg_jobs_qoe_per_rsu"].float())),
#             torch.relu(self.jobs_qoe_encoder(obs["self_jobs_qoe"].float())),
#             torch.relu(self.arrival_encoder(obs["self_arrival_jobs"].float())),
#             torch.relu(self.handling_encoder(obs["self_handling_jobs"].float())),
#             torch.relu(self.conn_queue_encoder(obs["self_connection_queue"].float())),
#             torch.relu(self.connected_encoder(obs["self_connected"].float())),
#         ]

#         combined = torch.cat(features, dim=-1)
#         return self.shared_fc(combined), state


# class CustomActionModel(CustomObsModel):
#     """实现动作空间处理的完整模型"""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # 动作空间参数
#         action_space = self.action_space
#         self.max_content = action_space["caching"].shape[0]

#         # 构建各动作分支
#         self._build_action_heads()

#         # 值函数头
#         self.value_head = nn.Linear(128, 1)

#     def _build_action_heads(self):
#         """构建各动作输出层"""
#         # 1. Connection (MultiDiscrete)
#         self.conn_head = nn.Linear(128, self.max_connections * (self.num_rsus + 2))

#         # 2. Handling Jobs (MultiBinary)
#         self.handle_head = nn.Linear(128, self.num_cores)

#         # 3-5. 连续动作
#         self.comp_alloc_mu = nn.Linear(128, self.num_cores)
#         self.comp_alloc_log_std = nn.Parameter(torch.zeros(1, self.num_cores))

#         self.bw_alloc_mu = nn.Linear(128, self.num_cores)
#         self.bw_alloc_log_std = nn.Parameter(torch.zeros(1, self.num_cores))

#         self.cp_usage_mu = nn.Linear(128, 1)
#         self.cp_usage_log_std = nn.Parameter(torch.zeros(1, 1))

#         # 6. Caching
#         self.cache_head = nn.Linear(128, self.max_content)

#     def forward(self, input_dict, state, seq_lens):
#         features, state = super().forward(input_dict, state, seq_lens)

#         # 生成各动作参数
#         action_params = {
#             "connection": self.conn_head(features).view(
#                 -1, self.max_connections, self.num_rsus + 2
#             ),
#             "handling_jobs": self.handle_head(features),
#             "computing_power_alloc": (
#                 torch.sigmoid(self.comp_alloc_mu(features)),
#                 torch.exp(self.comp_alloc_log_std) + 1e-5,
#             ),
#             "bw_alloc": (
#                 torch.sigmoid(self.bw_alloc_mu(features)),
#                 torch.exp(self.bw_alloc_log_std) + 1e-5,
#             ),
#             "cp_usage": (
#                 torch.sigmoid(self.cp_usage_mu(features)),
#                 torch.exp(self.cp_usage_log_std) + 1e-5,
#             ),
#             "caching": self.cache_head(features),
#         }

#         self._value_out = self.value_head(features).squeeze(1)
#         return action_params, state

#     def value_function(self):
#         return self._value_out


# # 自定义动作分布 ------------------------------------------------------------
# class TorchCustomDistribution(dist.Distribution):
#     def __init__(self, action_params):
#         self.dists = {
#             "connection": dist.Categorical(logits=action_params["connection"]),
#             "handling_jobs": dist.Bernoulli(logits=action_params["handling_jobs"]),
#             "computing_power_alloc": dist.Normal(
#                 *action_params["computing_power_alloc"]
#             ).clamp(0.0, 1.0),
#             "bw_alloc": dist.Normal(*action_params["bw_alloc"]).clamp(0.0, 1.0),
#             "cp_usage": dist.Normal(*action_params["cp_usage"]).clamp(0.0, 1.0),
#             "caching": dist.Bernoulli(logits=action_params["caching"]),
#         }

#     def sample(self):
#         return {k: d.sample() for k, d in self.dists.items()}

#     def log_prob(self, actions):
#         return sum(d.log_prob(actions[k]) for d in self.dists.values())

#     def entropy(self):
#         return sum(d.entropy() for d in self.dists.values())


# # 注册到RLlib


class ObsModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 获取各观察项的维度信息
        self.num_rsus = obs_space["cpu_usage_per_rsu"].shape[0]
        self.num_cores = obs_space["self_jobs_qoe"].shape[0]
        self.max_connections = obs_space["self_connection_queue"].shape[0]

        # 定义各观察项的编码器 -------------------------------------------------
        # 连续值编码器
        self.cpu_encoder = nn.Sequential(
            nn.Linear(self.num_rsus, 16), nn.LayerNorm(16), nn.ReLU()
        )

        self.qoe_encoder = nn.Sequential(
            nn.Linear(self.num_rsus, 16), nn.LayerNorm(16), nn.ReLU()
        )

        self.jobs_qoe_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 16), nn.LayerNorm(16), nn.ReLU()
        )

        # 二进制特征编码器
        self.arrival_jobs_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 8), nn.ReLU()
        )

        self.handling_jobs_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 8), nn.ReLU()
        )

        self.conn_queue_encoder = nn.Sequential(
            nn.Linear(self.max_connections, 8), nn.ReLU()
        )

        self.connected_encoder = nn.Sequential(
            nn.Linear(self.max_connections, 8), nn.ReLU()
        )

        # 合并后的全连接层 -----------------------------------------------------
        total_dim = 16 + 16 + 16 + 8 * 4  # 各编码器输出维度之和
        self.shared_fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 值函数分支
        self.value_branch = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # 处理连续值特征 -------------------------------------------------------
        cpu_features = self.cpu_encoder(obs["cpu_usage_per_rsu"].float())
        qoe_features = self.qoe_encoder(obs["avg_jobs_qoe_per_rsu"].float())
        jobs_qoe_features = self.jobs_qoe_encoder(obs["self_jobs_qoe"].float())

        # 处理二进制特征 -------------------------------------------------------
        arrival_features = self.arrival_jobs_encoder(obs["self_arrival_jobs"].float())
        handling_features = self.handling_jobs_encoder(
            obs["self_handling_jobs"].float()
        )
        conn_queue_features = self.conn_queue_encoder(
            obs["self_connection_queue"].float()
        )
        connected_features = self.connected_encoder(obs["self_connected"].float())

        # 特征合并 -----------------------------------------------------------
        combined = torch.cat(
            [
                cpu_features,
                qoe_features,
                jobs_qoe_features,
                arrival_features,
                handling_features,
                conn_queue_features,
                connected_features,
            ],
            dim=-1,
        )

        # 共享层处理 ---------------------------------------------------------
        features = self.shared_fc(combined)
        self._value_out = self.value_branch(features).squeeze(1)

        return features, state

    def value_function(self):
        return self._value_out


class ActionDistribution(dist.Distribution):
    """自定义动作分布处理混合动作空间"""

    def __init__(self, action_components):
        self.components = action_components

    def sample(self):
        return {k: d.sample() for k, d in self.components.items()}

    def log_prob(self, actions):
        return sum(d.log_prob(actions[k]) for k, d in self.components.items())

    def entropy(self):
        return sum(d.entropy() for d in self.components.values())

    def kl(self, other):
        return sum(
            dist.kl_divergence(d, other.components[k])
            for k, d in self.components.items()
        )


class ActionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 共享特征编码部分（与之前观察空间处理相同）
        self.base_model = ObsModel(
            obs_space, action_space, num_outputs, model_config, name
        )

        # 动作空间参数 ---------------------------------------------------------
        self.num_rsus = action_space["connection"].nvec[0] - 2
        self.max_connections = action_space["connection"].shape[0]
        self.num_cores = action_space["handling_jobs"].shape[0]
        self.max_content = action_space["caching"].shape[0]

        # 定义各动作分支的输出层 ------------------------------------------------
        # 1. Connection (MultiDiscrete)
        self.conn_branch = nn.Linear(128, self.max_connections * (self.num_rsus + 2))

        # 2. Handling Jobs (MultiBinary)
        self.handle_branch = nn.Linear(128, self.num_cores)

        # 3. Computing Power Allocation (Box)
        self.comp_alloc_mu = nn.Linear(128, self.num_cores)
        self.comp_alloc_sigma = nn.Parameter(torch.zeros(self.num_cores))

        # 4. Bandwidth Allocation (Box)
        self.bw_alloc_mu = nn.Linear(128, self.num_cores)
        self.bw_alloc_sigma = nn.Parameter(torch.zeros(self.num_cores))

        # 5. CPU Usage (Box)
        self.cp_usage_mu = nn.Linear(128, 1)
        self.cp_usage_sigma = nn.Parameter(torch.zeros(1))

        # 6. Caching (MultiBinary)
        self.caching_branch = nn.Linear(128, self.max_content)

        # 值函数分支
        self.value_branch = nn.Linear(128, 1)

        # 自动调整参数初始化
        self._init_weights()

    def _init_weights(self):
        # 离散动作分支使用较小初始化
        nn.init.xavier_uniform_(self.conn_branch.weight, gain=0.1)
        nn.init.constant_(self.handle_branch.weight, 0.01)
        # 连续动作分支初始化
        nn.init.xavier_normal_(self.comp_alloc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.bw_alloc_mu.weight, gain=0.01)
        nn.init.constant_(self.cp_usage_mu.weight, 0.01)

    def forward(self, input_dict, state, seq_lens):
        # 共享特征提取
        features, state = self.base_model(input_dict, state, seq_lens)

        # 生成各动作参数 -------------------------------------------------------
        action_params = {}

        # 1. Connection (MultiDiscrete)
        conn_logits = self.conn_branch(features).view(
            -1, self.max_connections, self.num_rsus + 2
        )
        action_params["connection"] = conn_logits

        # 2. Handling Jobs (MultiBinary)
        handle_logits = self.handle_branch(features)
        action_params["handling_jobs"] = handle_logits

        # 3. Computing Power Allocation (Box)
        comp_mu = torch.sigmoid(self.comp_alloc_mu(features))  # 限制在0-1
        comp_sigma = torch.exp(self.comp_alloc_sigma) + 0.01  # 最小标准差
        action_params["computing_power_alloc"] = (comp_mu, comp_sigma)

        # 4. Bandwidth Allocation (Box)
        bw_mu = torch.sigmoid(self.bw_alloc_mu(features))
        bw_sigma = torch.exp(self.bw_alloc_sigma) + 0.01
        action_params["bw_alloc"] = (bw_mu, bw_sigma)

        # 5. CPU Usage (Box)
        cp_mu = torch.sigmoid(self.cp_usage_mu(features))
        cp_sigma = torch.exp(self.cp_usage_sigma) + 0.01
        action_params["cp_usage"] = (cp_mu, cp_sigma)

        # 6. Caching (MultiBinary)
        cache_logits = self.caching_branch(features)
        action_params["caching"] = cache_logits

        # 值函数
        self._value_out = self.value_branch(features).squeeze(1)

        return action_params, state

    def value_function(self):
        return self._value_out

    def action_distribution_fn(self, input_dict, state, seq_lens, explore, timestep):
        """自定义动作分布函数"""
        action_params, state = self.forward(input_dict, state, seq_lens)

        # 构建各动作的分布
        dist_dict = {}

        # 1. Connection: 独立分类分布
        dist_dict["connection"] = dist.Independent(
            dist.Categorical(logits=action_params["connection"]), 1
        )

        # 2. Handling Jobs: 独立伯努利分布
        dist_dict["handling_jobs"] = dist.Independent(
            dist.Bernoulli(logits=action_params["handling_jobs"]), 1
        )

        # 3-5. 连续动作使用截断正态分布
        for key in ["computing_power_alloc", "bw_alloc", "cp_usage"]:
            mu, sigma = action_params[key]
            dist_dict[key] = dist.Independent(
                dist.TruncatedNormal(mu, sigma, low=0.0, high=1.0), 1
            )

        # 6. Caching: 独立伯努利
        dist_dict["caching"] = dist.Independent(
            dist.Bernoulli(logits=action_params["caching"]), 1
        )

        return ActionDistribution(dist_dict), state

    # 在模型类中添加动作采样验证
    def _validate_actions(self, action_params, sample_obs):
        # 测试采样
        test_dist = self.action_distribution_fn(
            {"obs": sample_obs}, None, None, True, 0
        )

        # 检查动作形状
        sample_actions = test_dist.sample()
        assert sample_actions["connection"].shape == (self.max_connections,)
        assert sample_actions["computing_power_alloc"].shape == (self.num_cores,)
        # 其他动作项的形状检查...

        # 检查数值范围
        assert torch.all(sample_actions["cp_usage"] >= 0.0)
        assert torch.all(sample_actions["cp_usage"] <= 1.0)


# 注册自定义分布到RLlib
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper


class TorchCustomDistribution(TorchDistributionWrapper):
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        self.dist = ActionDistribution(inputs)

    def sample(self):
        return self.dist.sample()

    def logp(self, actions):
        return self.dist.log_prob(actions)

    def entropy(self):
        return self.dist.entropy()

    def kl(self, other):
        return self.dist.kl(other.dist)


# 环境注册 -----------------------------------------------------------------
ModelCatalog.register_custom_model("custom_model", ActionModel)
ModelCatalog.register_custom_action_dist("custom_dist", TorchCustomDistribution)


def env_creator(config: Dict[str, Any] = {}):
    return Env(None)  # 传入实际环境参数


register_env("my_env", lambda config: ParallelPettingZooEnv(env_creator()))


# 训练配置 -----------------------------------------------------------------
def get_config(args):
    return {
        "env": "my_env",
        "experimental": {"_enable_new_api_stack": False},  # no new api
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "framework": "torch",
        # 模型配置
        "model": {
            "custom_model": "custom_model",
            # "custom_action_dist": "custom_dist",
            "custom_action_dist": "custom_dist",  # 注册自定义分布
            "_disable_preprocessor_api": True,
        },
        # 训练参数
        "lr": args.lr,
        "gamma": 0.99,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 1000,
        "num_sgd_iter": 5,
        "clip_param": 0.2,
        "grad_clip": 0.5,
        # 多智能体配置
        "multiagent": {
            "policies": {
                "shared_policy": (
                    None,
                    Env(None).observation_space(1),
                    Env(None).action_space(1),
                    {},
                )
            },
            "policy_mapping_fn": lambda agent_id: "shared_policy",
        },
        # 调试设置
        "disable_env_checking": True,
        "log_level": "INFO",
    }


# 主函数 -------------------------------------------------------------------
def main(args):
    ray.init()

    # 训练配置
    config = get_config(args)
    stop_criteria = {"training_iteration": args.max_iters}
    # Convert relative path to absolute path
    absolute_log_dir = Path(args.log_dir).resolve()

    # 启动训练
    analysis = tune.run(
        PPO,
        config=config,
        stop=stop_criteria,
        checkpoint_at_end=True,
        storage_path=absolute_log_dir,
        verbose=3,
        resume=args.resume,
    )

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-gpus", type=float, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--log-dir", type=str, default="./results")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)

    main(args)
