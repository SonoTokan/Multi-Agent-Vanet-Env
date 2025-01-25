import sys

sys.path.append("./")

from distutils.util import strtobool
from vanet_env.gym_env_sumo import Env

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, gain=1.0, bias_const=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        # 自动计算标准差
        fan_in = (
            layer.weight.size(1)
            if isinstance(layer, nn.Linear)
            else layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        )
        std = gain * np.sqrt(2.0 / fan_in)
        torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # 自动计算输入维度（MultiDiscrete各维度之和）
        self.obs_dim = sum(observation_space.nvec)
        self.action_dims = action_space.nvec  # 各动作分支的维度

        # 全连接网络处理向量观测
        self.network = nn.Sequential(
            self._layer_init(nn.Linear(self.obs_dim, 256)),
            nn.ReLU(),
            self._layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        # 多分支动作输出头
        self.actors = nn.ModuleList(
            [
                self._layer_init(nn.Linear(256, dim), std=0.01)
                for dim in self.action_dims
            ]
        )

        # 价值函数输出
        self.critic = self._layer_init(nn.Linear(256, 1), std=1.0)

    def _layer_init(self, layer, gain=1.0, std=None, bias_const=0.0):
        # 改进的初始化方法
        if isinstance(layer, nn.Linear):
            fan_in = layer.weight.size(1)
            if std is None:
                std = gain * np.sqrt(2.0 / fan_in)  # He初始化
            torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)

        # 生成各动作分支的logits
        logits = [actor(hidden) for actor in self.actors]

        # 创建独立分布
        distributions = [Categorical(logits=logit) for logit in logits]

        # 采样动作
        if action is None:
            action = torch.stack([d.sample() for d in distributions], dim=1)

        # 计算总log概率和熵
        log_probs = torch.stack(
            [
                distributions[i].log_prob(action[:, i])
                for i in range(len(distributions))
            ],
            dim=1,
        ).sum(dim=1)

        entropies = torch.stack([d.entropy() for d in distributions], dim=1).sum(dim=1)

        return action, log_probs, entropies, self.critic(hidden)


# 修改后的batchify_obs（适用于向量观测）
def batchify_obs(obs, device):
    """Converts dict of vector observations to batched tensor"""
    # Stack agent observations along batch dimension
    obs = np.stack([obs[agent] for agent in obs], axis=0)  # (n_agents, obs_dim)
    # Convert to float tensor (important for network)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    return obs  # shape: (batch_size, obs_dim)


# 修改后的unbatchify（适配MultiDiscrete动作）
def unbatchify(actions, env):
    """Converts batched actions to per-agent dict"""
    actions = actions.cpu().numpy()
    # 假设每个动作分支是一个列向量
    return {
        agent: [actions[i][k] for k in range(len(env.action_space.nvec))]
        for i, agent in enumerate(env.possible_agents)
    }


def batchify(x, device):
    """将PettingZoo风格的返回值（如rewards/dones）转换为批量张量"""
    x = np.stack([x[a] for a in x], axis=0)  # 堆叠成数组 (n_agents,)
    x = torch.tensor(x).to(device)  # 转换为张量
    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 4
    frame_size = (64, 64)
    max_cycles = 125
    total_episodes = 2

    """ ENV SETUP """
    env = Env(None)
    num_agents = len(env.possible_agents)
    action_space = env.action_space(env.possible_agents[0])
    observation_space = env.observation_space(env.possible_agents[0])

    action_dims = action_space.nvec  # 每个分支的维度数组
    num_action_branches = len(action_dims)  # 动作分支数量

    # observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(action_space=action_space, observation_space=observation_space).to(
        device
    )
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            # reset the episodic return
            total_episodic_return = 0

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device)

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    """ RENDER THE POLICY """
    env = Env()

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
