import sys
import os

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


def layer_init(layer, gain=1.0, bias_const=0.0, std=None):  
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        fan_in = (
            layer.weight.size(1)
            if isinstance(layer, nn.Linear)
            else layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        )
        # 如果提供了std参数则覆盖自动计算的值
        if std is None:
            std = gain * np.sqrt(2.0 / fan_in)
        torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # 处理不同类型的观测空间
        if isinstance(observation_space, gym.spaces.MultiDiscrete):
            self.obs_dim = sum(observation_space.nvec)
        elif isinstance(observation_space, gym.spaces.Box):
            self.obs_dim = observation_space.shape[0]
        else:
            raise NotImplementedError("Unsupported observation space")

        # 处理不同类型的动作空间
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_dims = action_space.nvec
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dims = [action_space.n]
        else:
            raise NotImplementedError("Unsupported action space")

        self.network = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        self.actors = nn.ModuleList(
            [layer_init(nn.Linear(256, dim), std=0.01) for dim in self.action_dims]
        )

        self.critic = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = [actor(hidden) for actor in self.actors]
        distributions = [Categorical(logits=logit) for logit in logits]

        if action is None:
            action = torch.stack([d.sample() for d in distributions], dim=1)

        log_probs = torch.stack(
            [
                distributions[i].log_prob(action[:, i])
                for i in range(len(distributions))
            ],
            dim=1,
        ).sum(dim=1)

        entropies = torch.stack([d.entropy() for d in distributions], dim=1).sum(dim=1)
        return action, log_probs, entropies, self.critic(hidden)


def batchify_obs(obs, device):
    """Convert dict of vector observations to batched tensor"""
    obs = np.stack([obs[agent] for agent in obs], axis=0)
    return torch.tensor(obs, dtype=torch.float32).to(device)


def unbatchify(actions, env):
    """Convert batched actions to per-agent dict"""
    actions = actions.cpu().numpy()
    return {agent: actions[i].tolist() for i, agent in enumerate(env.possible_agents)}


def batchify(x, device):
    """Convert PettingZoo-style dict to tensor"""
    return torch.tensor(np.stack([x[a] for a in x]), device=device)


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.01
    vf_coef = 0.5
    clip_coef = 0.2
    gamma = 0.99
    batch_size = 128
    max_cycles = 500
    total_episodes = 1000
    update_epochs = 4
    learning_rate = 2.5e-4

    """ ENV SETUP """
    env = Env(None)
    num_agents = len(env.possible_agents)
    agent_sample = env.possible_agents[0]
    action_space = env.action_space(agent_sample)
    observation_space = env.observation_space(agent_sample)

    """ LEARNER SETUP """
    agent = Agent(observation_space, action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    """ STORAGE SETUP """
    obs_shape = (num_agents, agent.obs_dim)
    rb_obs = torch.zeros((max_cycles, *obs_shape)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents, len(action_space.nvec))).to(
        device
    )
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    """ TRAINING LOOP """
    writer = SummaryWriter()
    global_step = 0

    for episode in range(total_episodes):
        with torch.no_grad():
            next_obs, _ = env.reset()
            episodic_return = 0
            end_step = max_cycles

            for step in range(max_cycles):
                global_step += num_agents
                obs = batchify_obs(next_obs, device)

                actions, logprobs, _, values = agent.get_action_and_value(obs)
                next_obs, rewards, terms, truncs, _ = env.step(unbatchify(actions, env))

                # 存储数据
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                episodic_return += rb_rewards[step].mean().item()

                if any(terms.values()) or any(truncs.values()):
                    end_step = step + 1
                    break

        """ ADVANTAGE CALCULATION """
        with torch.no_grad():
            advantages = torch.zeros_like(rb_rewards)
            lastgaelam = 0
            for t in reversed(range(end_step)):
                nextnonterminal = 1.0 - rb_terms[t]
                nextvalues = rb_values[t + 1] if t < end_step - 1 else 0.0
                delta = (
                    rb_rewards[t] + gamma * nextvalues * nextnonterminal - rb_values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + gamma * gamma * lastgaelam * nextnonterminal
                )
            returns = advantages + rb_values

        """ UPDATE NETWORK """
        b_obs = rb_obs[:end_step].reshape(-1, agent.obs_dim)
        b_actions = rb_actions[:end_step].reshape(-1, len(action_space.nvec))
        b_logprobs = rb_logprobs[:end_step].reshape(-1)
        b_advantages = advantages[:end_step].reshape(-1)
        b_returns = returns[:end_step].reshape(-1)

        for epoch in range(update_epochs):
            indices = torch.randperm(len(b_obs), device=device)
            for start in range(0, len(b_obs), batch_size):
                end = start + batch_size
                idx = indices[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[idx], b_actions[idx].long()
                )

                logratio = newlogprob - b_logprobs[idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Policy loss
                pg_loss1 = -b_advantages[idx] * ratio
                pg_loss2 = -b_advantages[idx] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((value.flatten() - b_returns[idx]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        """ LOGGING """
        writer.add_scalar("charts/episodic_return", episodic_return, episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)

        print(f"Episode {episode}/{total_episodes}")
        print(f"Episodic Return: {episodic_return:.2f}")
        print(f"Avg Reward: {episodic_return/end_step:.3f}")
        print("------------------------")

    """ SAVE MODEL """
    torch.save(agent.state_dict(), "vanet_agent.pth")
    writer.close()
