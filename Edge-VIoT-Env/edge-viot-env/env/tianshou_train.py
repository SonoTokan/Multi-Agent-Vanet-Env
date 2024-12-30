import os
from typing import Optional, Tuple

import gymnasium
import tianshou as ts
import torch
import numpy as np
from pettingzoo import AECEnv
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from edge_viot_env import EdgeVIoTEnv, raw_env

def get_env() -> PettingZooEnv:
    return PettingZooEnv(raw_env())

def get_agents(num_agents) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    
    env = get_env()
    
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    
    if agent_learn is None:
        # model
        net = ActorCritic(
            ActorProb(env.observation_space.shape, env.action_space.shape, hidden_sizes=[64, 64]),
            Critic(env.observation_space.shape, hidden_sizes=[64, 64])
        )
        
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=3e-4)
            
        agent_learn = PPOPolicy(
            net,
            optim,
            dist_fn=torch.distributions.Categorical,
            discount_factor=0.99,
            max_grad_norm=0.5,
            eps_clip=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            reward_normalization=True,
            action_scaling=True,
            action_bound_method="clip",
            gae_lambda=0.95,
            value_clip=True,
            dual_clip=None,
            advantage_normalization=True,
            recompute_advantage=True
        )
        
    agents = [agent_learn for _ in range(num_agents)]
    
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

def train(model_save_path: str,
          num_steps: int = 10_000,
          logdir: str = "log",
          training_num: int = 1,
          test_num: int = 1,
          seed: int = 114514,
          buffer_size: int = 20_000,
          batch_size: int = 64):
    
    # Initialize the custom PettingZoo environment
    env = get_env()

    # Define the network architecture
    # net = ActorCritic(
    #     ActorProb(env.observation_space.shape, env.action_space.shape, hidden_sizes=[64, 64]),
    #     Critic(env.observation_space.shape, hidden_sizes=[64, 64])
    # )
    # optim = torch.optim.Adam(net.parameters(), lr=3e-4)

    # # Define the PPO policy
    # policy = PPOPolicy(
    #     net,
    #     optim,
    #     dist_fn=torch.distributions.Categorical,
    #     discount_factor=0.99,
    #     max_grad_norm=0.5,
    #     eps_clip=0.2,
    #     vf_coef=0.5,
    #     ent_coef=0.01,
    #     reward_normalization=True,
    #     action_scaling=True,
    #     action_bound_method="clip",
    #     gae_lambda=0.95,
    #     value_clip=True,
    #     dual_clip=None,
    #     advantage_normalization=True,
    #     recompute_advantage=True
    # )

    # Create the collector
    # train_collector = Collector(policy, env, VectorReplayBuffer(20000, len(env)))
    # test_collector = Collector(policy, env)

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(test_num)])
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(env.num_agents)

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    train_collector.collect(n_step=batch_size * training_num)

    # ======== logging ========
    log_path = os.path.join(logdir, "viot", "ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    # ======== callback functions ======== 
    def save_best_fn(policy):
        # save num_agents policy
        for i in range(len(agents)):
            torch.save(
                policy.policies[agents[i]].state_dict(), os.path.join(model_save_path, f"policy_{agents[i]}.pth")
            )
    
    # ======== train the policy ======== 
    # result = onpolicy_trainer(
    #     policy,
    #     train_collector,
    #     test_collector,
    #     max_epoch=10,
    #     step_per_epoch=1000,
    #     repeat_per_collect=4,
    #     episode_per_test=10,
    #     batch_size=64,
    #     step_per_collect=10,
    #     save_best_fn=save_best_fn
    # )
    
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=10,
        step_per_epoch=1000,
        step_per_collect=10,
        test_num=10,
        batch_size=64,
        # train_fn=train_fn,
        # test_fn=test_fn,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        logger=logger,
        test_in_train=False,
    )
    
    print(f'Training finished! Use {result["duration"]}')

if __name__ == "__main__":
    result, agent = train("model", num_steps=10_000)