import sys

sys.path.append("./")

import numpy as np
import random
from collections import defaultdict
from vanet_env import env_light


class MultiAgentStrategies:
    def __init__(self, env):
        """
        Initialize the strategy module with the environment.

        Args:
            env: The multi-agent environment instance.
        """
        self.env = env
        self.num_agents = len(env.agents)
        self.action_spaces = {agent: env.action_space(agent) for agent in env.agents}

    def random_strategy(self, obs):
        """
        Random strategy where each RSU selects a random action.

        Returns:
            actions: Dict mapping agent IDs to their random actions.
        """
        actions = [self.action_spaces[agent].sample() for agent in self.env.agents]

        return [actions]

    def greedy_strategy(self):
        """
        Greedy strategy where each RSU selects the action maximizing the immediate reward.

        Returns:
            actions: Dict mapping agent IDs to their greedy actions.
        """
        actions = {}
        for agent in self.env.agents:
            best_action = None
            best_reward = float("-inf")
            self.env.reset()
            for _ in range(100):  # Sample 100 actions to approximate the best action.
                action = [
                    self.action_spaces[agent].sample() for agent in self.env.agents
                ]
                obs, reward, _, _, _ = self.env.step([action])
                if reward[agent] > best_reward:
                    best_reward = reward[agent]
                    best_action = action

            actions[agent] = best_action
        return [actions]

    def heuristic_strategy(self, obs):
        """
        A heuristic strategy balancing load among neighboring RSUs.

        Returns:
            actions: Dict mapping agent IDs to heuristic-based actions.
        """
        actions = {}
        for idx, agent in enumerate(self.env.agents):

            obs = obs[idx]
            local_obs = obs["local_obs"]

            # Example heuristic: prioritize balancing load and minimizing connections queue.
            # Simplified as assigning higher resources to underutilized RSUs.
            neighbor_loads = local_obs[2:]  # Extract neighboring RSU loads.
            target = np.argmin(neighbor_loads)  # Select the least loaded RSU.

            action = np.zeros(self.action_spaces[agent].shape)
            action[target] = 1.0  # Fully allocate resources to the selected RSU.
            actions[agent] = action

        return [actions]

    def run_experiment(self, strategy, steps=1000):
        """
        Run a simulation experiment with the given strategy.

        Args:
            strategy: Strategy function to generate actions.
            steps: Number of steps to simulate.

        Returns:
            metrics: Dict containing QoE, EE, and resource usage over time.
        """
        qoe_records = []
        ee_records = []
        resource_records = []
        reward_records = []

        obs, _ = self.env.reset()
        for _ in range(steps):
            actions = strategy(obs)
            obs, rewards, _, _, infos = self.env.step(actions)

            # Gather metrics.
            qoe = np.mean([float(v.job.qoe) for v in self.env.vehicles.values()])
            ee = np.mean([float(rsu.ee) for rsu in self.env.rsus])
            resource_usage = np.mean([rsu.cp_usage for rsu in self.env.rsus])
            if rewards != []:
                reward = np.mean([reward for reward in rewards.values()])

            qoe_records.append(qoe)
            ee_records.append(ee)
            resource_records.append(resource_usage)
            reward_records.append(reward)

        return {
            "QoE": qoe_records,
            "EE": ee_records,
            "Resources": resource_records,
            "Rewards": reward_records,
        }


env = env_light.Env(None, max_step=3600)
strategies = MultiAgentStrategies(env)
metrics_random = strategies.run_experiment(strategies.random_strategy, steps=3600)
print(f"random:{metrics_random}")
print(f"random_avg_step_reward:{sum(metrics_random['Rewards'])/ 3600}")
# metrics_greedy = strategies.run_experiment(strategies.greedy_strategy, steps=3600)
# print(f"random:{metrics_greedy}")
# metrics_heuristic = strategies.run_experiment(strategies.heuristic_strategy, steps=3600)
# print(f"random:{metrics_heuristic}")
