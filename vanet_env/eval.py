import sys

import pandas as pd

sys.path.append("./")

import numpy as np
import random
from collections import defaultdict
from vanet_env import env_light

seed = 114514
random.seed(114514)
np.random.seed(114514)


class MultiAgentStrategies:
    def __init__(self, env):
        """
        Initialize the strategy module with the environment.

        Args:
            env: The multi-agent environment instance.
        """
        self.env = env
        self.num_agents = len(env.agents)
        self.action_spaces = {
            agent: env.multi_discrete_action_space(agent) for agent in env.agents
        }

    def random_strategy(self, obs, infos):
        """
        Random strategy where each RSU selects a random action.

        Returns:
            actions: Dict mapping agent IDs to their random actions.
        """
        actions = [self.action_spaces[agent].sample() for agent in self.env.agents]

        return [actions]

    def greedy_strategy(self, obs, infos):
        """
        Greedy strategy where each RSU selects the action maximizing the immediate reward.

        Returns:
            actions: Dict mapping agent IDs to their greedy actions.
        """
        # actions = {}
        # for agent in self.env.agents:
        #     best_action = None
        #     best_reward = float("-inf")
        #     self.env.reset()
        #     for _ in range(100):  # Sample 100 actions to approximate the best action.
        #         action = [
        #             self.action_spaces[agent].sample() for agent in self.env.agents
        #         ]
        #         obs, reward, _, _, _ = self.env.step([action])
        #         if reward[agent] > best_reward:
        #             best_reward = reward[agent]
        #             best_action = action

        #     actions[agent] = best_action
        actions = {}
        for agent in self.env.agents:
            local_obs = obs[agent]["local_obs"]
            global_obs = obs[agent]["global_obs"]
            idles = infos[agent]["idle"]

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

        obs, _, infos = self.env.reset()
        for _ in range(steps):
            actions = strategy(obs, infos)
            obs, rewards, _, _, infos = self.env.step(actions)

            # Gather metrics.
            qoe = np.mean([float(v.job.qoe) for v in self.env.vehicles.values()])
            ee = np.mean([float(rsu.ee) for rsu in self.env.rsus])
            # resource_usage = np.mean([rsu.cp_usage for rsu in self.env.rsus])
            if rewards != []:
                reward = np.mean([reward for reward in rewards.values()])

            qoe_records.append(qoe)
            ee_records.append(ee)
            # resource_records.append(resource_usage)
            reward_records.append(reward)

        return {
            "QoE": qoe_records,
            "EE": ee_records,
            "Rewards": reward_records,
        }


step = 36000
env = env_light.Env(None, max_step=step)
strategies = MultiAgentStrategies(env)
metrics_random = strategies.run_experiment(strategies.random_strategy, steps=step)
print(f"random:{metrics_random}")
av = np.mean(metrics_random["Rewards"])
avg_qoe = np.mean(metrics_random["QoE"])
print(f"random_avg_step_reward:{av}")
print(f"random_avg_step_qoe:{avg_qoe}")
df = pd.DataFrame(metrics_random, columns=["QoE", "EE", "Rewards"])

from datetime import datetime

current_time = datetime.now()

formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
df.to_csv(f"data_{formatted_time}.csv", index=False)
print(f"CSV 文件已生成：data_{formatted_time}.csv.csv")
# metrics_greedy = strategies.run_experiment(strategies.greedy_strategy, steps=3600)
# print(f"random:{metrics_greedy}")

# metrics_heuristic = strategies.run_experiment(
#     strategies.heuristic_strategy, steps=36_000_000
# )
# print(f"random:{metrics_heuristic}")
