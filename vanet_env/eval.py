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

import sys
import os

sys.path.append("./")
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from vanet_env.env_light import Env
from vanet_env.onpolicy.config import get_config
from vanet_env.onpolicy.envs.env_wrappers import (
    ShareSubprocVecEnv,
    ShareDummyVecEnv,
)


env_max_step = 10240
max_step = env_max_step * 1000
is_discrete = True


def make_eval_env():
    def get_env_fn():
        def init_env():
            env = Env(None, max_step=env_max_step, is_discrete=is_discrete)

            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)


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
        Greedy: sample max_steps then storage

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


def parse_args(args, parser):
    parser.add_argument(
        "--map_name", type=str, default="Seattle", help="Which sumo map to run on"
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def rmappo(args):
    from env_config import SEED

    n_training_threads = 1
    cuda_deterministic = False
    env_name = "vanet"
    alg_name = "rMAPPO"
    use_wandb = True
    seed = SEED
    is_eval = True
    use_eval = False
    parser = get_config()
    all_args = parse_args(args, parser)
    if not all_args.seed_specify:
        all_args.seed = np.random.randint(10000, 100000)

    print("seed is :", all_args.seed)

    all_args.num_env_steps = max_step
    all_args.episode_length = env_max_step
    all_args.log_interval = 1
    prefix = "tran" if not is_eval else "eval"
    all_args.algorithm_name = "rmappo"
    all_args.experiment_name = "Mulit_discrete" if is_discrete else "Box"
    all_args.model_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/saved_model"
        )
        / all_args.algorithm_name
        / all_args.experiment_name
    )

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print(
            "u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False"
        )
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(n_training_threads)
        if cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(n_training_threads)

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / env_name
        / alg_name
    )

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name=str(prefix)
            + "_"
            + str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_"
            + str("nb")
            + "_seed"
            + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training" if not is_eval else "evaling",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(alg_name) + "-" + str(env_name))

    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # env
    envs = make_eval_env()
    eval_envs = make_eval_env()

    # eval_envs = make_eval_env()
    num_agents = envs.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    # if all_args.share_policy:
    #     from onpolicy.runner.shared.smac_runner import SMACRunner as Runner
    # else:
    #     from onpolicy.runner.separated.smac_runner import SMACRunner as Runner
    from vanet_env.onpolicy.runner.shared.vanet_runner2 import (
        VANETRunner as Runner,
    )

    runner = Runner(config)
    runner.run_eval()

    # post process
    envs.close()
    # if use_eval and eval_envs is not envs:
    #     eval_envs.close()

    if use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


def other_policy():

    step = 10240
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

    exp_name = "multi_discrete"
    alg_name = "random"
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"data_{formatted_time}.csv", index=False)
    print(f"CSV 文件已生成：data_{exp_name}_{alg_name}_{formatted_time}.csv.csv")
    # metrics_greedy = strategies.run_experiment(strategies.greedy_strategy, steps=3600)
    # print(f"random:{metrics_greedy}")

    # metrics_heuristic = strategies.run_experiment(
    #     strategies.heuristic_strategy, steps=36_000_000
    # )
    # print(f"random:{metrics_heuristic}")


def main(args):
    rmappo(args=args)


if __name__ == "__main__":
    main(sys.argv[1:])
