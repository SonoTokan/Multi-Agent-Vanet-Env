import cProfile
import math
import pstats
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

    def heuristic_strategy(self, obs, infos):
        """
        heuristic_strategy makes self and neighboring rsu avg resources and avg balance
        A heuristic strategy balancing load among neighboring RSUs.

        """
        actions = []
        for idx, agent in enumerate(self.env.agents):
            agent_obs = obs[agent]
            local_obs = agent_obs["local_obs"]
            self_handlings = local_obs[1]
            nb_state = local_obs[4]

            idle_nb_ratio = 1 - nb_state
            idle_self_ratio = 1 - self_handlings
            idle_all_ratio = idle_self_ratio + idle_nb_ratio

            queue_connections = local_obs[2]
            connected = local_obs[3]
            idle_connected = 1 - connected

            if idle_all_ratio == 0:
                # 几乎不可能都吃满，假如都吃满则均分
                self_mratio = [2] * self.env.action_space_dims[0]
                nb_mratio = [2] * self.env.action_space_dims[1]
            else:
                self_mratio = [
                    math.floor(idle_self_ratio / idle_all_ratio * self.env.bins)
                ] * self.env.action_space_dims[0]
                nb_mratio = [
                    math.floor(idle_nb_ratio / idle_all_ratio * self.env.bins)
                ] * self.env.action_space_dims[1]
            # 顺便确认nb是否正常
            # 根据idle_ratio确定分配值？
            # choice里的值可以改为 math.floor(... * self.env.bins)
            if idle_all_ratio >= 1:
                # 高空闲倾向于分配高job_ratio和高alloc、cp_usage等
                job_ratios = [
                    int(np.random.choice([3, 4]))
                ] * self.env.action_space_dims[2]
                cp_alloc = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    3
                ]
                # 节能选择
                cp_usage = [
                    int(np.random.choice([0, 1, 2]))
                ] * self.env.action_space_dims[5]
            else:
                job_ratios = [
                    int(np.random.choice([0, 1, 2]))
                ] * self.env.action_space_dims[2]
                cp_alloc = [
                    int(np.random.choice([0, 1, 2]))
                ] * self.env.action_space_dims[3]
                # 激进选择
                cp_usage = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    5
                ]

            # 空闲可支配大于需要的，bw策略可以激进
            if idle_connected >= queue_connections:
                bw_alloc = [int(np.random.choice([3, 4]))] * self.env.action_space_dims[
                    4
                ]
            else:
                bw_alloc = [
                    int(np.random.choice([0, 1, 2]))
                ] * self.env.action_space_dims[4]

            # 模拟FIFO，这里应该写在env里然后这里调用，时间匆忙，直接写外面了
            # 模拟LRU，这里应该写在env里然后这里调用，时间匆忙，直接写外面了
            # 模拟Random，这里应该写在env里然后这里调用，时间匆忙，直接写外面了
            rsu = self.env.rsus[idx]
            veh_id = rsu.range_connections.last()
            # 根据max_caching改动这一项
            if veh_id in self.env.vehicle_ids:
                caching_content = [self.env.vehicles[veh_id].job_type]
            else:
                caching_content = [
                    int(np.random.choice([i for i in range(self.env.bins)]))
                ]

            action = (
                self_mratio
                + nb_mratio
                + job_ratios
                + cp_alloc
                + bw_alloc
                + cp_usage
                + caching_content
            )

            if self.action_spaces[agent].contains(action):
                pass
            else:
                print("action 不在 action_space 内")
                assert IndexError("...")

            actions.append(action)

        return [actions]

    def fairalloc_strategy(self, obs, infos):
        pass

    # def heuristic_strategy(self, obs):
    #     """
    #     A heuristic strategy balancing load among neighboring RSUs.

    #     Returns:
    #         actions: Dict mapping agent IDs to heuristic-based actions.
    #     """
    #     actions = {}
    #     for idx, agent in enumerate(self.env.agents):

    #         obs = obs[idx]
    #         local_obs = obs["local_obs"]

    #         # Example heuristic: prioritize balancing load and minimizing connections queue.
    #         # Simplified as assigning higher resources to underutilized RSUs.
    #         neighbor_loads = local_obs[2:]  # Extract neighboring RSU loads.
    #         target = np.argmin(neighbor_loads)  # Select the least loaded RSU.

    #         action = np.zeros(self.action_spaces[agent].shape)
    #         action[target] = 1.0  # Fully allocate resources to the selected RSU.
    #         actions[agent] = action

    #     return [actions]

    def run_experiment(self, strategy=None, strategy_name=None, steps=1000):
        """
        Run a simulation experiment with the given strategy.

        Args:
            strategy: Strategy function to generate actions.
            steps: Number of steps to simulate.

        Returns:
            metrics: Dict containing QoE, EE, and resource usage over time.
        """
        if strategy is not None:
            self.strategy = strategy
        else:
            if strategy_name == "random_strategy":
                self.strategy = self.random_strategy
            elif strategy_name == "heuristic_strategy":
                self.strategy = self.heuristic_strategy
            else:
                self.strategy = self.random_strategy
        qoe_records = []

        ee_records = []
        resource_records = []
        reward_records = []
        hit_ratio_records = []

        obs, _, infos = self.env.reset()
        for _ in range(steps):
            actions = self.strategy(obs, infos)
            obs, rewards, _, _, infos = self.env.step(actions)

            # Gather metrics.
            qoe = np.mean([float(v.job.qoe) for v in self.env.vehicles.values()])
            qoe_real = []
            ees = []
            hit_ratios = []
            for rsu in self.env.rsus:
                for vid in rsu.range_connections:
                    if vid in self.env.vehicle_ids:
                        qoe_real.append(float(self.env.vehicles[vid].job.qoe))
                        for hit_ratio in rsu.hit_ratios:
                            hit_ratios.append(hit_ratio)
                        # 为什么ee至少有1.0
                        ees.append(float(rsu.ee))

            # resource_usage = np.mean([rsu.cp_usage for rsu in self.env.rsus])
            if rewards != []:
                reward = np.mean([reward for reward in rewards.values()])

            qoe_records.append(np.mean(qoe_real))
            ee_records.append(np.mean(ees))
            # resource_records.append(resource_usage)
            reward_records.append(reward)
            hit_ratio_records.append(np.nanmean(hit_ratios))

        return {
            "QoE": qoe_records,
            "EE": ee_records,
            "Rewards": reward_records,
            "Hit_ratio": hit_ratio_records,
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
    prefix = "train" if not is_eval else "eval"
    all_args.algorithm_name = "rmappo"
    all_args.experiment_name = "Mulit_discrete" if is_discrete else "Box"
    model_dir = "C:\\Users\\chentokan\\Documents\\Figs_Data\\model"
    # all_args.model_dir = (
    #     Path(
    #         os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    #         + "/saved_model"
    #     )
    #     / all_args.algorithm_name
    #     / all_args.experiment_name
    # )
    all_args.model_dir = model_dir

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
    exp_name = "multi_discrete"
    alg_name = "random_strategy"

    log = True
    # alg_name = "heuristic_strategy"

    step = 10240
    env = env_light.Env(None, max_step=step)
    strategies = MultiAgentStrategies(env)

    if log:
        metrics = strategies.run_experiment(strategy_name=alg_name, steps=step)

        print(f"{alg_name}:{metrics}")
        av = np.mean(metrics["Rewards"])
        avg_qoe = np.mean(metrics["QoE"])
        avg_hit_ratio = np.nanmean(metrics["Hit_ratio"])
        print(f"{alg_name}_avg_step_reward:{av}")
        print(f"{alg_name}_avg_step_qoe:{avg_qoe}")
        print(f"{alg_name}_avg_step_hit_ratio:{avg_hit_ratio}")
        df = pd.DataFrame(metrics, columns=["QoE", "EE", "Rewards", "Hit_ratio"])

        from datetime import datetime

        current_time = datetime.now()

        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(f"data_{exp_name}_{alg_name}_{formatted_time}.csv", index=False)
        print(f"CSV 文件已生成：data_{exp_name}_{alg_name}_{formatted_time}.csv")
    # metrics_greedy = strategies.run_experiment(strategies.greedy_strategy, steps=3600)
    # print(f"random:{metrics_greedy}")

    # metrics_heuristic = strategies.run_experiment(
    #     strategies.heuristic_strategy, steps=36_000_000
    # )
    # print(f"random:{metrics_heuristic}")


def main(args):
    # rmappo(args=args)
    # cProfile.run("other_policy()", sort="time")
    profiler = cProfile.Profile()
    profiler.enable()
    rmappo(args)
    profiler.disable()
    # 创建 Stats 对象并排序
    stats = pstats.Stats(profiler)
    stats.sort_stats("time")  # 按内部时间排序
    stats.reverse_order()  # 反转排序顺序（从升序变为降序，或从降序变为升序）
    stats.print_stats()  # 打印结果


if __name__ == "__main__":
    main(sys.argv[1:])
