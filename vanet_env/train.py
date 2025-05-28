#!/usr/bin/env python
import math
import sys
import os

sys.path.append("./")
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from vanet_env.env import Env
from vanet_env.onpolicy.config import get_config
from vanet_env.onpolicy.envs.env_wrappers import (
    ShareSubprocVecEnv,
    ShareDummyVecEnv,
)


env_max_step = 10850
max_step = env_max_step * 1000
is_discrete = True
map_name = "london"


def make_train_env():
    def get_env_fn():
        def init_env():
            env = Env(
                None, max_step=env_max_step, is_discrete=is_discrete, map=map_name
            )

            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)


def make_eval_env():
    def get_env_fn():
        def init_env():
            env = Env(
                None, max_step=env_max_step, is_discrete=is_discrete, map=map_name
            )
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn()], is_discrete=is_discrete)


def parse_args(args, parser):
    parser.add_argument(
        "--map_name", type=str, default="Seattle", help="Which sumo map to run on"
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    from env_config import SEED

    n_training_threads = 1
    cuda_deterministic = False
    # trick1: time-spilit
    time_spilit = False
    # trick2: self-attention
    use_cadp = True
    cadp_breakpoint = math.floor(max_step * 0.01)
    env_name = "vanet"
    alg_name = "ippo"
    exp_prefix = "time_all" if not time_spilit else "time_spilitted"
    use_wandb = True
    seed = SEED
    use_eval = False
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.map_name = map_name

    if not all_args.seed_specify:
        all_args.seed = np.random.randint(10000, 100000)

    print("seed is :", all_args.seed)

    all_args.use_cadp = use_cadp
    all_args.cadp_breakpoint = cadp_breakpoint
    all_args.critic_lr = 4e-4
    all_args.lr = 5e-4
    all_args.num_env_steps = max_step
    all_args.episode_length = env_max_step if not time_spilit else env_max_step // 10
    all_args.log_interval = 1
    all_args.algorithm_name = alg_name
    all_args.experiment_name = (
        (
            exp_prefix + "_" + "Mulit_discrete"
            if is_discrete
            else exp_prefix + "_" + "Box"
        )
        + "_cadp"
        if use_cadp
        else ""
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
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_"
            + str("nb")
            + "_seed"
            + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
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
    envs = make_train_env()
    eval_envs = make_eval_env()
    num_agents = envs.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from vanet_env.onpolicy.runner.shared.vanet_runner2 import (
        VANETRunner as Runner,
    )

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if use_eval and eval_envs is not envs:
        eval_envs.close()

    if use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
