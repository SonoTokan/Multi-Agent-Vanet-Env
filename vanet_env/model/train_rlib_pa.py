from pathlib import Path
import sys
from typing import Optional

import torch

sys.path.append("./")

import os

os.environ["RAY_DISABLE_ENV_CHECK"] = "1"  # 禁用环境检查

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from vanet_env.gym_env_sumo import Env
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class ActionMaskModelV2(TorchModelV2, nn.Module):
    """自定义模型处理动作掩码"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # 定义神经网络层
        self.fc = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.logits = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        # 提取观察中的动作掩码
        obs = input_dict["observtion"]
        action_mask = input_dict["action_mask"]

        # 前向传播
        features = self.fc(obs)
        logits = self.logits(features)

        # 应用动作掩码
        inf_mask = torch.clamp(
            torch.log(action_mask), min=torch.finfo(torch.float32).min
        )
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return torch.reshape(self.value_branch(self.features), [-1])


def env_creator(args):
    env = Env(None)
    return env


class CustomParallelPettingZooEnv(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        self.par_env = env
        self.par_env.reset()
        self._agent_ids = set(self.par_env.agents)

        self.observation_space = self.par_env.observation_space(self._agent_ids[0])
        self.action_space = self.par_env.action_space(self._agent_ids[1])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info or {}

    def step(self, action_dict):
        obss, rews, terminateds, truncateds, infos = self.par_env.step(action_dict)
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        return obss, rews, terminateds, truncateds, infos

    def close(self):
        self.par_env.close()

    def render(self):
        return self.par_env.render(self.render_mode)

    @property
    def get_sub_environments(self):
        return self.par_env.unwrapped


if __name__ == "__main__":
    from ray.rllib.utils import check_env

    # check_env(ParallelPettingZooEnv(Env(None)))

    ray.init()

    env_name = "vanet"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # ModelCatalog.register_custom_model("ActionMaskModelV2", ActionMaskModelV2)

    config = (
        PPOConfig()
        # .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .rollouts(
            num_rollout_workers=0,  # 禁用 Rollout Worker
            num_envs_per_worker=1,  # 每个 Worker 仅运行一个环境实例
        )
        .environment(env=env_name, clip_actions=True)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # 规范路径定义

    local_dir = os.path.join("C:\\Users\\chentokan\\ray_results\\vanet")

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 36000 * 200 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir=local_dir,
        config=config.to_dict(),
    )
