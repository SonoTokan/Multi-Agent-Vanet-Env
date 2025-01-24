import sys

sys.path.append("./")

import torch
import torch.nn as nn
import torch.distributions as dist
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from vanet_env.model.obs import ObsModel


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


# 配置更新
config = {
    # ... 之前的配置 ...
    "model": {
        "custom_model": "custom_model",
        "custom_action_dist": "TorchCustomDistribution",  # 注册自定义分布
        "_disable_preprocessor_api": True,
    },
    # 调整动作相关参数
    "clip_param": 0.3,
    "lambda": 0.95,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": [[0, 0.01], [100000, 0.001]],  # 逐渐减少离散动作的探索
    # 连续动作特定参数
    "normalize_actions": False,  # 必须关闭自动标准化
    "clip_actions": False,  # 使用分布本身的截断
}
