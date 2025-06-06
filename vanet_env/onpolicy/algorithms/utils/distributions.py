import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class SquashedNormal(torch.distributions.TransformedDistribution):
    def __init__(self, loc, scale):
        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    def log_probs(self, value):
        # 自动计算变换后的对数概率（含 Jacobian 校正）
        return super().log_prob(value).sum(-1, keepdim=True)

    def mode(self):
        return self.mean

    def entropy(self):
        return self.base_dist.entropy().sum(-1)


class FixedBeta(torch.distributions.Beta):
    def mode(self):
        return (self.concentration0 - 1) / (
            self.concentration0 + self.concentration1 - 2
        )

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def sample(self):
        return super().sample()


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

    # 采样后限制到 [0.0, 1.0]
    def clamped_sample(self):
        return torch.sigmoid(super().sample())


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian_beta(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian_beta, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        # Get the log standard deviation, applying a non-negative constraint
        action_logstd = self.logstd(zeros)

        # Apply a softmax/clamp to ensure action_logstd doesn't contain negative values
        action_logstd = torch.clamp(
            action_logstd, min=1e-6
        )  # Ensure no values are less than a small positive number

        # Exponentiate the logstd to get std
        action_logstd = action_logstd.exp()

        return FixedBeta(action_mean, action_logstd)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        # super(DiagGaussian, self).__init__()
        # init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        # def init_(m):
        #     init_method(m.weight, gain=gain)
        #     nn.init.constant_(m.bias, 0)
        #     return m

        # self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        # self.logstd = nn.Parameter(torch.zeros(num_outputs))

        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        # action_mean = torch.sigmoid(self.fc_mean(x))  # 使用 Sigmoid 限制到 [0.0, 1.0]
        # # action_mean = torch.tanh(self.fc_mean(x))  # 使用 Tanh 限制到 [-1.0, 1.0]
        # # action_mean = (action_mean + 1) / 2  # 缩放到 [0.0, 1.0]
        # # 原始：
        # # action_mean = self.fc_mean(x)

        # #  An ugly hack for my KFAC implementation.
        # zeros = torch.zeros(action_mean.size())
        # if x.is_cuda:
        #     zeros = zeros.cuda()

        # action_logstd = self.logstd(zeros)

        action_mean = self.fc_mean(x)
        # action_mean = torch.tanh(self.fc_mean(x))  # 使用 Tanh 限制到 [-1.0, 1.0]
        # action_mean = (action_mean + 1) / 2  # 缩放到 [0.0, 1.0]
        # 原始：
        # action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

        # action_mean = self.fc_mean(x)
        # action_std = torch.exp(self.logstd)
        # return SquashedNormal(action_mean, action_std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        # my env shape torch.Size([20, 32])
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
