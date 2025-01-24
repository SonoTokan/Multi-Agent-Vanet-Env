import sys

sys.path.append("./")

import torch.nn as nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class ObsModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # 获取各观察项的维度信息
        self.num_rsus = obs_space["cpu_usage_per_rsu"].shape[0]
        self.num_cores = obs_space["self_jobs_qoe"].shape[0]
        self.max_connections = obs_space["self_connection_queue"].shape[0]

        # 定义各观察项的编码器 -------------------------------------------------
        # 连续值编码器
        self.cpu_encoder = nn.Sequential(
            nn.Linear(self.num_rsus, 16), nn.LayerNorm(16), nn.ReLU()
        )

        self.qoe_encoder = nn.Sequential(
            nn.Linear(self.num_rsus, 16), nn.LayerNorm(16), nn.ReLU()
        )

        self.jobs_qoe_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 16), nn.LayerNorm(16), nn.ReLU()
        )

        # 二进制特征编码器
        self.arrival_jobs_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 8), nn.ReLU()
        )

        self.handling_jobs_encoder = nn.Sequential(
            nn.Linear(self.num_cores, 8), nn.ReLU()
        )

        self.conn_queue_encoder = nn.Sequential(
            nn.Linear(self.max_connections, 8), nn.ReLU()
        )

        self.connected_encoder = nn.Sequential(
            nn.Linear(self.max_connections, 8), nn.ReLU()
        )

        # 合并后的全连接层 -----------------------------------------------------
        total_dim = 16 + 16 + 16 + 8 * 4  # 各编码器输出维度之和
        self.shared_fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 值函数分支
        self.value_branch = nn.Linear(128, 1)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # 处理连续值特征 -------------------------------------------------------
        cpu_features = self.cpu_encoder(obs["cpu_usage_per_rsu"].float())
        qoe_features = self.qoe_encoder(obs["avg_jobs_qoe_per_rsu"].float())
        jobs_qoe_features = self.jobs_qoe_encoder(obs["self_jobs_qoe"].float())

        # 处理二进制特征 -------------------------------------------------------
        arrival_features = self.arrival_jobs_encoder(obs["self_arrival_jobs"].float())
        handling_features = self.handling_jobs_encoder(
            obs["self_handling_jobs"].float()
        )
        conn_queue_features = self.conn_queue_encoder(
            obs["self_connection_queue"].float()
        )
        connected_features = self.connected_encoder(obs["self_connected"].float())

        # 特征合并 -----------------------------------------------------------
        combined = torch.cat(
            [
                cpu_features,
                qoe_features,
                jobs_qoe_features,
                arrival_features,
                handling_features,
                conn_queue_features,
                connected_features,
            ],
            dim=-1,
        )

        # 共享层处理 ---------------------------------------------------------
        features = self.shared_fc(combined)
        self._value_out = self.value_branch(features).squeeze(1)

        return features, state

    def value_function(self):
        return self._value_out
