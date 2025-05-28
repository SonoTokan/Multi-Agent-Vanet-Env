import os
import sys

sys.path.append("./")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SumoTrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len, expected_vehicle_ids, x_max, y_max):
        """
        参数:
        csv_file: CSV 文件路径，文件中应包含 'real_time', 'vehicle_id', 'x', 'y'
        seq_len: 输入序列长度（目标是第 seq_len+1 个 timestep）
        expected_vehicle_ids: 预期车辆 ID 列表，如 ['veh1', 'veh2', 'veh3', 'veh4', 'veh5']
        x_max, y_max: 坐标归一化所用的最大值（归一化公式: pos_norm = pos / max_val）
        """
        self.seq_len = seq_len
        self.expected_vehicle_ids = expected_vehicle_ids
        self.num_vehicles = len(expected_vehicle_ids)

        current_dir = os.getcwd()
        print('reading trajectory csv')
        # 读取 CSV 数据
        df = pd.read_csv(csv_file)
        # 如果 CSV 中还未归一化，则根据 x_max 和 y_max 归一化
        df["x"] = df["x"] / x_max
        df["y"] = df["y"] / y_max

        # 按照 timestep 分组，构建字典： timestep -> {vehicle_id: (x,y)}
        # self.time_groups = {}
        self.time_groups = (
            df.groupby('real_time')
            .apply(lambda g: dict(zip(g['vehicle_id'], zip(g['x'], g['y']))))
            .to_dict()
        )
        # for t, group in df.groupby("real_time"):
        #     positions = {}
        #     for _, row in group.iterrows():
        #         vid = row["vehicle_id"]
        #         positions[vid] = (row["x"], row["y"])
        #     self.time_groups[t] = positions

        # 按时间顺序获取所有的 timestep（确保连续采样）
        self.timesteps = sorted(self.time_groups.keys())

        # 构造样本：每个样本由连续 (seq_len+1) 个 timestep 组成（前 seq_len 个作为输入，第 seq_len+1 个作为目标）
        self.samples = []
        for i in range(len(self.timesteps) - seq_len):
            seq_ts = self.timesteps[i : i + seq_len + 1]
            self.samples.append(seq_ts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回：
        input_seq: Tensor, shape (seq_len, num_vehicles, 2)
        target: Tensor, shape (num_vehicles, 2)
        mask: Tensor, shape (num_vehicles,) ，1 表示该车辆在目标 timestep 中有效，0 表示无效
        """
        seq_ts = self.samples[idx]  # 包含 seq_len+1 个连续 timestep
        input_seq = []
        # 构造输入序列：遍历前 seq_len 个 timestep
        for t in seq_ts[:-1]:
            positions = self.time_groups[t]
            frame = []
            for vid in self.expected_vehicle_ids:
                if vid in positions:
                    frame.append(list(positions[vid]))
                else:
                    frame.append([0.0, 0.0])  # 填充无效数据
            input_seq.append(frame)
        input_seq = np.array(
            input_seq, dtype=np.float32
        )  # shape: (seq_len, num_vehicles, 2)

        # 构造目标：最后一个 timestep
        t_target = seq_ts[-1]
        positions = self.time_groups[t_target]
        target_frame = []
        mask = []  # 针对 target 的 mask，1 表示该车辆有效，0 表示无效
        for vid in self.expected_vehicle_ids:
            if vid in positions:
                target_frame.append(list(positions[vid]))
                mask.append(1.0)
            else:
                target_frame.append([0.0, 0.0])
                mask.append(0.0)
        target_frame = np.array(
            target_frame, dtype=np.float32
        )  # shape: (num_vehicles, 2)
        mask = np.array(mask, dtype=np.float32)  # shape: (num_vehicles,)

        return (
            torch.tensor(input_seq),
            torch.tensor(target_frame),
            torch.tensor(mask),
        )

# ========================
# 1. Positional Encoding 模块
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 输入嵌入维度
        dropout: dropout 比例
        max_len: 允许的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码矩阵，形状为 (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 按照公式除以指数函数：PE(pos,2i)=sin(pos/10000^(2i/d_model))，PE(pos,2i+1)=cos(pos/10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor，形状 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

# ========================
# 2. 构造数据集：随机轨迹数据
# ========================
class TrajectoryDataset(Dataset):
    def __init__(self, num_samples, seq_len, num_vehicles):
        """
        num_samples: 样本数量
        seq_len: 序列长度（不包含目标时刻）
        num_vehicles: 车辆数量
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_vehicles = num_vehicles

        self.data = []  # 存储输入序列：[seq_len, num_vehicles, 2]
        self.targets = []  # 存储目标：下一时刻的车辆位置 [num_vehicles, 2]

        # 构造随机轨迹
        for _ in range(num_samples):
            # 初始化各车辆位置，随机在 [0,1] 内
            traj = np.zeros((seq_len + 1, num_vehicles, 2), dtype=np.float32)
            traj[0] = np.random.rand(num_vehicles, 2).astype(np.float32)
            for t in range(1, seq_len + 1):
                # 模拟小幅随机运动（可以根据实际物理模型修改）
                traj[t] = traj[t - 1] + 0.01 * np.random.randn(
                    num_vehicles, 2
                ).astype(np.float32)
                traj[t] = np.clip(traj[t], 0, 1)
            self.data.append(traj[:-1])  # 前 seq_len 步作为输入
            self.targets.append(traj[-1])  # 第 seq_len+1 步作为目标

        self.data = np.array(
            self.data
        )  # shape: (num_samples, seq_len, num_vehicles, 2)
        self.targets = np.array(
            self.targets
        )  # shape: (num_samples, num_vehicles, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回 Tensor 类型数据
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx])

# ========================
# 3. Transformer 模型定义
# ========================
class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, num_vehicles):
        """
        input_dim: 每个时间步输入的特征数（这里为 num_vehicles * 2，将所有车辆的位置拼接在一起）
        embed_dim: Transformer 中的嵌入维度
        num_layers: Transformer encoder 层数
        num_heads: 多头注意力头数
        num_vehicles: 车辆数量（用于 reshape 输出）
        """
        super(TransformerTrajectoryPredictor, self).__init__()
        self.input_dim = input_dim  # = num_vehicles * 2
        self.num_vehicles = num_vehicles

        # 先将输入通过全连接层映射到 embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer Encoder 层（注意 PyTorch 要求输入形状为 (seq_len, batch, embed_dim)）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 用一个全连接层将最后时刻的表示映射到预测值，输出维度与输入相同（预测所有车辆的位置）
        self.decoder = nn.Linear(embed_dim, input_dim)
        # 使用 Sigmoid 激活函数确保输出在 [0,1] 内
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        """
        src: Tensor, shape (batch_size, seq_len, num_vehicles * 2)
        """
        # embed: (batch_size, seq_len, embed_dim)
        src = self.embedding(src)
        src = self.pos_encoder(src)

        # Transformer 要求输入形状 (seq_len, batch_size, embed_dim)
        src = src.transpose(0, 1)
        encoded = self.transformer_encoder(src)

        # 取序列最后一个时间步的输出
        last_output = encoded[-1]  # shape: (batch_size, embed_dim)
        output = self.decoder(last_output)  # shape: (batch_size, num_vehicles*2)
        output = self.sigmoid(output)  # 强制将输出限制在 [0,1]

        # 重塑为 (batch_size, num_vehicles, 2)
        output = output.view(-1, self.num_vehicles, 2)
        return output

# 训练掩码，当车辆不到num_vehicles时
def masked_mse_loss(output, target, mask):
    """
    output: Tensor, shape (batch_size, num_vehicles, 2)
    target: Tensor, shape (batch_size, num_vehicles, 2)
    mask: Tensor, shape (batch_size, num_vehicles)，值为 1 表示该车辆数据有效，0 表示无效
    # 示例：假设 batch_size=2, num_vehicles=5, 每辆车有 2 个数值
    output = torch.tensor([[[0.5, 0.5],
                            [0.6, 0.6],
                            [0.7, 0.7],
                            [0.8, 0.8],
                            [0.9, 0.9]],
                            [[0.1, 0.1],
                            [0.2, 0.2],
                            [0.3, 0.3],
                            [0.4, 0.4],
                            [0.5, 0.5]]], dtype=torch.float32)

    target = torch.tensor([[[0.4, 0.4],
                            [0.5, 0.5],
                            [0.6, 0.6],
                            [0.0, 0.0],  # 无效数据（目标随便填，不会计算loss）
                            [0.0, 0.0]], # 无效数据
                            [[0.1, 0.1],
                            [0.2, 0.2],
                            [0.3, 0.3],
                            [0.4, 0.4],
                            [0.0, 0.0]]], dtype=torch.float32)

    # 构造 mask：1 表示有效，0 表示无效
    mask = torch.tensor([[1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0]], dtype=torch.float32)

    """
    # 计算每个数据点的 MSE，结果 shape: (batch_size, num_vehicles, 2)
    mse = (output - target) ** 2
    # 对最后一维求和得到每辆车的误差，shape: (batch_size, num_vehicles)
    mse = mse.sum(dim=-1)

    # 使用 mask，保证无效的车辆的误差为 0
    mse = mse * mask

    # 求所有有效点的平均
    loss = mse.sum() / mask.sum().clamp(min=1)
    return loss

def real_traj():
    # 配置参数
    
    # path = os.path.join(os.path.dirname(__file__), "data", "SMMnet", "course-meta.csv")
    csv_file = os.path.join(os.path.dirname(__file__), "data", "trajectory_log.csv") # 之前保存的 CSV 文件
    seq_len = 10
    # 每个RSU智能体下最多能看见的车辆数
    expected_vehicle_ids = ["veh1", "veh2", "veh3", "veh4", "veh5"]
    x_max = 1000.0  # 根据 SUMO 配置设置
    y_max = 1000.0

    dataset = SumoTrajectoryDataset(
        csv_file, seq_len, expected_vehicle_ids, x_max, y_max
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型及超参数
    num_vehicles = len(expected_vehicle_ids)
    input_dim = num_vehicles * 2
    embed_dim = 64
    num_layers = 2
    num_heads = 4
    learning_rate = 0.001

    model = TransformerTrajectoryPredictor(
        input_dim, embed_dim, num_layers, num_heads, num_vehicles
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    num_epochs = 1000
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            # batch 中包含：input_seq (batch, seq_len, num_vehicles, 2)，target (batch, num_vehicles, 2)，mask (batch, num_vehicles)
            input_seq, target, mask = batch
            batch_size = input_seq.shape[0]
            # 将 input_seq reshape 为 (batch, seq_len, num_vehicles*2)
            input_seq = input_seq.view(batch_size, seq_len, -1)

            optimizer.zero_grad()
            output = model(input_seq)  # 输出 (batch, num_vehicles, 2)
            loss = masked_mse_loss(output, target, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")