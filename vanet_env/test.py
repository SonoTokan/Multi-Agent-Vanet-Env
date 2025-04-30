import io
import os
import pstats
import sys

sys.path.append("./")
import numpy as np
from shapely import Point


import cProfile
from pettingzoo.test import parallel_api_test
from vanet_env.entites import Rsu, CustomVehicle
from vanet_env import utils
from vanet_env import network
from network import channel_capacity
import osmnx as ox
import matplotlib.pyplot as plt


def run_time_test():
    cProfile.run("channel_capacity(d, 20)", "restats_channel_capacity.restats")
    cProfile.run("render_test()", "restats_render_test.restats")


def print_stats():
    import pstats

    p = pstats.Stats("restats_channel_capacity.restats")
    p.sort_stats("cumulative").print_stats(10)
    p = pstats.Stats("restats_render_test.restats")
    p.sort_stats("cumulative").print_stats(10)


def network_test():
    # real distance (km)
    step = 0.001
    rsu = Rsu(1, (0, 0))
    while step <= 0.5:
        vh = CustomVehicle(1, Point((utils.real_distance_to_distance(step), 0)))
        print(f"real_distance: {step*1000:.2f} m, {channel_capacity(rsu, vh):.2f} Mbps")
        step += 0.01
    pass


def path_loss_test():
    winnerb1 = network.WinnerB1()
    winnerb1.test()


def render_test():
    env = Env(3)
    for i in range(5):
        env.render()


def test():
    env = Env(3)
    env.test()


def osmx_test():

    file_path = os.path.join(os.path.dirname(__file__), "assets", "seattle", "map.osm")
    G = ox.graph_from_xml(file_path)

    fig, ax = ox.plot_graph(G, node_size=5, edge_linewidth=0.5)
    plt.show()


# 3600s takes 25 seconds if render_mode = None
# 112,293,666 function calls in 97.557 seconds if render_mode = None when take _manage_rsu_vehicle_connections()
# 13864794 function calls in 6.782 seconds if render_mode = None without _manage_rsu_vehicle_connections()
# 3600s takes 105.441 seconds if render_mode = "human"
# 3600s takes 137.505 seconds if render lines by logic
# 3600s takes 136.182 seconds if using kdTree
# 3600s: 93277269 function calls in 125.288 seconds if using kdTree
# 3600s: 135685748 function calls in 111.391 seconds using new logic and min window
# 500s takes 16.412 seconds if render lines by logic
# 500s takes 16.939 seconds if using kdTree
# 500s takes 17.127 seconds using new logic
# 500s takes 11 seconds new render
# + queue list
# 11,405,122 function calls (11404904 primitive calls) in 15.304 seconds
# + lots of logics
# 12,107,925 function calls (12107707 primitive calls) in 14.846 seconds
# + lots of logics
# 12,041,754 function calls (12041532 primitive calls) in 11.103 seconds
# 100 sim_step * fps(10)
# 683965 function calls (683735 primitive calls) in 2.919 seconds
# None render
# 500 step-normal: 1,920,955 function calls in 1.502 seconds
# 500 step-getPos: 2,725,563 function calls in 4.650 seconds
# 500 step-getPos-logic: 12,153,777 function calls in 10.417 seconds
# 500 step-getPos-hasTree-logic: 8,218,740 function calls in 7.415 seconds
# 500 step-getPos-hasTree-logic-delete-render(): 3,926,358 function calls in 4.180 seconds
# 500 step-getPos-hasTree-logic-render()-init_all(): 3,516,416 function calls in 4.055 seconds
# 500 step-logic: 14,373,235 function calls in 12.490 seconds
# 500 step-getPos-hasTree-logic-render()-init_all() + Simulation version 1.21.0 started via libsumo with time: 0.00.
# 1,681,262 function calls in 1.604 seconds
# + orderd rsu conn list
# 1,995,522 function calls (1995304 primitive calls) in 2.147 seconds
# + orderd rsu conn list logic 2
# 2,213,830 function calls (2213612 primitive calls) in 1.877 seconds
# + queue list
# 2,152,464 function calls (1973712 primitive calls) in 1.983 seconds
# no empty determine
# 2,364,018 function calls (2363800 primitive calls) in 2.033 seconds
# + veh logics
# 2,652,298 function calls (2652080 primitive calls) in 2.224 seconds
# + frame step
# 1,867,325 function calls (1867095 primitive calls) in 1.448 seconds
# 3600 steps
# 13,356,345 function calls in 11.738 seconds


# fps 144?
def sumo_env_test():
    from vanet_env.env_light import Env

    fps = 10
    # render_mode="human", None
    env = Env("human")
    env.reset()

    for i in range(100 * fps):
        env.step({})


def draw():
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    # 初始化数据
    steps = []
    utilities = []
    caching_ratios = []

    # 创建图表
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # 创建两个子图，上下排列

    # 初始化第一个图表（utility vs step）
    (line1,) = ax1.plot(steps, utilities, "r-", label="Utility")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Utility")
    ax1.set_title("Utility over Steps")
    ax1.legend()

    # 初始化第二个图表（caching ratio vs step）
    (line2,) = ax2.plot(steps, caching_ratios, "b-", label="Caching Ratio")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Caching Ratio")
    ax2.set_title("Caching Ratio over Steps")
    ax2.legend()

    # 调整子图间距
    plt.tight_layout()

    # 模拟数据更新
    for step in range(100):  # 假设有100个step
        # 模拟utility和caching ratio的计算
        utility = np.random.rand()  # 随机数代替utility
        caching_ratio = np.random.rand()  # 随机数代替caching ratio

        # 更新数据
        steps.append(step)
        utilities.append(utility)
        caching_ratios.append(caching_ratio)

        # 更新第一个图表（utility）
        line1.set_xdata(steps)
        line1.set_ydata(utilities)
        ax1.relim()  # 重新计算坐标轴范围
        ax1.autoscale_view()  # 自动调整坐标轴范围

        # 更新第二个图表（caching ratio）
        line2.set_xdata(steps)
        line2.set_ydata(caching_ratios)
        ax2.relim()  # 重新计算坐标轴范围
        ax2.autoscale_view()  # 自动调整坐标轴范围

        # 绘制更新
        plt.draw()
        plt.pause(0.1)  # 暂停0.1秒以模拟实时更新

    plt.ioff()  # 关闭交互模式
    plt.show()


def trajtest():
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

    def random_traj():
        # ========================
        # 4. 训练设置与训练循环
        # ========================

        # 设置超参数
        num_samples = 1000  # 样本总数
        seq_len = 10  # 输入序列长度（时间步数）
        num_vehicles = 5  # 车辆数量
        batch_size = 32
        num_epochs = 10
        input_dim = num_vehicles * 2  # 每个时间步的输入特征维度
        embed_dim = 64
        num_layers = 2
        num_heads = 4
        learning_rate = 0.001

        # 构造数据集和 DataLoader
        dataset = TrajectoryDataset(num_samples, seq_len, num_vehicles)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型、损失函数和优化器
        model = TransformerTrajectoryPredictor(
            input_dim, embed_dim, num_layers, num_heads, num_vehicles
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练循环
        print("开始训练……")
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_idx, (data_batch, target_batch) in enumerate(dataloader):
                # 将输入数据 reshape：原始形状 (batch, seq_len, num_vehicles, 2) 转换为 (batch, seq_len, num_vehicles*2)
                batch_size_curr = data_batch.shape[0]
                data_batch = data_batch.view(batch_size_curr, seq_len, -1)

                optimizer.zero_grad()
                output = model(data_batch)  # 输出形状: (batch, num_vehicles, 2)
                loss = criterion(output, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # ========================
        # 5. 预测示例：给出一条输入轨迹，预测下一时刻的位置
        # ========================
        model.eval()
        with torch.no_grad():
            # 取数据集中的第 0 个样本作为示例
            sample_input, sample_target = dataset[0]
            # sample_input 的形状: (seq_len, num_vehicles, 2) -> reshape 为 (1, seq_len, num_vehicles*2)
            sample_input_flat = sample_input.view(1, seq_len, -1)
            predicted_next = model(sample_input_flat)  # 输出形状: (1, num_vehicles, 2)

            # 输出演示
            print("\n=== 预测示例 ===")
            print("输入序列最后一个时刻的车辆位置（ground truth 的最后输入时刻）:")
            print(sample_input[-1])  # shape: (num_vehicles, 2)
            print("\n真实下一时刻车辆位置:")
            print(sample_target)  # shape: (num_vehicles, 2)
            print("\n预测的下一时刻车辆位置:")
            print(predicted_next.squeeze(0))

    def real_traj():
        # 配置参数
        
        # path = os.path.join(os.path.dirname(__file__), "data", "SMMnet", "course-meta.csv")
        csv_file = os.path.join(os.path.dirname(__file__), "data", "trajectory_log.csv") # 之前保存的 CSV 文件
        seq_len = 10
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

    real_traj()


if __name__ == "__main__":
    # cProfile.run("sumo_env_test()", sort="time")
    # sumo_env_test()
    trajtest()
    # is_full = np.array([True, False, False])
    # is_in = np.array([False, True, False])
    # update_mask = is_full & is_in  # 需要更新的 RSU
    # store_mask = ~is_full & ~is_in  # 需要存入的 RSU
    # valid_mask = update_mask | store_mask
    # print(update_mask)
    # print(store_mask)
    # print(valid_mask)

    pass
