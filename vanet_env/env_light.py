import collections
import functools
import itertools
import math
import random
import sys

from vanet_env import utility_light
from vanet_env import env_config

sys.path.append("./")
import pandas as pd
from shapely import Point
from sklearn.preprocessing import MinMaxScaler

from vanet_env import data_preprocess

sys.path.append("./")

import os
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import poisson

# gym
import gymnasium as gym
from gymnasium import spaces
from gym import spaces as gym_spaces

# custom
from vanet_env import network, caching
from vanet_env.utils import (
    RSU_MARKER,
    VEHICLE_MARKER,
    interpolate_color,
    sumo_detector,
    is_empty,
    discrete_to_multi_discrete,
    multi_discrete_space_to_discrete_space,
)
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle

# sumo
import traci
import libsumo
from sumolib import checkBinary

# next performance improvement
import multiprocessing as mp


def raw_env(render_mode=None):
    env = Env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class Env(ParallelEnv):
    metadata = {"name": "sumo_vanet_environment_v0", "render_modes": ["human", None]}

    def __init__(
        self,
        render_mode="human",
        multi_core=8,
        caching_fps=10,  # time split to 10 pieces
        fps=10,
        max_step=36000,  # need focus on fps
        seed=env_config.SEED,
        is_discrete=True,
    ):

        self.is_discrete = is_discrete
        self.num_rsus = env_config.NUM_RSU
        # self.max_size = config.MAP_SIZE  # deprecated
        self.road_width = env_config.ROAD_WIDTH
        self.seed = seed
        self.render_mode = render_mode
        self.multi_core = multi_core
        self.max_step = max_step
        self.caching_fps = caching_fps
        self.caching_step = max_step // caching_fps

        random.seed(self.seed)

        # important rsu info
        self.max_connections = env_config.MAX_CONNECTIONS
        self.num_cores = env_config.NUM_CORES
        self.max_content = env_config.NUM_CONTENT
        self.max_caching = env_config.RSU_CACHING_CAPACITY
        # rsu range veh wait for connections
        self.max_queue_len = 10
        # every single weight
        self.max_weight = 10
        self.qoe_weight = 10
        self.bins = 5
        # fps means frame per second(frame per sumo timestep)
        self.fps = fps

        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.max_connections,
                max_cores=self.num_cores,
            )
            for i in range(self.num_rsus)
        ]
        self.rsu_ids = [i for i in range(self.num_rsus)]
        self._agent_ids = [f"rsu_{i}" for i in range(self.num_rsus)]
        self.avg_u_global = 0
        self.avg_u_local = 0

        # max data rate
        self.max_data_rate = network.max_rate(self.rsus[0])
        print(f"single atn max_data_rate:{self.max_data_rate:.2f}")

        # for convience
        self.rsu_positions = [rsu.position for rsu in self.rsus]
        self.rsu_coords = np.array(
            [
                (self.rsu_positions[rsu.id].x, self.rsu_positions[rsu.id].y)
                for rsu in self.rsus
            ]
        )
        self.rsu_tree = KDTree(self.rsu_coords)

        # network
        self.connections_queue = []
        self.connections = []
        self.rsu_network = network.network(self.rsu_coords, self.rsu_tree)

        # rsu max connection distance
        self.max_distance = network.max_distance_mbps(self.rsus[0])
        print(f"max_distance:{self.max_distance}")
        self.sumo = traci

        # pettingzoo init
        list_agents = list(self._agent_ids)
        self.agents = list.copy(list_agents)
        self.possible_agents = list.copy(list_agents)

        self.timestep = 0
        # 异常监测
        # 监测无法编排次数
        self.full_count = 0
        # 监测云次数
        self.cloud_times = 0

        self.content_loaded = False
        self._space_init()
        self._sumo_init()

    def _content_init(self):
        self.cache = caching.Caching(
            self.caching_fps, self.max_content, self.max_caching, self.seed
        )
        self.content_list, self.aggregated_df_list, self.aggregated_df = (
            self.cache.get_content_list()
        )
        self.cache.get_content(min(self.timestep // self.caching_step, 9))

    def _sumo_init(self):
        # SUMO detector
        # sumo_detector()
        print("sumo init")

        self.cfg_file_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "osm.sumocfg"
        )

        gui_settings_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "gui_hide_all.xml"
        )

        self.icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

        # start sumo sim
        if self.render_mode is not None:

            self.sumo.start(
                ["sumo-gui", "-c", self.cfg_file_path, "--step-length", "1", "--start"]
            )
            # paint rsu
            for rsu in self.rsus:
                poi_id = f"rsu_icon_{rsu.id}"
                # add rsu icon
                self.sumo.poi.add(
                    poi_id,
                    rsu.position.x,
                    rsu.position.y,
                    (205, 254, 194, 255),  # green
                    width=10,
                    height=10,
                    imgFile=self.icon_path,
                    layer=20,
                )
            # paint range
            for rsu in self.rsus:
                num_segments = 36
                for i in range(num_segments):
                    angle1 = 2 * np.pi * i / num_segments
                    angle2 = 2 * np.pi * (i + 1) / num_segments
                    x1 = rsu.position.x + self.max_distance * np.cos(angle1)
                    y1 = rsu.position.y + self.max_distance * np.sin(angle1)
                    x2 = rsu.position.x + self.max_distance * np.cos(angle2)
                    y2 = rsu.position.y + self.max_distance * np.sin(angle2)
                    self.sumo.polygon.add(
                        f"circle_segment_rsu{rsu.id}_{i}",
                        [(x1, y1), (x2, y2)],
                        color=(255, 0, 0, 255),
                        fill=False,
                        lineWidth=0.2,
                        layer=20,
                    )
        else:
            # check SUMO bin
            sumoBinary = checkBinary("sumo")

            # start sumo no gui (for train)
            self.sumo = libsumo
            libsumo.start(["sumo", "-c", self.cfg_file_path])
            # traci.start([sumoBinary, "-c", cfg_file_path])

        net_boundary = self.sumo.simulation.getNetBoundary()
        self.map_size = net_boundary[1]
        self.sumo_has_init = True

    def _space_init(self):
        neighbor_num = len(self.rsu_network[0])

        # handling_jobs num / all
        # handling_jobs ratio * num / all,
        # queue_connections num / all,
        # connected num /all
        # all neighbor (only) handling_jobs ratio * num / job capicity / neighbor num
        self.local_neighbor_obs_space = spaces.Box(0.0, 1.0, shape=(5,))
        # neighbor rsus:
        # avg handling jobs = ratio * num / all job capicity per rsu
        # avg connections = connection_queue.size() / max size
        # 详细邻居情况（包含自己吗？包含的话0是自己）
        self.global_neighbor_obs_space = spaces.Box(
            0.0, 1.0, shape=((neighbor_num + 1) * 2,)
        )

        # 事实上由于norm的原因，这个越多基本上random越占优
        # 第一个动作，将connections queue的veh迁移至哪个邻居？0或1，box即0-0.49 0.5-1.0
        # 第一个动作可以改为每个rsu的任务分配比例（自己，邻居1，邻居2），这样适合box，
        # 按比例来搞任务分配的话似乎不太行，random太强势了，可以加个任务max分配大小
        # （max分配大小乘以其任务分配比例即该rsu，算能耗时也可以用到）
        # 所以第二个动作就是# 每个connections 的 max分配大小
        # 这样改的话handling_jobs就是个元组（veh, 任务比例）
        # 第3个动作，将handling jobs内的算力资源进行分配，observation需要修改
        # 第4个动作，将connections内的通信资源进行分配
        # 第5个动作, 分配总算力
        # 第6个动作, 缓存策略 math.floor(caching * self.max_content)
        self.box_neighbor_action_space = spaces.Box(
            0.0,
            1.0,
            shape=(
                (neighbor_num + 1) * self.max_connections
                + self.max_connections  # 每个connections 的 max分配大小
                + self.num_cores
                + self.max_connections
                + 1
                + self.max_caching,
            ),
        )

        # 离散化区间
        bins = self.bins

        self.action_space_dims = [
            self.max_connections,  # 动作1: 自己任务分配比例
            self.max_connections,  # 动作2: 邻居任务分配比例，与上数相操作可得
            self.max_connections,  # 动作3: 每个连接的最大分配大小，可不用但是如果不用random会很高？
            self.num_cores,  # 动作4: 算力资源分配
            self.max_connections,  # 动作5: 通信资源分配
            1,  # 动作6: 总算力分配，只需一个动作
            self.max_caching,  # 动作7: 缓存策略，不需要bin因为本来就是离散的
        ]

        # self.action_space_dims = []

        action_space_dims = self.action_space_dims

        self.md_discrete_action_space = spaces.MultiDiscrete(
            [bins] * action_space_dims[0]
            + [bins] * action_space_dims[1]
            + [bins] * action_space_dims[2]
            + [bins] * action_space_dims[3]
            + [bins] * action_space_dims[4]
            + [bins] * action_space_dims[5]
            + [self.max_content] * action_space_dims[6]
        )

    def reset(self, seed=env_config.SEED, options=None):
        self.timestep = 0

        # reset rsus
        self.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.max_connections,
                max_cores=self.num_cores,
            )
            for i in range(self.num_rsus)
        ]

        # reset content
        if not self.content_loaded:
            self._content_init()
            self.content_loaded = True

        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        # reset sumo
        if not self.sumo_has_init:
            self._sumo_init()
            self.sumo_has_init = True

        # step once
        # not sure need or not
        # self.sumo.simulationStep()

        self.vehicle_ids = self.sumo.vehicle.getIDList()

        self.vehicles = {
            vehicle_id: Vehicle(
                vehicle_id,
                self.sumo,
                self.timestep,
                self.cache.get_content(min(self.timestep // self.caching_step, 9)),
            )
            for vehicle_id in self.vehicle_ids
        }

        self._update_connections_queue()

        self.agents = list.copy(self.possible_agents)

        observations = self._beta_update_box_observations(-1)

        self._update_all_rsus_idle()

        infos = {
            a: {"bad_transition": False, "idle": self.rsus[idx].idle}
            for idx, a in enumerate(self.agents)
        }

        terminations = {a: False for idx, a in enumerate(self.agents)}

        infos = {a: {"idle": self.rsus[idx].idle} for idx, a in enumerate(self.agents)}

        return observations, terminations, infos

    def step(self, actions):

        # if env not reset auto, reset before update env
        if not self.sumo_has_init:
            observations, terminations, _ = self.reset()

        # random
        random.seed(self.seed + self.timestep)
        # take action
        if self.is_discrete:
            self._beta_take_actions(actions)
        else:
            self._beta_take_box_actions(actions)
        # caculate rewards
        # dev tag: calculate per timestep? or per fps?
        # calculate frame reward!
        rewards = self._calculate_box_rewards()

        # sumo simulation every 10 time steps
        if self.timestep % self.fps == 0:
            self.sumo.simulationStep()
            # update veh status(position and job type) after sim step
            self._update_vehicles()
            # update connections queue, very important
            self._update_connections_queue()
            # remove deprecated jobs 需要在上面的if里吗还是在外面
            self._update_job_handlings()

        # update observation space
        observations = self._beta_update_box_observations(time_step=self.timestep)
        # time up or sumo done

        truncations = {a: False for a in self.agents}
        infos = {}

        self._update_all_rsus_idle()

        # self.sumo.simulation.getMinExpectedNumber() <= 0
        if self.timestep >= self.max_step:
            # bad transition means real terminal
            terminations = {a: True for a in self.agents}
            infos = {
                a: {"bad_transition": True, "idle": self.rsus[idx].idle}
                for idx, a in enumerate(self.agents)
            }
            print(f"cloud_count:{self.cloud_times}")
            print(f"full_count:{self.full_count}")
            self.sumo.close()
            self.sumo_has_init = False
        else:
            infos = {
                a: {"bad_transition": False, "idle": self.rsus[idx].idle}
                for idx, a in enumerate(self.agents)
            }
            terminations = {a: False for idx, a in enumerate(self.agents)}

        self.timestep += 1

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _update_job_handlings(self):
        # for rsu in self.rsus:
        #     rsu.update_conn_list()

        # 更新所有rsu的handling_jobs，将无效的handling job移除（
        # 例如veh已经离开地图，或者veh已经离开前一个rsu的范围，
        # 这里需要给veh设置一个pre_rsu标识前一个rsu是谁）
        for rsu in self.rsus:
            for tuple_veh in rsu.handling_jobs:
                if tuple_veh is None:
                    continue
                veh, ratio = tuple_veh
                if veh is None:
                    continue
                veh: Vehicle
                if (
                    veh.vehicle_id not in self.vehicle_ids
                    or veh.connected_rsu_id
                    != veh.pre_connected_rsu_id  # 离开上一个rsu范围，不保证正确
                    or veh not in self.connections
                ):
                    # 需不需要往前移（最好不要）？以及邻居是否也要移除该veh
                    # 并没有移除！
                    rsu.remove_job(elem=veh)

                    veh.job_deprocess(self.rsus, self.rsu_network)
        # may not necessary
        # for rsu in self.rsus:
        #     rsu.job_clean()
        ...

    # check_idle的外层循环
    def _update_all_rsus_idle(self):
        # 第一阶段：收集所有 RSU 的邻居更新状态，并更新自身RSU状态
        updates = {}
        for rsu in self.rsus:
            updates[rsu.id] = rsu.check_idle(self.rsus, self.rsu_network)
            self.rsus[rsu.id].idle = updates[rsu.id]["self_idle"]

        ...
        # 第二阶段：统一更新所有 RSU 的邻居状态，但不更新自己的状态
        for rsu_id, update in updates.items():

            for neighbor_id, idle_state in update["neighbors_idle"].items():
                self.rsus[neighbor_id].idle = idle_state

    def _update_vehicles(self):
        current_vehicle_ids = set(self.sumo.vehicle.getIDList())
        previous_vehicle_ids = set(self.vehicle_ids)

        # find new veh in map
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            self.vehicles[vehicle_id] = Vehicle(
                vehicle_id,
                self.sumo,
                self.timestep,
                self.cache.get_content(min(self.timestep // self.caching_step, 9)),
            )

        # find leaving veh
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids

        self.vehicles = {
            veh_ids: vehicle
            for veh_ids, vehicle in self.vehicles.items()
            if veh_ids not in removed_vehicle_ids
        }

        # update vehicle_ids
        self.vehicle_ids = list(current_vehicle_ids)

        # update every veh's position and direction
        for vehicle in self.vehicles.values():
            vehicle.update_pos_direction()
            # dev tag: update content?
            vehicle.update_job_type(
                self.cache.get_content(min(self.timestep // self.caching_step, 9))
            )

        # vehs need pending job
        # self.pending_job_vehicles = [veh for veh in self.vehicles if not veh.job.done()]

    def _beta_take_actions(self, actions):
        def _mig_job(rsu: Rsu, action, idx):
            if rsu.connections_queue.is_empty():
                return

            m_actions_self = action[: self.action_space_dims[0]]
            pre = self.action_space_dims[0]
            m_actions_nb = action[pre : pre + self.action_space_dims[1]]
            pre = pre + self.action_space_dims[1]
            m_actions_job_ratio = action[pre : pre + self.action_space_dims[2]]

            nb_rsu1: Rsu = self.rsus[self.rsu_network[rsu.id][0]]
            nb_rsu2: Rsu = self.rsus[self.rsu_network[rsu.id][1]]

            # 取样，这个list可以改为其他
            for m_idx, ma in enumerate(m_actions_self):

                veh_id: Vehicle = rsu.connections_queue.remove(index=m_idx)

                if veh_id is None:
                    continue

                veh = self.vehicles[veh_id]

                # 防止奖励突变+ 1e-6
                self_ratio = m_actions_self[m_idx] / self.bins + 1e-6
                nb_ratio = m_actions_nb[m_idx] / self.bins + 1e-6
                job_ratio = m_actions_job_ratio[m_idx] / self.bins + 1e-6

                sum_ratio = self_ratio + nb_ratio

                # 0就是不迁移
                if sum_ratio > 0:

                    # 这种都可以靠遍历，如果k>3 需要修改逻辑
                    is_full = [
                        rsu.handling_jobs.is_full(),
                        nb_rsu1.handling_jobs.is_full(),
                        nb_rsu2.handling_jobs.is_full(),
                    ]

                    index_in_rsu = rsu.handling_jobs.index((veh, 0))
                    index_in_nb_rsu1 = nb_rsu1.handling_jobs.index((veh, 0))
                    index_in_nb_rsu2 = nb_rsu2.handling_jobs.index((veh, 0))
                    idxs_in_rsus = [index_in_rsu, index_in_nb_rsu1, index_in_nb_rsu2]
                    in_rsu = [elem is not None for elem in idxs_in_rsus]

                    # 理论上到这里的都是在范围内，可以debug看下是不是
                    # 三个都满了直接cloud
                    if all(is_full):
                        # 都满了，却不在这三个任意一个
                        if not any(in_rsu):
                            veh.is_cloud = True
                            veh.job.is_cloud = True
                            self.cloud_times += 1
                            if not veh.job.processing_rsus.is_empty():
                                for rsu in veh.job.processing_rsus:
                                    if rsu is not None:
                                        rsu.remove_job(veh)
                                veh.job.processing_rsus.clear()
                            continue

                    # 假如已在里面只需调整ratio，理论上到这一步基本不会失败因为至少有个非full
                    # 或者至少在一个里面，但是存在只分配成功一个位置，因此需要最后计算ratio

                    self_ratio = self_ratio / sum_ratio
                    nb_ratio = nb_ratio / sum_ratio

                    mig_rsus = np.array([rsu, nb_rsu1, nb_rsu2])

                    mig_ratio = np.array(
                        [
                            self_ratio,
                            nb_ratio / 2,
                            nb_ratio / 2,
                        ]
                    )
                    # 能到这里说明三个rsu至少一个没空或至少有一个在三个rsu里
                    # is_full 为rsu是否满，in_rsu为是否在里面
                    is_full = np.array(is_full)
                    in_rsu = np.array(in_rsu)
                    # 生成布尔掩码
                    update_mask = in_rsu  # 需要更新的 RSU
                    store_mask = (
                        ~is_full & ~in_rsu
                    )  # 需要存入的 RSU，当且仅当没满且没在rsu里

                    # 合并更新和存入的掩码
                    valid_mask = update_mask | store_mask  # 需要更新或存入的 RSU

                    # 只对满足条件的 RSU 的 mig_ratio 进行归一化
                    if np.any(valid_mask):  # 如果有需要更新或存入的 RSU
                        valid_ratios = mig_ratio[valid_mask]  # 提取满足条件的 mig_ratio
                        total_ratio = np.sum(valid_ratios)  # 计算总和
                        if total_ratio > 0:  # 避免除以零
                            mig_ratio[valid_mask] = valid_ratios / total_ratio  # 归一化
                    else:
                        # 异常情况？
                        self.full_count += 1
                        continue

                    # 更新
                    for u_idx, rsu in enumerate(mig_rsus):
                        if update_mask[u_idx]:
                            rsu: Rsu
                            rsu.handling_jobs[idxs_in_rsus[u_idx]] = (
                                veh,
                                float(mig_ratio[u_idx] * job_ratio),
                            )

                    # 存入
                    for s_idx, rsu in enumerate(mig_rsus):
                        if store_mask[s_idx]:
                            rsu: Rsu
                            # connections有可能爆满
                            veh_disconnect = rsu.connections.queue_jumping(veh)
                            rsu.handling_jobs.append(
                                (veh, float(mig_ratio[s_idx] * job_ratio))
                            )
                            veh.job_process(s_idx, rsu)

                            # 假如veh被断开连接
                            if veh_disconnect is not None:
                                veh_disconnect: Vehicle
                                veh_disconnect.job_deprocess(
                                    self.rsus, self.rsu_network
                                )
                else:
                    # cloud
                    self.cloud_times += 1
                    veh.is_cloud = True
                    veh.job.is_cloud = True
                    if not veh.job.processing_rsus.is_empty():
                        for rsu in veh.job.processing_rsus:
                            if rsu is not None:
                                rsu.remove_job(veh)
                        veh.job.processing_rsus.clear()

                pass

        # env 0
        actions = actions[0]

        # 将 actions 和它们的原始索引组合成元组列表
        indexed_actions = list(enumerate(actions))

        # 随机打乱元组列表或用什么权重方法，
        # 因为如果按顺序后面的车基本不可能能安排到邻居节点
        random.shuffle(indexed_actions)
        num_nb = len(self.rsu_network[0])

        for idx, action in indexed_actions:
            rsu: Rsu = self.rsus[idx]

            if rsu.idle:
                continue

            _mig_job(rsu=rsu, action=action, idx=idx)

        for idx, action in indexed_actions:
            rsu: Rsu = self.rsus[idx]

            if rsu.idle:
                continue

            # resource alloc after all handling
            dims = self.action_space_dims
            pre = sum(dims[:3])
            cp_alloc_actions = np.array(action[pre : pre + dims[3]]) / self.bins
            # print(f"cp_alloc:{cp_alloc_actions}")
            pre = pre + dims[3]
            bw_alloc_actions = np.array(action[pre : pre + dims[4]]) / self.bins
            # print(f"bw_alloc:{bw_alloc_actions}")
            pre = pre + dims[4]
            cp_usage = np.array(action[pre : pre + dims[5]]) / self.bins
            # print(f"cp_usage:{cp_usage}")
            pre = pre + dims[5]

            # 已经转为box了
            rsu.box_alloc_cp(alloc_cp_list=cp_alloc_actions, cp_usage=cp_usage)
            rsu.box_alloc_bw(alloc_bw_list=bw_alloc_actions, veh_ids=self.vehicle_ids)

            # independ caching policy here, 也可以每个时间步都caching
            a = action[pre:]

            rsu.frame_cache_content(a, self.max_content)

    def _beta_update_box_observations(self, time_step):
        observations = {}

        max_handling_job = self.rsus[0].handling_jobs.max_size
        usage_all = 0
        num_handling_job = 0
        count = 0

        if time_step == 141:
            pass

        for idx, a in enumerate(self.agents):
            rsu = self.rsus[idx]

            nb_id1, nb_id2 = self.rsu_network[idx]
            nb_rsus = [self.rsus[nb_id1], self.rsus[nb_id2]]

            nb1_h = nb_rsus[0].handling_jobs.olist
            nb2_h = nb_rsus[1].handling_jobs.olist

            norm_nb1_h = (
                sum([v[1] for v in nb1_h if v is not None])
                / nb_rsus[0].handling_jobs.max_size
            )
            norm_nb2_h = (
                sum([v[1] for v in nb2_h if v is not None])
                / nb_rsus[1].handling_jobs.max_size
            )

            nb1_c = (
                nb_rsus[0].connections_queue.size()
                / nb_rsus[0].connections_queue.max_size
            )
            nb2_c = (
                nb_rsus[1].connections_queue.size()
                / nb_rsus[1].connections_queue.max_size
            )

            norm_nb_h = (norm_nb1_h + norm_nb2_h) / 2

            norm_self_handling = rsu.handling_jobs.size() / rsu.handling_jobs.max_size

            norm_self_handling_ratio = (
                sum([v[1] for v in rsu.handling_jobs.olist if v is not None])
                / rsu.handling_jobs.max_size
            )

            norm_num_conn_queue = (
                rsu.connections_queue.size() / rsu.connections_queue.max_size
            )

            norm_num_connected = rsu.connections.size() / rsu.connections.max_size

            global_obs = (
                [norm_self_handling_ratio]
                + [norm_num_conn_queue]
                + [norm_nb1_h]
                + [nb1_c]
                + [norm_nb2_h]
                + [nb2_c]
            )

            local_obs = (
                [norm_self_handling]
                + [norm_self_handling_ratio]
                + [norm_num_conn_queue]
                + [norm_num_connected]
                + [norm_nb_h]
            )
            # act_mask = self._single_frame_discrete_action_mask(idx, time_step + 1)
            # act_mask = self._single_frame_discrete_action_mask(
            #     self.agents.index(a), time_step + 1
            # )

            observations[a] = {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "action_mask": [],
            }

        return observations

    def _calculate_box_rewards(self):
        rewards = {}

        rsu_qoe_dict, caching_ratio_dict = utility_light.calculate_box_utility(
            vehs=self.vehicles,
            rsus=self.rsus,
            rsu_network=self.rsu_network,
            time_step=self.timestep,
            fps=self.fps,
            weight=self.max_weight,
        )

        self.rsu_qoe_dict = rsu_qoe_dict
        for rid, ratio in caching_ratio_dict.items():
            self.rsus[rid].hit_ratios.append(ratio)

        for idx, agent in enumerate(self.agents):
            # dev tag: factor may need specify
            a = rsu_qoe_dict[idx]

            sum = 0
            if len(a) > 0:
                for r in a:
                    sum += r
                try:
                    flattened = list(itertools.chain.from_iterable(a))
                except TypeError:
                    # 如果展平失败（例如 a 是单个可迭代对象，但不是嵌套的），直接使用 a
                    flattened = list(a)
                rewards[agent] = np.mean(flattened)
            else:
                rewards[agent] = 0.0

        return rewards

    # peformance issue
    def _update_connections_queue(self, kdtree=True):
        """
        connection logic
        """
        # clear connections
        self.connections_queue = []

        # connections update
        for rsu in self.rsus:
            rsu.range_connections.clear()
            rsu.distances.clear()

        for veh_id, veh in self.vehicles.items():
            vehicle_x, vehicle_y = veh.position.x, veh.position.y
            vehicle_coord = np.array([vehicle_x, vehicle_y])

            # 距离排序
            # only connect 1
            distances, sorted_indices = self.rsu_tree.query(
                vehicle_coord, k=len(env_config.RSU_POSITIONS)
            )
            idx = sorted_indices[0]
            dis = distances[0]

            # connected
            if veh.connected_rsu_id is not None:
                veh.pre_connected_rsu_id = veh.connected_rsu_id
                veh.connected_rsu_id = idx
                veh.first_time_caching = True
            else:
                veh.pre_connected_rsu_id = idx
                veh.connected_rsu_id = idx

            veh.distance_to_rsu = dis
            rsu = self.rsus[idx]
            rsu.range_connections.append(veh.vehicle_id)
            rsu.distances.append(dis)

        for rsu in self.rsus:
            rsu.connections_queue.olist = list.copy(rsu.range_connections.olist)
            # disconnect out of range jobs
            # connections里是object，range_connections是id
            for veh in rsu.connections:
                if veh is None:
                    continue
                veh: Vehicle

                if veh.vehicle_id not in rsu.range_connections:
                    # 最好不要shift因为shift会导致遍历问题，但是新策略得要，看看是否有问题
                    rsu.connections.remove_and_shift(veh)

    # improve：返回polygons然后后面统一绘制？
    def _render_connections(self):
        # Batch add polygons
        polygons_to_add = []

        # render QoE
        # not correct now
        for conn in self.connections_queue:
            color = interpolate_color(0, 1, conn.qoe)

            color_with_alpha = (*color, 255)

            polygons_to_add.append(
                (
                    f"dynamic_line_rsu{conn.rsu.id}_to_{conn.veh.vehicle_id}",
                    [
                        (conn.rsu.position.x, conn.rsu.position.y),
                        (conn.veh.position.x, conn.veh.position.y),
                    ],
                    color_with_alpha,
                )
            )

        for polygon_id, points, color in polygons_to_add:
            self.sumo.polygon.add(
                polygon_id, points, color=color, fill=False, lineWidth=0.3, layer=30
            )

    def render(self, mode=None):
        # only render after sim
        if self.timestep % self.fps == 0:
            mode = self.render_mode if mode is None else mode
            # human
            if mode is not None:
                # get veh ID

                # clear all dynamic rendered polygon
                # draw QoE
                for polygon_id in self.sumo.polygon.getIDList():
                    if polygon_id.startswith("dynamic_"):
                        self.sumo.polygon.remove(polygon_id)

                self._render_connections()

                return

        return

    def close(self):
        self.sumo.close()
        self.sumo_has_init = False
        return super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.local_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def global_observation_space(self, agent):
        return self.global_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def local_observation_space(self, agent):
        return self.local_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def multi_discrete_action_space(self, agent):
        return self.md_discrete_action_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.box_neighbor_action_space

    # def get_agent_ids(self):
    #     return self._agent_ids
