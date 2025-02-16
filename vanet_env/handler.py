from abc import ABC, abstractmethod
import itertools
import random
from gymnasium import spaces
import sys

import numpy as np
import pandas as pd
from shapely import Point

from vanet_env import env_config, utility_light
from vanet_env.entites import Vehicle, Rsu

sys.path.append("./")


# env handler, imp or extend to use
class Handler(ABC):
    def __init__(self, env):
        self.env = env
        pass

    # step func here
    @abstractmethod
    def step(self, actions):
        pass

    # reward func here, return reward
    @abstractmethod
    def reward(self):
        pass

    # init action and obs spaces
    @abstractmethod
    def spaces_init(self):
        pass

    # take action logic here
    @abstractmethod
    def take_action(self):
        pass

    @abstractmethod
    def update_observation(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class TrajectoryHandler(Handler):
    def __init__(self, env):
        self.env = env
        self.log_trajectory = False
        if self.log_trajectory:
            self.trajectory_records = []
        pass

    def step(self, actions):
        # if env not reset auto, reset before update env
        if not self.env.sumo_has_init:
            observations, terminations, _ = self.env.reset()

        # random
        random.seed(self.env.seed + self.env.timestep)
        # take action
        if self.env.is_discrete:
            self.take_action(actions)
        else:
            self.env._beta_take_box_actions(actions)
        # caculate rewards
        # dev tag: calculate per timestep? or per fps?
        # calculate frame reward!
        rewards = self.reward()

        # sumo simulation every 10 time steps
        if self.env.timestep % self.env.fps == 0:
            self.env.sumo.simulationStep()
            # update veh status(position and job type) after sim step
            self._update_vehicles()

            if self.log_trajectory:
                for vehicle in self.env.vehicles.values():
                    record = {
                        "real_time": self.env.timestep // self.env.fps,
                        "vehicle_id": vehicle.vehicle_id,
                        "x": vehicle.position.x,
                        "y": vehicle.position.y,
                    }
                    self.trajectory_records.append(record)
            # update connections queue, very important
            self._update_connections_queue()
            # remove deprecated jobs 需要在上面的if里吗还是在外面
            self._update_job_handlings()

            self.env.render()

        # update observation space
        observations = self.update_observation()
        # time up or sumo done

        truncations = {a: False for a in self.env.agents}
        infos = {}

        self._update_all_rsus_idle()

        # self.env.sumo.simulation.getMinExpectedNumber() <= 0
        if self.env.timestep >= self.env.max_step:
            # bad transition means real terminal
            terminations = {a: True for a in self.env.agents}
            infos = {
                a: {"bad_transition": True, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            print(f"cloud_count:{self.env.cloud_times}")
            print(f"full_count:{self.env.full_count}")
            self.env.sumo.close()
            self.env.sumo_has_init = False
            if self.log_trajectory:
                df = pd.DataFrame(self.trajectory_records)
                df.to_csv("trajectory_log.csv", index=False)
                print("轨迹数据已保存到 trajectory_log.csv")
        else:
            infos = {
                a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            terminations = {a: False for idx, a in enumerate(self.env.agents)}

        self.env.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed):
        self.env.timestep = 0

        # reset rsus
        self.env.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.env.max_connections,
                max_cores=self.env.num_cores,
            )
            for i in range(self.env.num_rsus)
        ]

        # reset content
        if not self.env.content_loaded:
            self.env._content_init()
            self.env.content_loaded = True

        np.random.seed(seed)
        random.seed(seed)
        self.env.seed = seed

        # reset sumo
        if not self.env.sumo_has_init:
            self.env._sumo_init()
            self.env.sumo_has_init = True

        # step once
        # not sure need or not
        # self.env.sumo.simulationStep()

        self.env.vehicle_ids = self.env.sumo.vehicle.getIDList()

        self.env.vehicles = {
            vehicle_id: Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )
            for vehicle_id in self.env.vehicle_ids
        }

        self._update_connections_queue()

        self.env.agents = list.copy(self.env.possible_agents)

        observations = self.update_observation()

        self._update_all_rsus_idle()

        infos = {
            a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        terminations = {a: False for idx, a in enumerate(self.env.agents)}

        infos = {
            a: {"idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        return observations, terminations, infos

    def reward(self):
        rewards = {}
        reward_per_agent = []

        rsu_qoe_dict, caching_ratio_dict = utility_light.fixed_calculate_utility(
            vehs=self.env.vehicles,
            rsus=self.env.rsus,
            rsu_network=self.env.rsu_network,
            time_step=self.env.timestep,
            fps=self.env.fps,
            weight=self.env.max_weight,
        )

        self.env.rsu_qoe_dict = rsu_qoe_dict

        for rid, ratio in caching_ratio_dict.items():
            self.env.rsus[rid].hit_ratios.append(ratio)

        for idx, agent in enumerate(self.env.agents):
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

        for idx, agent in enumerate(self.env.agents):

            if not self.env.rsus[idx].idle:
                reward_per_agent.append(rewards[agent])

        self.env.ava_rewards.append(np.mean(reward_per_agent))

        return rewards
        pass

    # return obs space and action space
    def spaces_init(self):
        neighbor_num = len(self.env.rsu_network[0])

        # handling_jobs num / all
        # handling_jobs ratio * num / all,
        # queue_connections num / all,
        # connected num /all
        # all neighbor (only) handling_jobs ratio * num / job capicity / neighbor num
        self.env.local_neighbor_obs_space = spaces.Box(0.0, 1.0, shape=(5,))
        # neighbor rsus:
        # avg handling jobs = ratio * num / all job capicity per rsu
        # avg connections = connection_queue.size() / max size
        # 详细邻居情况（包含自己吗？包含的话0是自己）
        self.env.global_neighbor_obs_space = spaces.Box(
            0.0, 1.0, shape=((neighbor_num + 1) * 2,)
        )

        self.env.veh_locations = []

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
        # 第6个动作, 缓存策略 math.floor(caching * self.env.max_content)
        self.env.box_neighbor_action_space = spaces.Box(
            0.0,
            1.0,
            shape=(
                (neighbor_num + 1) * self.env.max_connections
                + self.env.max_connections  # 每个connections 的 max分配大小
                + self.env.num_cores
                + self.env.max_connections
                + 1
                + self.env.max_caching,
            ),
        )

        # 离散化区间
        bins = self.env.bins

        self.env.action_space_dims = [
            self.env.max_connections,  # 动作1: 自己任务分配比例
            self.env.max_connections,  # 动作2: 邻居任务分配比例，与上数相操作可得
            self.env.max_connections,  # 动作3: 每个连接的最大分配大小，可不用但是如果不用random会很高？
            self.env.num_cores,  # 动作4: 算力资源分配
            self.env.max_connections,  # 动作5: 通信资源分配
            1,  # 动作6: 总算力分配，只需一个动作
            self.env.max_caching,  # 动作7: 缓存策略，不需要bin因为本来就是离散的
        ]

        # self.env.action_space_dims = []

        action_space_dims = self.env.action_space_dims

        self.env.md_discrete_action_space = spaces.MultiDiscrete(
            [bins] * action_space_dims[0]
            + [bins] * action_space_dims[1]
            + [bins] * action_space_dims[2]
            + [bins] * action_space_dims[3]
            + [bins] * action_space_dims[4]
            + [bins] * action_space_dims[5]
            + [self.env.max_content] * action_space_dims[6]
        )
        pass

    def take_action(self, actions):
        def _mig_job(rsu: Rsu, action, idx):
            if rsu.connections_queue.is_empty():
                return

            m_actions_self = action[: self.env.action_space_dims[0]]
            pre = self.env.action_space_dims[0]
            m_actions_nb = action[pre : pre + self.env.action_space_dims[1]]
            pre = pre + self.env.action_space_dims[1]
            m_actions_job_ratio = action[pre : pre + self.env.action_space_dims[2]]

            nb_rsu1: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][0]]
            nb_rsu2: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][1]]

            # 取样，这个list可以改为其他
            # 如果取样的是connections_queue，那需要注意prase idx
            for m_idx, ma in enumerate(rsu.connections_queue):

                veh_id: Vehicle = rsu.connections_queue.remove(index=m_idx)

                if veh_id is None:
                    continue

                veh = self.env.vehicles[veh_id]
                real_m_idx = m_idx % self.env.max_connections
                # 防止奖励突变+ 1e-6
                self_ratio = m_actions_self[real_m_idx] / self.env.bins + 1e-6
                nb_ratio = m_actions_nb[real_m_idx] / self.env.bins + 1e-6
                job_ratio = m_actions_job_ratio[real_m_idx] / self.env.bins + 1e-6

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
                            self.env.cloud_times += 1
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
                        self.env.full_count += 1
                        continue

                    # 更新
                    for u_idx, u_rsu in enumerate(mig_rsus):
                        if update_mask[u_idx]:
                            u_rsu: Rsu
                            u_rsu.handling_jobs[idxs_in_rsus[u_idx]] = (
                                veh,
                                float(mig_ratio[u_idx] * job_ratio),
                            )

                    # 存入
                    for s_idx, s_rsu in enumerate(mig_rsus):
                        if store_mask[s_idx]:
                            s_rsu: Rsu
                            veh_disconnect: Vehicle = None
                            # connections有可能爆满
                            if veh not in rsu.connections:
                                # 会不会重复connection？
                                # 2.1 重复connection逻辑 fix！改为append！
                                veh_disconnect = rsu.connections.append_and_out(veh)
                                # rsu.connections.append(veh)

                            s_rsu.handling_jobs.append(
                                (veh, float(mig_ratio[s_idx] * job_ratio))
                            )
                            veh.job_process(s_idx, s_rsu)

                            # 假如有veh被断开连接
                            if veh_disconnect is not None:
                                veh_disconnect: Vehicle
                                veh_disconnect.job_deprocess(
                                    self.env.rsus, self.env.rsu_network
                                )
                else:
                    # cloud
                    self.env.cloud_times += 1
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
        num_nb = len(self.env.rsu_network[0])

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            _mig_job(rsu=rsu, action=action, idx=idx)

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            # resource alloc after all handling
            dims = self.env.action_space_dims
            pre = sum(dims[:3])
            cp_alloc_actions = np.array(action[pre : pre + dims[3]]) / self.env.bins
            # print(f"cp_alloc:{cp_alloc_actions}")
            pre = pre + dims[3]
            bw_alloc_actions = np.array(action[pre : pre + dims[4]]) / self.env.bins
            # print(f"bw_alloc:{bw_alloc_actions}")
            pre = pre + dims[4]
            cp_usage = np.array(action[pre : pre + dims[5]]) / self.env.bins
            # print(f"cp_usage:{cp_usage}")
            pre = pre + dims[5]

            # 已经转为box了
            rsu.box_alloc_cp(alloc_cp_list=cp_alloc_actions, cp_usage=cp_usage)
            rsu.box_alloc_bw(
                alloc_bw_list=bw_alloc_actions, veh_ids=self.env.vehicle_ids
            )

            # independ caching policy here, 也可以每个时间步都caching
            a = action[pre:]

            rsu.frame_cache_content(a, self.env.max_content)

        pass

    def update_observation(self):
        observations = {}

        max_handling_job = self.env.rsus[0].handling_jobs.max_size
        usage_all = 0
        num_handling_job = 0
        count = 0

        for idx, a in enumerate(self.env.agents):
            rsu = self.env.rsus[idx]

            nb_id1, nb_id2 = self.env.rsu_network[idx]
            nb_rsus = [self.env.rsus[nb_id1], self.env.rsus[nb_id2]]

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
            # act_mask = self.env._single_frame_discrete_action_mask(idx, time_step + 1)
            # act_mask = self.env._single_frame_discrete_action_mask(
            #     self.env.agents.index(a), time_step + 1
            # )

            observations[a] = {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "action_mask": [],
            }

        return observations
        pass

    def _update_job_handlings(self):
        # for rsu in self.env.rsus:
        #     rsu.update_conn_list()

        # 更新所有rsu的handling_jobs，将无效的handling job移除（
        # 例如veh已经离开地图，或者veh已经离开前一个rsu的范围，
        # 这里需要给veh设置一个pre_rsu标识前一个rsu是谁）
        for rsu in self.env.rsus:
            rsu.pre_handling_jobs.olist = list.copy(rsu.handling_jobs.olist)
            for tuple_veh in rsu.handling_jobs:
                if tuple_veh is None:
                    continue
                veh, ratio = tuple_veh
                if veh is None:
                    continue
                veh: Vehicle
                if (
                    veh.vehicle_id not in self.env.vehicle_ids  # 离开地图
                    or veh not in self.env.connections
                ):
                    # 需不需要往前移（最好不要）？以及邻居是否也要移除该veh
                    # 并没有移除！
                    rsu.remove_job(elem=veh)

                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
                if (
                    veh.connected_rsu_id != veh.pre_connected_rsu_id
                ):  # 离开上一个rsu范围，不保证正确
                    self.env.rsus[veh.pre_connected_rsu_id].remove_job(elem=veh)
                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
        # may not necessary
        # for rsu in self.rsus:
        #     rsu.job_clean()

    # check_idle的外层循环
    def _update_all_rsus_idle(self):
        # 第一阶段：收集所有 RSU 的邻居更新状态，并更新自身RSU状态
        updates = {}
        for rsu in self.env.rsus:
            updates[rsu.id] = rsu.check_idle(self.env.rsus, self.env.rsu_network)
            self.env.rsus[rsu.id].idle = updates[rsu.id]["self_idle"]

        ...
        # 第二阶段：统一更新所有 RSU 的邻居状态，但不更新自己的状态
        for rsu_id, update in updates.items():

            for neighbor_id, idle_state in update["neighbors_idle"].items():
                self.env.rsus[neighbor_id].idle = idle_state

    def _update_vehicles(self):
        current_vehicle_ids = set(self.env.sumo.vehicle.getIDList())
        previous_vehicle_ids = set(self.env.vehicle_ids)

        # find new veh in map
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            self.env.vehicles[vehicle_id] = Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )

        # find leaving veh
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids

        self.env.vehicles = {
            veh_ids: vehicle
            for veh_ids, vehicle in self.env.vehicles.items()
            if veh_ids not in removed_vehicle_ids
        }

        # update vehicle_ids
        self.env.vehicle_ids = list(current_vehicle_ids)

        # update every veh's position and direction
        for vehicle in self.env.vehicles.values():
            vehicle.update_pos_direction()
            # dev tag: update content?
            vehicle.update_job_type(
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                )
            )

        # vehs need pending job
        # self.pending_job_vehicles = [veh for veh in self.vehicles if not veh.job.done()]

    def _update_connections_queue(self):
        """
        connection logic
        """
        # clear connections
        self.env.connections_queue = []

        # connections update
        for rsu in self.env.rsus:
            rsu.range_connections.clear()
            rsu.distances.clear()

        for veh_id, veh in self.env.vehicles.items():
            vehicle_x, vehicle_y = veh.position.x, veh.position.y
            vehicle_coord = np.array([vehicle_x, vehicle_y])

            # 距离排序
            # only connect 1
            distances, sorted_indices = self.env.rsu_tree.query(
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
            rsu = self.env.rsus[idx]
            rsu.range_connections.append(veh.vehicle_id)
            rsu.distances.append(dis)

        for rsu in self.env.rsus:
            rsu.connections_queue.olist = list.copy(rsu.range_connections.olist)
            # disconnect out of range jobs
            # connections里是object，range_connections是id
            for veh in rsu.connections:

                if veh is None:
                    continue
                veh: Vehicle
                # 车辆已离开
                if (
                    veh.vehicle_id not in self.env.vehicle_ids
                    or veh.vehicle_id not in rsu.range_connections
                    or veh.connected_rsu_id != rsu.id
                ):
                    # veh296不知道为什么不会被移除？
                    if veh.vehicle_id == "veh296":
                        ...
                    rsu.connections.remove(veh)


class MappoHandler(Handler):
    def __init__(self, env):
        self.env = env
        pass

    def step(self, actions):
        # if env not reset auto, reset before update env
        if not self.env.sumo_has_init:
            observations, terminations, _ = self.env.reset()

        # random
        random.seed(self.env.seed + self.env.timestep)
        # take action
        if self.env.is_discrete:
            self.take_action(actions)
        else:
            self.env._beta_take_box_actions(actions)
        # caculate rewards
        # dev tag: calculate per timestep? or per fps?
        # calculate frame reward!
        rewards = self.reward()

        # sumo simulation every 10 time steps
        if self.env.timestep % self.env.fps == 0:
            self.env.sumo.simulationStep()
            # update veh status(position and job type) after sim step
            self._update_vehicles()
            # update connections queue, very important
            self._update_connections_queue()
            # remove deprecated jobs 需要在上面的if里吗还是在外面
            self._update_job_handlings()

            self.env.render()

        # update observation space
        observations = self.update_observation()
        # time up or sumo done

        truncations = {a: False for a in self.env.agents}
        infos = {}

        self._update_all_rsus_idle()

        # self.env.sumo.simulation.getMinExpectedNumber() <= 0
        if self.env.timestep >= self.env.max_step:
            # bad transition means real terminal
            terminations = {a: True for a in self.env.agents}
            infos = {
                a: {"bad_transition": True, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            print(f"cloud_count:{self.env.cloud_times}")
            print(f"full_count:{self.env.full_count}")
            self.env.sumo.close()
            self.env.sumo_has_init = False
        else:
            infos = {
                a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
                for idx, a in enumerate(self.env.agents)
            }
            terminations = {a: False for idx, a in enumerate(self.env.agents)}

        self.env.timestep += 1

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed):
        self.env.timestep = 0

        # reset rsus
        self.env.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.env.max_connections,
                max_cores=self.env.num_cores,
            )
            for i in range(self.env.num_rsus)
        ]

        # reset content
        if not self.env.content_loaded:
            self.env._content_init()
            self.env.content_loaded = True

        np.random.seed(seed)
        random.seed(seed)
        self.env.seed = seed

        # reset sumo
        if not self.env.sumo_has_init:
            self.env._sumo_init()
            self.env.sumo_has_init = True

        # step once
        # not sure need or not
        # self.env.sumo.simulationStep()

        self.env.vehicle_ids = self.env.sumo.vehicle.getIDList()

        self.env.vehicles = {
            vehicle_id: Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )
            for vehicle_id in self.env.vehicle_ids
        }

        self._update_connections_queue()

        self.env.agents = list.copy(self.env.possible_agents)

        observations = self.update_observation()

        self._update_all_rsus_idle()

        infos = {
            a: {"bad_transition": False, "idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        terminations = {a: False for idx, a in enumerate(self.env.agents)}

        infos = {
            a: {"idle": self.env.rsus[idx].idle}
            for idx, a in enumerate(self.env.agents)
        }

        return observations, terminations, infos

    def reward(self):
        rewards = {}
        reward_per_agent = []

        rsu_qoe_dict, caching_ratio_dict = utility_light.calculate_box_utility(
            vehs=self.env.vehicles,
            rsus=self.env.rsus,
            rsu_network=self.env.rsu_network,
            time_step=self.env.timestep,
            fps=self.env.fps,
            weight=self.env.max_weight,
        )

        self.env.rsu_qoe_dict = rsu_qoe_dict

        for rid, ratio in caching_ratio_dict.items():
            self.env.rsus[rid].hit_ratios.append(ratio)

        for idx, agent in enumerate(self.env.agents):
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

        for idx, agent in enumerate(self.env.agents):

            if not self.env.rsus[idx].idle:
                reward_per_agent.append(rewards[agent])

        self.env.ava_rewards.append(np.mean(reward_per_agent))

        return rewards
        pass

    # return obs space and action space
    def spaces_init(self):
        neighbor_num = len(self.env.rsu_network[0])

        # handling_jobs num / all
        # handling_jobs ratio * num / all,
        # queue_connections num / all,
        # connected num /all
        # all neighbor (only) handling_jobs ratio * num / job capicity / neighbor num
        self.env.local_neighbor_obs_space = spaces.Box(0.0, 1.0, shape=(5,))
        # neighbor rsus:
        # avg handling jobs = ratio * num / all job capicity per rsu
        # avg connections = connection_queue.size() / max size
        # 详细邻居情况（包含自己吗？包含的话0是自己）
        self.env.global_neighbor_obs_space = spaces.Box(
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
        # 第6个动作, 缓存策略 math.floor(caching * self.env.max_content)
        self.env.box_neighbor_action_space = spaces.Box(
            0.0,
            1.0,
            shape=(
                (neighbor_num + 1) * self.env.max_connections
                + self.env.max_connections  # 每个connections 的 max分配大小
                + self.env.num_cores
                + self.env.max_connections
                + 1
                + self.env.max_caching,
            ),
        )

        # 离散化区间
        bins = self.env.bins

        self.env.action_space_dims = [
            self.env.max_connections,  # 动作1: 自己任务分配比例
            self.env.max_connections,  # 动作2: 邻居任务分配比例，与上数相操作可得
            self.env.max_connections,  # 动作3: 每个连接的最大分配大小，可不用但是如果不用random会很高？
            self.env.num_cores,  # 动作4: 算力资源分配
            self.env.max_connections,  # 动作5: 通信资源分配
            1,  # 动作6: 总算力分配，只需一个动作
            self.env.max_caching,  # 动作7: 缓存策略，不需要bin因为本来就是离散的
        ]

        # self.env.action_space_dims = []

        action_space_dims = self.env.action_space_dims

        self.env.md_discrete_action_space = spaces.MultiDiscrete(
            [bins] * action_space_dims[0]
            + [bins] * action_space_dims[1]
            + [bins] * action_space_dims[2]
            + [bins] * action_space_dims[3]
            + [bins] * action_space_dims[4]
            + [bins] * action_space_dims[5]
            + [self.env.max_content] * action_space_dims[6]
        )
        pass

    def take_action(self, actions):
        def _mig_job(rsu: Rsu, action, idx):
            if rsu.connections_queue.is_empty():
                return

            m_actions_self = action[: self.env.action_space_dims[0]]
            pre = self.env.action_space_dims[0]
            m_actions_nb = action[pre : pre + self.env.action_space_dims[1]]
            pre = pre + self.env.action_space_dims[1]
            m_actions_job_ratio = action[pre : pre + self.env.action_space_dims[2]]

            nb_rsu1: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][0]]
            nb_rsu2: Rsu = self.env.rsus[self.env.rsu_network[rsu.id][1]]

            # 取样，这个list可以改为其他
            # 如果取样的是connections_queue，那需要注意prase idx
            for m_idx, ma in enumerate(rsu.connections_queue):

                veh_id: Vehicle = rsu.connections_queue.remove(index=m_idx)

                if veh_id is None:
                    continue

                veh = self.env.vehicles[veh_id]
                real_m_idx = m_idx % self.env.max_connections
                # 防止奖励突变+ 1e-6
                self_ratio = m_actions_self[real_m_idx] / self.env.bins + 1e-6
                nb_ratio = m_actions_nb[real_m_idx] / self.env.bins + 1e-6
                job_ratio = m_actions_job_ratio[real_m_idx] / self.env.bins + 1e-6

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
                            self.env.cloud_times += 1
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
                        self.env.full_count += 1
                        continue

                    # 更新
                    for u_idx, u_rsu in enumerate(mig_rsus):
                        if update_mask[u_idx]:
                            u_rsu: Rsu
                            u_rsu.handling_jobs[idxs_in_rsus[u_idx]] = (
                                veh,
                                float(mig_ratio[u_idx] * job_ratio),
                            )

                    # 存入
                    for s_idx, s_rsu in enumerate(mig_rsus):
                        if store_mask[s_idx]:
                            s_rsu: Rsu
                            veh_disconnect: Vehicle = None
                            # connections有可能爆满
                            if veh not in rsu.connections:
                                # 会不会重复connection？
                                # 2.1 重复connection逻辑 fix！改为append！
                                veh_disconnect = rsu.connections.append_and_out(veh)
                                # rsu.connections.append(veh)

                            s_rsu.handling_jobs.append(
                                (veh, float(mig_ratio[s_idx] * job_ratio))
                            )
                            veh.job_process(s_idx, s_rsu)

                            # 假如有veh被断开连接
                            if veh_disconnect is not None:
                                veh_disconnect: Vehicle
                                veh_disconnect.job_deprocess(
                                    self.env.rsus, self.env.rsu_network
                                )
                else:
                    # cloud
                    self.env.cloud_times += 1
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
        num_nb = len(self.env.rsu_network[0])

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            _mig_job(rsu=rsu, action=action, idx=idx)

        for idx, action in indexed_actions:
            rsu: Rsu = self.env.rsus[idx]

            if rsu.idle:
                continue

            # resource alloc after all handling
            dims = self.env.action_space_dims
            pre = sum(dims[:3])
            cp_alloc_actions = np.array(action[pre : pre + dims[3]]) / self.env.bins
            # print(f"cp_alloc:{cp_alloc_actions}")
            pre = pre + dims[3]
            bw_alloc_actions = np.array(action[pre : pre + dims[4]]) / self.env.bins
            # print(f"bw_alloc:{bw_alloc_actions}")
            pre = pre + dims[4]
            cp_usage = np.array(action[pre : pre + dims[5]]) / self.env.bins
            # print(f"cp_usage:{cp_usage}")
            pre = pre + dims[5]

            # 已经转为box了
            rsu.box_alloc_cp(alloc_cp_list=cp_alloc_actions, cp_usage=cp_usage)
            rsu.box_alloc_bw(
                alloc_bw_list=bw_alloc_actions, veh_ids=self.env.vehicle_ids
            )

            # independ caching policy here, 也可以每个时间步都caching
            a = action[pre:]

            rsu.frame_cache_content(a, self.env.max_content)

        pass

    def update_observation(self):
        observations = {}

        max_handling_job = self.env.rsus[0].handling_jobs.max_size
        usage_all = 0
        num_handling_job = 0
        count = 0

        for idx, a in enumerate(self.env.agents):
            rsu = self.env.rsus[idx]

            nb_id1, nb_id2 = self.env.rsu_network[idx]
            nb_rsus = [self.env.rsus[nb_id1], self.env.rsus[nb_id2]]

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
            # act_mask = self.env._single_frame_discrete_action_mask(idx, time_step + 1)
            # act_mask = self.env._single_frame_discrete_action_mask(
            #     self.env.agents.index(a), time_step + 1
            # )

            observations[a] = {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "action_mask": [],
            }

        return observations
        pass

    def _update_job_handlings(self):
        # for rsu in self.env.rsus:
        #     rsu.update_conn_list()

        # 更新所有rsu的handling_jobs，将无效的handling job移除（
        # 例如veh已经离开地图，或者veh已经离开前一个rsu的范围，
        # 这里需要给veh设置一个pre_rsu标识前一个rsu是谁）
        for rsu in self.env.rsus:
            rsu.pre_handling_jobs.olist = list.copy(rsu.handling_jobs.olist)
            for tuple_veh in rsu.handling_jobs:
                if tuple_veh is None:
                    continue
                veh, ratio = tuple_veh
                if veh is None:
                    continue
                veh: Vehicle
                if (
                    veh.vehicle_id not in self.env.vehicle_ids  # 离开地图
                    or veh not in self.env.connections
                ):
                    # 需不需要往前移（最好不要）？以及邻居是否也要移除该veh
                    # 并没有移除！
                    rsu.remove_job(elem=veh)

                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
                if (
                    veh.connected_rsu_id != veh.pre_connected_rsu_id
                ):  # 离开上一个rsu范围，不保证正确
                    self.env.rsus[veh.pre_connected_rsu_id].remove_job(elem=veh)
                    veh.job_deprocess(self.env.rsus, self.env.rsu_network)
        # may not necessary
        # for rsu in self.rsus:
        #     rsu.job_clean()

    # check_idle的外层循环
    def _update_all_rsus_idle(self):
        # 第一阶段：收集所有 RSU 的邻居更新状态，并更新自身RSU状态
        updates = {}
        for rsu in self.env.rsus:
            updates[rsu.id] = rsu.check_idle(self.env.rsus, self.env.rsu_network)
            self.env.rsus[rsu.id].idle = updates[rsu.id]["self_idle"]

        ...
        # 第二阶段：统一更新所有 RSU 的邻居状态，但不更新自己的状态
        for rsu_id, update in updates.items():

            for neighbor_id, idle_state in update["neighbors_idle"].items():
                self.env.rsus[neighbor_id].idle = idle_state

    def _update_vehicles(self):
        current_vehicle_ids = set(self.env.sumo.vehicle.getIDList())
        previous_vehicle_ids = set(self.env.vehicle_ids)

        # find new veh in map
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            self.env.vehicles[vehicle_id] = Vehicle(
                vehicle_id,
                self.env.sumo,
                self.env.timestep,
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                ),
            )

        # find leaving veh
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids

        self.env.vehicles = {
            veh_ids: vehicle
            for veh_ids, vehicle in self.env.vehicles.items()
            if veh_ids not in removed_vehicle_ids
        }

        # update vehicle_ids
        self.env.vehicle_ids = list(current_vehicle_ids)

        # update every veh's position and direction
        for vehicle in self.env.vehicles.values():
            vehicle.update_pos_direction()
            # dev tag: update content?
            vehicle.update_job_type(
                self.env.cache.get_content(
                    min(self.env.timestep // self.env.caching_step, 9)
                )
            )

        # vehs need pending job
        # self.pending_job_vehicles = [veh for veh in self.vehicles if not veh.job.done()]

    def _update_connections_queue(self):
        """
        connection logic
        """
        # clear connections
        self.env.connections_queue = []

        # connections update
        for rsu in self.env.rsus:
            rsu.range_connections.clear()
            rsu.distances.clear()

        for veh_id, veh in self.env.vehicles.items():
            vehicle_x, vehicle_y = veh.position.x, veh.position.y
            vehicle_coord = np.array([vehicle_x, vehicle_y])

            # 距离排序
            # only connect 1
            distances, sorted_indices = self.env.rsu_tree.query(
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
            rsu = self.env.rsus[idx]
            rsu.range_connections.append(veh.vehicle_id)
            rsu.distances.append(dis)

        for rsu in self.env.rsus:
            rsu.connections_queue.olist = list.copy(rsu.range_connections.olist)
            # disconnect out of range jobs
            # connections里是object，range_connections是id
            for veh in rsu.connections:

                if veh is None:
                    continue
                veh: Vehicle
                # 车辆已离开
                if (
                    veh.vehicle_id not in self.env.vehicle_ids
                    or veh.vehicle_id not in rsu.range_connections
                    or veh.connected_rsu_id != rsu.id
                ):
                    # veh296不知道为什么不会被移除？
                    if veh.vehicle_id == "veh296":
                        ...
                    rsu.connections.remove(veh)
