import math
import random
import sys
from typing import List

from shapely import Point

from vanet_env import env_config
import traci

sys.path.append("./")

import numpy as np
from vanet_env import utils


# rectange, unit (meters)
# class Road:
#     def __init__(self, x, y, width=20, ):
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height


class OrderedQueueList:
    def __init__(self, max_size, init_num=None):

        self.max_size = max_size
        if init_num is not None:
            self.olist = [init_num] * max_size
        else:
            self.olist = [None] * max_size

    def queue_jumping(self, elem):
        """
        Insert the element into the head of the queue
        for self job handling
        """
        t = self.olist[-1]
        self.olist = [elem] + self.olist[:-1]
        return t
        pass

    # Queuing
    def append(self, elem):
        for i in range(self.max_size):
            if self.olist[i] is None:
                self.olist[i] = elem
                return True
        return False

    def replace(self, elem, index):
        temp = self.olist[index]
        self.olist[index] = elem
        return temp

    def top_and_sink(self):
        top = self.olist[0]
        self.olist = self.olist[1:] + [top]
        return top

    def pop(self):
        top = self.olist[0]
        self.olist = self.olist[1:] + [None]
        return top

    def remove(self, elem=None, index=-1):
        if elem is not None:
            self.remove(index=self.index(elem))
        else:
            if index == None:
                return None

            if 0 <= index < self.max_size:
                t = self.olist[index]
                self.olist[index] = None
                return t
            else:
                assert IndexError("Index out of range")

    def remove_and_shift(self, elem=None, index=-1):
        """
        better not use, shift may have some iter issue
        """
        if elem is not None:
            self.remove_and_shift(index=self.index(elem))
        else:
            if index == None:
                return None
                assert IndexError("Index out of range")

            if 0 <= index < self.max_size:
                t = self.olist[index]

                # 将 index 之后的元素向前移动
                while index + 1 < self.max_size:
                    self.olist[index] = self.olist[index + 1]
                    index += 1

                self.olist[self.max_size - 1] = None  # 最后一个位置设置为 None

                return t
            else:
                assert IndexError("Index out of range")

    def to_list_replace_none(self):
        return [0 if x is None else x for x in self.olist]

    def size(self):
        return sum(1 for elem in self.olist if elem is not None)

    def is_full(self):
        return self.size() >= self.max_size

    def is_empty(self):
        return self.size() == 0

    def avg(self):
        """
        bug
        """
        filtered_values = [v for v in self.olist if v is not None]
        if filtered_values:
            return self.sum() / len(filtered_values)
        else:
            return 0

    def sum(self):
        filtered_values = [v for v in self.olist if v is not None]
        return sum(filtered_values)

    def __iter__(self):
        return iter(self.olist)

    def __str__(self):
        return str(self.olist)

    def __getitem__(self, index):
        if index < self.max_size:
            return self.olist[index]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if 0 <= index < self.max_size:
            self.olist[index] = value
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        """返回队列的最大容量"""
        return self.max_size

    def __contains__(self, item):
        """支持 in 操作符"""
        if isinstance(item, tuple):
            # 找到第一个匹配 elem[0] 的元组的索引
            for i, elem in enumerate(self.olist):
                if isinstance(elem, tuple) and elem[0] == item[0]:
                    return True

            return False

        if isinstance(item, np.ndarray):  # 如果 item 是 NumPy 数组
            return any(np.array_equal(item, x) for x in self.olist if x is not None)
        else:  # 如果 item 是普通值
            return item in self.olist

    def clear(self):
        self.olist = [None] * self.max_size

    def index(self, elem):
        if isinstance(elem, tuple):
            # 找到第一个匹配 elem[0] 的元组的索引
            for i, item in enumerate(self.olist):
                if isinstance(item, tuple) and item[0] == elem[0]:
                    return i
            return None

        return self.olist.index(elem)


# a = OrderedQueueList(5)
# a.append((1, "a"))
# a.append((2, "a"))
# a.append((3, "a"))
# a.append((4, "a"))
# a.append((5, "a"))
# a.remove_and_shift(elem=(2, 0))
# a.remove_and_shift(elem=(4, 0))
# # 不存在的话报错或者return none
# a.remove_and_shift(elem=(6, 0))
# a.remove(elem=(3, 1))
# print(a)
# b = OrderedQueueList(5)
# b.append(0)
# b.append(1)
# b.append(2)
# b.remove(2)
# b.remove_and_shift(elem=0)
# print(b)


class Rsu:
    def __init__(
        self,
        id,
        position: Point,
        bw=env_config.RSU_MAX_TRANSMITTED_BANDWIDTH,
        frequency=env_config.RSU_FREQUENCY,
        transmitted_power=env_config.RSU_TRANSMITTED_POWER,
        height=env_config.RSU_ANTENNA_HEIGHT,
        noise_power=env_config.RSU_NOISE_POWER,
        snr_threshold=env_config.RSU_SNR_THRESHOLD,
        computation_power=env_config.RSU_COMPUTATION_POWER,
        caching_capacity=env_config.RSU_CACHING_CAPACITY,
        num_atn=env_config.RSU_NUM_ANTENNA,
        tx_gain=env_config.ANTENNA_GAIN,
        max_connections=env_config.MAX_CONNECTIONS,
        max_cores=env_config.NUM_CORES,
    ):
        # args
        self.id = id
        self.position = position
        self.bw = bw
        self.frequency = frequency
        self.transmitted_power = transmitted_power
        self.noise_power = noise_power
        self.height = height
        self.computation_power = computation_power
        self.caching_capacity = caching_capacity
        self.snr_threshold = snr_threshold
        self.tx_gain = tx_gain
        self.max_connections = max_connections
        self.max_cores = max_cores
        self.num_atn = num_atn
        # if idle not doing anything
        self.idle = False

        # distance to vehs
        self.distances = OrderedQueueList(max_connections)
        # copy from range_connections when update connections
        self.connections_queue = OrderedQueueList(max_connections)
        # conn in range, not modify only if update connections
        self.range_connections = OrderedQueueList(max_connections)
        self.connections = OrderedQueueList(max_connections)
        self.handling_job_queue = OrderedQueueList(max_cores)  # may not necessary
        self.handling_jobs = OrderedQueueList(max_cores)
        self.bw_alloc = OrderedQueueList(max_connections)
        self.computation_power_alloc = OrderedQueueList(max_cores)
        self.real_cp_alloc = OrderedQueueList(max_cores)
        self.caching_contents = OrderedQueueList(caching_capacity)

        self.energy_efficiency = 0

        # cp_usage max is weight
        self.cp_usage = 10
        self.bw_ratio = 5
        self.tx_ratio = 100

        self.cp_norm = [0] * self.handling_jobs.max_size
        self.bw_norm = [0] * self.connections.max_size

        self.ee = 0
        self.max_ee = 3
        self.utility = 0

        self.qoe_list = []
        self.avg_u = 0

    def get_tx_power(self):
        return self.transmitted_power * self.tx_ratio / 100 + self.tx_gain

    def remove_job(self, elem):
        if isinstance(elem, tuple):
            self.handling_jobs.remove(elem)
        else:
            self.handling_jobs.remove((elem, 0))

    def box_alloc_cp(self, alloc_cp_list, cp_usage):
        # 0 - 1
        self.cp_usage = cp_usage
        self.computation_power_alloc.olist = list.copy(alloc_cp_list.tolist())

        ava_alloc = []

        for idx, veh_info in enumerate(self.handling_jobs):
            if veh_info is not None:
                veh, raito = veh_info
                veh: Vehicle
                ava_alloc.append(self.computation_power_alloc[idx])
            else:
                ava_alloc.append(None)

        # improve performance
        if utils.all_none(ava_alloc):
            return
        
        sum_alloc = sum([a if a is not None else 0 for a in ava_alloc])

        if sum_alloc != 0:
            self.cp_norm = [
                (
                    a * self.computation_power_alloc[a_idx] / sum_alloc
                    if a is not None
                    else 0
                )
                for a_idx, a in enumerate(ava_alloc)
            ]
        else:
            assert NotImplementedError("why you here")

        real_cp = self.computation_power * self.cp_usage
        self.real_cp_alloc.olist = [real_cp * cp_n for cp_n in self.cp_norm]
        pass

    def box_alloc_bw(self, alloc_bw_list, veh_ids):
        self.bw_alloc.olist = list.copy(alloc_bw_list.tolist())
        from vanet_env import network

        ava_alloc = []
        for idx, veh in enumerate(self.connections):
            veh: Vehicle
            if veh is not None and veh.vehicle_id in veh_ids:
                ava_alloc.append(self.bw_alloc[idx])
            else:
                ava_alloc.append(None)

        # improve performance
        if utils.all_none(ava_alloc):
            return

        sum_alloc = sum([a if a is not None else 0 for a in ava_alloc])

        if sum_alloc != 0:
            self.bw_norm = [
                a * self.bw_alloc[a_idx] / sum_alloc if a is not None else 0
                for a_idx, a in enumerate(ava_alloc)
            ]
        else:
            assert NotImplementedError("why you here")

        for idx, veh in enumerate(self.connections):
            veh: Vehicle
            if veh is not None and veh.vehicle_id in veh_ids:
                veh.data_rate = network.channel_capacity(
                    self,
                    veh,
                    veh.distance_to_rsu,
                    self.bw * self.bw_norm[idx],
                )
                veh
        pass

    def frame_allocate_computing_power(
        self,
        alloc_index: int,
        cp_a: int,
        cp_usage: int,
        proc_veh_set: set["Vehicle"],
        veh_ids,
    ):
        self.cp_usage = cp_usage
        self.computation_power_alloc.replace(cp_a, alloc_index)
        self.cp_norm = utils.normalize_array_np(self.computation_power_alloc)
        veh: Vehicle = self.handling_jobs[alloc_index]
        if veh is not None and veh.vehicle_id in veh_ids:
            proc_veh_set.add(veh.vehicle_id)

    def frame_cache_content(self, caching_decision, num_content):
        caching_decision = math.floor(caching_decision * num_content)
        self.caching_contents.queue_jumping(caching_decision)

    # notice, cal utility only when connect this rsu
    def frame_allocate_bandwidth(
        self,
        alloc_index: int,
        bw_a: int,
        proc_veh_set: set["Vehicle"],
        veh_ids,
        bw_ratio=1,
    ):
        from vanet_env import network

        self.bw_ratio = bw_ratio
        self.bw_alloc.replace(bw_a, alloc_index)
        self.bw_norm = utils.normalize_array_np(self.bw_alloc)

        for idx, veh in enumerate(self.connections):
            veh: Vehicle
            if veh is not None and veh.vehicle_id in veh_ids:
                proc_veh_set.add(veh.vehicle_id)
                veh.data_rate = network.channel_capacity(
                    self,
                    veh,
                    distance=veh.distance_to_rsu,
                    bw=self.bw * self.bw_norm[idx] * self.bw_ratio,
                )

    def allocate_computing_power(
        self, ac_list: list, cp_usage, proc_veh_set: set["Vehicle"]
    ):
        self.cp_usage = cp_usage
        self.computation_power_alloc = list.copy(ac_list)
        self.cp_norm = utils.normalize_array_np(self.computation_power_alloc)
        ...

    def cache_content(self, caching_decision: list):
        content_index_list = np.where(caching_decision == 1)[0][:10].tolist()
        self.caching_contents = list.copy(content_index_list)
        ...

    def allocate_bandwidth(self, abw_list: list, bw_ratio):
        from vanet_env import network

        self.bw_ratio = bw_ratio
        self.bw_alloc = list.copy(abw_list)
        self.bw_norm = utils.normalize_array_np(self.bw_alloc)

        # update all or update one?
        for idx, veh in enumerate(self.connections):
            if veh is not None:
                veh.data_rate = network.channel_capacity(
                    self, veh, self.bw * self.bw_norm[idx] * self.bw_ratio
                )

    # dev tag: index connect?
    def connect(self, conn, jumping=True, index=-1):
        conn.connect(self)

        if jumping:
            self.disconnect_last()
            self.connections.queue_jumping(conn)
        else:
            if index == -1:
                self.connections.append(conn)
            else:
                self.connections.replace(conn, index)

    def disconnect_last(self):
        if self.connections[-1] is not None:
            self.disconnect(self.connections[-1])

    def disconnect(self, conn):
        conn.disconnect()
        self.connections.remove(elem=conn)

    def update_conn_list(self):
        # clean deprecated connections
        for idx, conn in enumerate(self.connections):
            if conn is None:
                continue
            if conn not in self.range_connections:
                self.disconnect(conn)

    # def update_job_handling_list(self):
    #     for idx, veh_id in enumerate(self.handling_jobs):
    #          if veh_id is None:
    #             continue

    # clean deprecated jobs
    # for idx, hconn in enumerate(self.handling_jobs):
    #     if hconn is None:
    #         continue
    #     if hconn.connected == False:
    #         self.handling_jobs.remove(idx)

    # python 3.9+
    def frame_handling_job(
        self,
        proc_veh_set: set["Vehicle"],
        rsu: "Rsu",
        h_index: int,
        handling: int,
        veh_ids,
    ):
        # not handling 也 抛弃
        veh: Vehicle = self.handling_job_queue[h_index]

        if veh is not None and veh.vehicle_id in veh_ids:
            proc_veh_set.add(veh.vehicle_id)

            if handling == 1:
                if veh not in self.handling_jobs:
                    veh.job.processing_rsu_id = self.id
                    veh_replaced: Vehicle = self.handling_jobs.replace(
                        elem=veh, index=h_index
                    )
                    # 替换自动云连接
                    if veh_replaced is not None:
                        veh_replaced.is_cloud = True
                        veh_replaced.job.processing_rsu_id = env_config.NUM_RSU
            # cloud
            else:
                veh.is_cloud = True
                veh.job.processing_rsu_id = env_config.NUM_RSU

    # python < 3.11
    def frame_queuing_job(
        self, conn_rsu: "Rsu", veh: "Vehicle", index: int, cloud: bool = False
    ):
        if cloud:
            # there can be modify to more adaptable
            # specify rsu process this
            # veh.connected_rsu_id = config.NUM_RSU
            veh.is_cloud = True
            veh.job.processing_rsu_id = env_config.NUM_RSU

        # 唯一一个使is_cloud失效的地方
        veh.is_cloud = False
        self.handling_job_queue.replace(elem=veh, index=index)

    def handling_job(self, jbh_list: list):
        # handle
        for idx, hconn in enumerate(self.handling_job_queue):
            # handle if 1
            if jbh_list[idx]:
                # dev tag: append or direct change?
                # if append, need remove logic
                if hconn not in self.handling_jobs:
                    hconn.veh.job.processing_rsu_id = self.id
                    self.handling_jobs.replace(hconn, idx)
                    # dev tag: replace or append?
                    hconn.rsu.connect(hconn)
            # else:
            #     hconn.disconnect()
        ...

    def queuing_job(self, conn, cloud=False):
        if cloud:
            # there can be modify to more adaptable
            conn.rsu = self
            conn.is_cloud = True
            conn.veh.job.processing_rsu_id = env_config.NUM_RSU

        # pending handle
        # if self handling, queue-jumping
        if conn.rsu.id == self.id:
            self.handling_job_queue.queue_jumping(conn)
        else:
            self.handling_job_queue.append(conn)

    def distance(self, vh_position):
        return np.sqrt(
            (self.position.x - vh_position.x) ** 2
            + (self.position.y - vh_position.y) ** 2
        )

    # to km
    def real_distance(self, vh_position):
        return self.distance(vh_position) / (1000 / env_config.COORDINATE_UNIT)

    def get_d1_d2(self, vh_position: Point, vh_direction):
        # vh_direction is angle in degree, 0 points north, 90 points east ...
        # for convince: 0-45°, 315°-360° is north; 135°-225° is south
        # if (0 <= vh_direction <= 45) or (315 <= vh_direction <= 360):
        #     # "North"
        #     return abs(self.position.y - vh_position.y), abs(
        #         self.position.x - vh_position.x
        #     )
        # elif 135 <= vh_direction <= 225:
        #     # "South"
        #     return abs(self.position.y - vh_position.y), abs(
        #         self.position.x - vh_position.x
        #     )
        # else:
        #     return abs(self.position.x - vh_position.x), abs(
        #         self.position.y - vh_position.y
        #     )

        # Convert direction from degrees to radians
        angle_rad = math.radians(vh_direction)

        # Calculate the relative position of RSU with respect to the vehicle
        dx = self.position.x - vh_position.x
        dy = self.position.y - vh_position.y

        # Calculate the unit vector of the vehicle's direction
        unit_x = math.sin(angle_rad)  # x-component of the unit vector
        unit_y = math.cos(angle_rad)  # y-component of the unit vector

        # Project the relative position onto the vehicle's direction
        # Horizontal distance (parallel to the vehicle's direction)
        horizontal_distance = abs(dx * unit_x + dy * unit_y)

        # Vertical distance (perpendicular to the vehicle's direction)
        vertical_distance = abs(dx * unit_y - dy * unit_x)

        return vertical_distance, horizontal_distance


class Job:
    def __init__(self, job_id, job_size, job_type):
        self.job_id = job_id
        # deprecated
        self.job_size = job_size
        self.job_type = job_type
        self.qoe = 0
        # processed size # deprecated
        self.job_processed = 0
        # processing rsu
        self.is_cloud = False
        # 3 can modify as nb num
        self.processing_rsus = OrderedQueueList(3)

    # deprecated
    def done(self):
        return self.job_size - self.job_processed <= 0

    # deprecated
    def job_info(self):
        return max(self.job_size - self.job_processed, 0)


class Vehicle:
    def __init__(
        self,
        vehicle_id,
        sumo,
        join_time,
        job_type=None,
        init_all=True,
        seed=env_config.SEED,
        max_connections=4,
    ):
        self.vehicle_id = vehicle_id
        self.height = env_config.VEHICLE_ANTENNA_HEIGHT
        self.position = None
        self.angle = None
        self.sumo = sumo
        self.seed = seed
        self.join_time = join_time

        random.seed(self.seed)

        job_size = random.randint(8, env_config.MAX_JOB_SIZE)

        # done -> Need for popularity modeling
        job_type = job_type

        # single job, id is veh id
        self.job = Job(vehicle_id, job_size, job_type)

        if init_all:
            self.position = Point(sumo.vehicle.getPosition(vehicle_id))
            self.angle = sumo.vehicle.getAngle(self.vehicle_id)

        self.is_cloud = False
        # update these when update connection queue
        self.connected_rsu_id = None
        # 前一步连接的rsu是谁？
        self.pre_connected_rsu_id = None

        self.data_rate = 0
        self.distance_to_rsu = None
        # connected rsus, may not needed
        # self.connections = OrderedQueueList(max_connections)

    def job_process(self, idx, rsu):
        self.job.is_cloud = False
        self.is_cloud = False
        self.job.processing_rsus[idx] = rsu

    def job_deprocess(self):
        self.job.is_cloud = True
        self.is_cloud = True

        if not self.job.processing_rsus.is_empty():
            for rsu in self.job.processing_rsus:
                if rsu is None:
                    continue
                rsu: Rsu
                rsu.remove_job(elem=self)

        self.job.processing_rsus.clear()

    # only use from Connection.disconnect()
    def disconnect(self, conn):
        if conn in self.connections:
            self.connections.remove(elem=conn)

    def update_job_type(self, job_type):
        self.job_type = job_type

    def update_pos_direction(self):
        self.position = Point(self.sumo.vehicle.getPosition(self.vehicle_id))
        self.angle = self.sumo.vehicle.getAngle(self.vehicle_id)

    def get_speed(self):
        return self.sumo.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        self.sumo.vehicle.setSpeed(self.vehicle_id, speed)

    def get_position(self):
        return (
            self.position
            if self.position is not None
            else self.sumo.vehicle.getPosition(self.vehicle_id)
        )

    def get_angle(self):
        return (
            self.angle
            if self.angle is not None
            else self.sumo.vehicle.getAngle(self.vehicle_id)
        )


class CustomVehicle(Vehicle):
    def __init__(
        self,
        id,
        position: Point,
        sumo=traci,
        height=env_config.VEHICLE_ANTENNA_HEIGHT,
        direction=0,
    ):
        self.id = id
        self.position = position
        self.height = height
        self.speed = 0
        self.acceleration = 0
        # n s w e, ↑ ↓ ← →, 0 1 2 3
        self.direction = direction
        self.sumo = sumo

    def get_speed(self):
        return self.sumo.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        self.sumo.vehicle.setSpeed(self.vehicle_id, speed)

    def get_position(self):
        return self.position

    def get_angle(self):
        if self.direction == 0:
            return 0
        elif self.direction == 1:
            return 180
        elif self.direction == 2:
            return 270
        else:
            return 90


class Connection:
    def __init__(self, rsu: Rsu, veh: Vehicle, data_rate=0.0, cloud=False):
        self.is_cloud = cloud

        self.veh = veh
        self.rsu = None if self.is_cloud else rsu
        self.data_rate = data_rate
        self.qoe = 0
        self.connected = False
        # str
        self.id = str(rsu.id) + veh.vehicle_id

    def check_connection(self):
        return self.is_cloud if not self.connected else True

    def connect(self, rsu, is_cloud=False):
        """
        only connect when take action
        """
        if is_cloud:
            self.is_cloud = True
            self.rsu = None
        else:
            self.rsu = rsu
            self.connected = True

    # process by rsu
    def disconnect(self):
        # self.rsu = None
        # self.veh.job.processing_rsu_id = None
        self.is_cloud = False
        self.connected = False
        self.veh.disconnect(self)

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id
