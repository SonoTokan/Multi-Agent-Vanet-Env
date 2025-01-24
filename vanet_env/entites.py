import random
import sys

from shapely import Point

import traci

sys.path.append("./")

import numpy as np
from vanet_env import config, utils, network


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
        self.olist = [elem] + self.olist[:-1]
        pass

    # Queuing
    def append(self, elem):
        for i in range(self.max_size):
            if self.olist[i] is None:
                self.olist[i] = elem
                return True
        return False

    def replace(self, elem, index):
        self.olist[index] = elem

    def remove(self, index=-1, elem=None):
        if elem is not None:
            self.remove(index=self.olist.index(elem))
        else:
            if 0 <= index < self.max_size:
                self.olist[index] = None
            else:
                assert IndexError

    def to_list_replace_none(self):
        return [0 if x is None else x for x in self.olist]

    def size(self):
        return sum(1 for conn in self.olist if conn is not None)

    def is_empty(self):
        """
        may has bug, do not use
        """
        return all(conn is None for conn in self.olist)

    def avg(self):
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
        if 0 <= index < self.max_size:
            return self.olist[index]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if 0 <= index < self.max_size:
            self.olist[index] = value
        else:
            raise IndexError("Index out of range")


class Rsu:
    def __init__(
        self,
        id,
        position: Point,
        bw=config.RSU_MAX_TRANSMITTED_BANDWIDTH,
        frequency=config.RSU_FREQUENCY,
        transmitted_power=config.RSU_TRANSMITTED_POWER,
        height=config.RSU_ANTENNA_HEIGHT,
        noise_power=config.RSU_NOISE_POWER,
        snr_threshold=config.RSU_SNR_THRESHOLD,
        computation_power=config.RSU_COMPUTATION_POWER,
        caching_capacity=config.RSU_CACHING_CAPACITY,
        num_atn=config.RSU_NUM_ANTENNA,
        tx_gain=config.ANTENNA_GAIN,
        max_connections=10,
        max_cores=8,
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

        self.connections_queue = OrderedQueueList(max_connections)
        self.connections = OrderedQueueList(max_connections)
        self.handling_job_queue = OrderedQueueList(max_cores)  # may not necessary
        self.handling_jobs = OrderedQueueList(max_cores)
        self.bw_alloc = OrderedQueueList(max_cores)
        self.computation_power_alloc = OrderedQueueList(max_cores)
        self.caching_contents = OrderedQueueList(caching_capacity)

        self.energy_efficiency = 0

        self.cp_usage = 1.0
        self.bw_ratio = 1.0
        self.tx_ratio = 1.0

    def get_tx_power(self):
        return self.transmitted_power * self.tx_ratio / 100 + self.tx_gain

    def allocate_computing_power(self, ac_list: list, cp_usage):
        self.cp_usage = cp_usage
        self.computation_power_alloc = list.copy(ac_list)
        self.cp_norm = utils.normalize_array_np(self.computation_power_alloc)
        ...

    def cache_content(self, caching_decision: list):
        content_index_list = np.where(caching_decision == 1)[0][:10].tolist()
        self.caching_contents = list.copy(content_index_list)
        ...

    def allocate_bandwidth(self, abw_list: list, bw_ratio):
        self.bw_ratio = bw_ratio
        self.bw_alloc = list.copy(abw_list)
        self.bw_norm = utils.normalize_array_np(self.bw_alloc)

        for idx, conn in enumerate(self.connections):
            if conn is not None:
                conn.data_rate = network.channel_capacity(
                    self, conn.veh, self.bw * self.bw_norm[idx] * self.bw_ratio
                )

    # dev tag: index connect?
    def connect(self, conn, index=-1):
        conn.connect(self)

        if index == -1:
            self.connections.append(conn)
        else:
            self.connections.replace(conn, index)

    def disconnect(self, conn):
        conn.disconnect()
        self.connections.remove(elem=conn)

    def update_job_conn_list(self):
        # clean deprecated connections
        for idx, conn in enumerate(self.connections):
            if conn is None:
                continue
            if conn not in self.connections_queue:
                self.disconnect(conn)

        # clean deprecated jobs
        for idx, hconn in enumerate(self.handling_jobs):
            if hconn is None:
                continue
            if hconn.connected == False:
                self.handling_jobs.remove(idx)

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
            conn.veh.job.processing_rsu_id = config.NUM_RSU

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
        return self.distance(vh_position) / (1000 / config.COORDINATE_UNIT)

    def get_d1_d2(self, vh_position: Point, vh_direction):
        # vh_direction is angle in degree, 0 points north, 90 points east ...
        # for convince: 0-45°, 315°-360° is north; 135°-225° is south
        if (0 <= vh_direction <= 45) or (315 <= vh_direction <= 360):
            # "North"
            return abs(self.position.y - vh_position.y), abs(
                self.position.x - vh_position.x
            )
        elif 135 <= vh_direction <= 225:
            # "South"
            return abs(self.position.y - vh_position.y), abs(
                self.position.x - vh_position.x
            )
        else:
            return abs(self.position.x - vh_position.x), abs(
                self.position.y - vh_position.y
            )


class Job:
    def __init__(self, job_id, job_size, job_type):
        self.job_id = job_id
        # deprecated
        self.job_size = job_size
        self.job_type = job_type
        # processed size # deprecated
        self.job_processed = 0
        # processing rsu
        self.processing_rsu_id = None

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
        job_type=None,
        init_all=True,
        seed=config.SEED,
        max_connections=4,
    ):
        self.vehicle_id = vehicle_id
        self.height = config.VEHICLE_ANTENNA_HEIGHT
        self.position = None
        self.angle = None
        self.sumo = sumo
        self.seed = seed

        random.seed(self.seed)

        job_size = random.randint(8, config.MAX_JOB_SIZE)

        # done -> Need for popularity modeling
        job_type = job_type

        # single job, id is veh id
        self.job = Job(vehicle_id, job_size, job_type)

        if init_all:
            self.position = Point(sumo.vehicle.getPosition(vehicle_id))
            self.angle = sumo.vehicle.getAngle(self.vehicle_id)

        # connected rsus, may not needed
        self.connections = OrderedQueueList(max_connections)

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
        height=config.VEHICLE_ANTENNA_HEIGHT,
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

    def disconnect(self):
        self.rsu = None
        self.is_cloud = False
        self.connected = False

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id
