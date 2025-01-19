import random
import sys

from shapely import Point

import traci

sys.path.append("./")

import numpy as np
from vanet_env import config


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

    def append(self, elem):
        for i in range(self.max_size):
            if self.olist[i] is None:
                self.olist[i] = elem
                return True
        return False  # 如果没有空位，返回 False

    def insert(self, elem, index):
        self.olist[index] = elem

    def remove(self, index):
        if 0 <= index < self.max_size:
            self.olist[index] = None
        else:
            assert ValueError

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
        max_connection=10,
    ):
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
        self.connections = OrderedQueueList(max_connection)
        self.handling_job = OrderedQueueList(max_connection)  # may not necessary
        self.bw_alloc = OrderedQueueList(max_connection)
        self.computation_power_alloc = OrderedQueueList(max_connection)
        self.caching_content = OrderedQueueList(caching_capacity)
        self.num_atn = num_atn

    def allocate_computing_power(self, ac_list: list):
        self.computation_power_alloc = np.copy(ac_list)
        ...

    def cache_content(self, cc_list: list):
        self.caching_content = np.copy(cc_list)
        ...

    def allocate_bandwidth(self, abw_list: list):
        self.bw_alloc = np.copy(abw_list)
        ...

    def handle_job(self, job):
        # pending handle
        ...

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


class Vehicle:
    def __init__(self, vehicle_id, sumo, init_all=True, seed=config.SEED):
        self.vehicle_id = vehicle_id
        self.height = config.VEHICLE_ANTENNA_HEIGHT
        self.position = None
        self.angle = None
        self.sumo = sumo
        self.seed = seed

        random.seed(self.seed)

        self.job_size = random.randint(8, config.MAX_JOB_SIZE)
        self.job_type = random.randint(0, config.NUM_CONTENT)

        if init_all:
            self.position = Point(sumo.vehicle.getPosition(vehicle_id))
            self.angle = sumo.vehicle.getAngle(self.vehicle_id)
        # connected rsus, not needed
        # self.connections = []

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
    def __init__(self, rsu: Rsu, veh: Vehicle, data_rate=0.0):
        self.veh = veh
        self.rsu = rsu
        self.data_rate = data_rate
        self.connected = False
        # str
        self.id = str(rsu.id) + veh.vehicle_id

    def connect(self):
        """
        only connect when take action
        """
        self.connected = True

    def disconnect(self):
        self.connected = False

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id
