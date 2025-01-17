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
        # {"connected_veh": None, "connected_quality": None}
        self.connections = []
        self.bind_road = []
        self.num_atn = num_atn

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
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.height = config.VEHICLE_ANTENNA_HEIGHT
        self.position = Point(traci.vehicle.getPosition(vehicle_id))
        # connected rsus, not needed
        # self.connections = []

    def get_speed(self):
        return traci.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        traci.vehicle.setSpeed(self.vehicle_id, speed)

    def get_position(self):
        return self.position

    def get_angle(self):
        return traci.vehicle.getAngle(self.vehicle_id)


class CustomVehicle(Vehicle):
    def __init__(
        self, id, position: Point, height=config.VEHICLE_ANTENNA_HEIGHT, direction=0
    ):
        self.id = id
        self.position = position
        self.height = height
        self.speed = 0
        self.acceleration = 0
        # n s w e, ↑ ↓ ← →, 0 1 2 3
        self.direction = direction

    def get_speed(self):
        return traci.vehicle.getSpeed(self.vehicle_id)

    def set_speed(self, speed):
        traci.vehicle.setSpeed(self.vehicle_id, speed)

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
