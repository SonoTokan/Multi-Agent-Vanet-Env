import sys

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
        position,
        bw=config.RSU_MAX_TRANSMITTED_BANDWIDTH,
        frequency=config.RSU_FREQUENCY,
        transmitted_power=config.RSU_TRANSMITTED_POWER,
        height=config.RSU_ANTENNA_HEIGHT,
        noise_power=config.RSU_NOISE_POWER,
        snr_threshold=config.RSU_SNR_THRESHOLD,
        computation_power=config.RSU_COMPUTATION_POWER,
        caching_capacity=config.RSU_CACHING_CAPACITY,
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
        self.connected_vehicles = []
        self.bind_road = []

    def distance(self, vh_position):
        return np.sqrt(
            (self.position[0] - vh_position[0]) ** 2
            + (self.position[1] - vh_position[1]) ** 2
        )

    # to km
    def real_distance(self, vh_position):
        return self.distance(vh_position) / (1000 / config.COORDINATE_UNIT)

    def get_d1_d2(self, vh_position, vh_direction):
        # when vh movement is left or right i.e. w or e, d1 is the x distance, d2 is the y distance
        # when vh movement is up or down i.e. n or s, d1 is the y distance, d2 is the x distance
        if vh_direction <= 1:
            return abs(self.position[1] - vh_position[1]), abs(
                self.position[0] - vh_position[0]
            )
        else:
            return abs(self.position[0] - vh_position[0]), abs(
                self.position[1] - vh_position[1]
            )

        pass


class Vehicle:
    def __init__(self, id, position, height=config.VEHICLE_ANTENNA_HEIGHT, direction=0):
        self.id = id
        self.position = position
        self.height = height
        self.speed = 0
        self.acceleration = 0
        # n s w e, ↑ ↓ ← →, 0 1 2 3
        self.direction = direction
