import math
import sys

sys.path.append("./")

# from vanet_env.entites import Rsu, Vehicle
# class Config:
#     def __init__(self):
#         self.path_loss = "OkumuraHata"
#         self.rsu = Rsu()
SEED = 114514
MAP_NAME = "london"
# Canvas Size
# MAP_SIZE = (400, 400)

"""
Rsu Config
"""
# Road width (meters)
ROAD_WIDTH = 10

# RSU Config, road align 77

RSU_POSITIONS = []

# roadside_down
if MAP_NAME != "london":
    for x in range(40, 400, 90):
        for y in range(0, 77 * 5, 77):
            RSU_POSITIONS.append((x, y))
else:
    for x in range(40, 400, 90):
        for y in range(0, 77 * 5, 77):
            RSU_POSITIONS.append((x, y))
# roadside_up
# for x in range(0, 400, 100):
#     for y in range(17, (77 + 17) * 5, 77 + 17):
#         RSU_POSITIONS.append((x, y))

NUM_RSU = len(RSU_POSITIONS)


# Computation Power TFLOPs/s based on RTX 4060 Ti 22.1 tflops、RTX 4070 29.1 tflops、RTX 4080 48.7 tflops、RTX 4090 82.6 tflops
RSU_COMPUTATION_POWER = 82.6 + 48.7
# Caching Capacity
RSU_CACHING_CAPACITY = 1
# Transmitted power (( P_t )): 1 Watt (27 dBm)
RSU_TRANSMITTED_POWER = 30
# Noise power (( N )): ( 10^{-9} ) Watts (-90 dBm)
RSU_NOISE_POWER = 1e-9
# Max Transmitted Bandwidth (( B )): 10 MHz, i.e. 10e6 Hz
RSU_MAX_TRANSMITTED_BANDWIDTH = 20e6
# Frequency (( f )): 2.4, 5.9 GHz, i.e. 2400 MHz, 5905 MHz
RSU_FREQUENCY = 5905
# Antenna Height
RSU_ANTENNA_HEIGHT = 10
# Path loss exponent (( n )): 3
RSU_PATH_LOSS_EXPONENT = 3
# Reference distance 1 meters
RSU_REFERENCE_DISTANCE = 1e-3
# Path loss at reference distance (( PL(d_0) )): 40 dB
RSU_PATH_LOSS_REFERENCE_DISTANCE = 40
# SNR threshold
RSU_SNR_THRESHOLD = 2e-8
# MIMO, numbers of antenna
RSU_NUM_ANTENNA = 2
# ANTENNA Gain (( dBi )), V2X Omnidirectional Fiberglass Antenna
ANTENNA_GAIN = 3
# Data rate threshold
DATA_RATE_TR = 40
# max computation cores
NUM_CORES = 5
# max downlink connections, better to equal cores and 最好能被fps整除
MAX_CONNECTIONS = 5
# ee ratio i.e. MAX_EE - EE = cp_usage
MAX_EE = 0.2

"""
Vehicle Config
"""
NUM_VEHICLES = 150
# Antenna Height
VEHICLE_ANTENNA_HEIGHT = 1.5
# max Job size
MAX_JOB_SIZE = 256
# content num
NUM_CONTENT = 5

"""
Cloud Config
"""
CLOUD_COMPUTATIONALLY_TIME = 0.5
CLOUD_TRANS_TIME = 10

"""
Others
"""
# Render Config
# Coordinate unit (meters)
COORDINATE_UNIT = 1

# freespace speed of light (m/s)
C = 3e8

"""
utility
"""
# non caching latency factor
NON_CACHING_FACTOR = 1.25
# Hop latency (ms)
HOP_LATENCY = 3
# Hop optimize per frame
HOP_OPT_FACTOR = 90
# vrc Meta Quest 2 with Complex Avatar 35-65 tflops, 25-70 Mbps
# 90fps -> 1frame=0.01111 s -> 11.11ms
W = 2160
H = 2160
CP_REQUIRE_PIX = 9.5e4
FPS_REQUIRE = 90
# tflops
JOB_CP_REQUIRE = math.ceil(W * H * CP_REQUIRE_PIX * FPS_REQUIRE * 1e-12)
# mbps
JOB_DR_REQUIRE = 40
JOB_FPS_REQUIRE = 90
# latency factor
LATENCY_FACTOR = 0.5
MAX_QOE = 1.0


# computational factor
COMPUTATIONAL_FACTOR = 0.5

print("config loaded")
