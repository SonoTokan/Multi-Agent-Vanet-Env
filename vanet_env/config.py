import sys

sys.path.append("./")

print("config loaded")

# from vanet_env.entites import Rsu, Vehicle

# class Config:
#     def __init__(self):
#         self.path_loss = "OkumuraHata"
#         self.rsu = Rsu()
SEED = 1024

# Canvas Size
# MAP_SIZE = (400, 400)

# Road Config
# Road width (meters)
ROAD_WIDTH = 10

# RSU Config, road align 77

RSU_POSITIONS = []

# roadside_down
for x in range(0, 400, 100):
    for y in range(0, 77 * 5, 77):
        RSU_POSITIONS.append((x, y))

# roadside_up
# for x in range(0, 400, 100):
#     for y in range(17, (77 + 17) * 5, 77 + 17):
#         RSU_POSITIONS.append((x, y))

NUM_RSU = len(RSU_POSITIONS)

# Rsu Config
# Computation Power cores and ghz
RSU_COMPUTATION_POWER = 256
# Caching Capacity
RSU_CACHING_CAPACITY = 10
# Transmitted power (( P_t )): 1 Watt (30 dBm)
RSU_TRANSMITTED_POWER = 30
# Noise power (( N )): ( 10^{-9} ) Watts (-90 dBm)
RSU_NOISE_POWER = 1e-9
# Max Transmitted Bandwidth (( B )): 20 MHz, i.e. 20e6 Hz
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
RSU_NUM_ANTENNA = 4
# Data rate threshold
DATA_RATE_TR = 8

# Vehicle Config
NUM_VEHICLES = 50
# Antenna Height
VEHICLE_ANTENNA_HEIGHT = 1.5
# max Job size
MAX_JOB_SIZE = 256
# content num
NUM_CONTENT = 100

# Render Config

# Coordinate unit (meters)
COORDINATE_UNIT = 1

# freespace speed of light (m/s)
C = 3e8
