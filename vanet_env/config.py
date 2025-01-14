import sys

sys.path.append("./")

# from vanet_env.entites import Rsu, Vehicle

# class Config:
#     def __init__(self):
#         self.path_loss = "OkumuraHata"
#         self.rsu = Rsu()


# RSU Config
NUM_RSU = 6
RSU_POSITIONS = [
    (6.25, 4),
    (6.25, 11.25),  # Intersection 1
    (13.5, 4),
    (20.75, 4),
    (20.75, 11.25),  # Intersection 2
    (13.5, 11.25),
]

# Rsu Config
# Computation Power
RSU_COMPUTATION_POWER = 256
# Caching Capacity
RSU_CACHING_CAPACITY = 64
# Transmitted power (( P_t )): 1 Watt (30 dBm)
RSU_TRANSMITTED_POWER = 30
# Noise power (( N )): ( 10^{-9} ) Watts (-90 dBm)
RSU_NOISE_POWER = 1e-9
# Max Transmitted Bandwidth (( B )): 20 MHz, i.e. 20e6 Hz
RSU_MAX_TRANSMITTED_BANDWIDTH = 20e6
# Frequency (( f )): 2.4, 5.9 GHz, i.e. 2400 MHz, 5905 MHz
RSU_FREQUENCY = 900
# Antenna Height
RSU_ANTENNA_HEIGHT = 30
# Path loss exponent (( n )): 3
RSU_PATH_LOSS_EXPONENT = 3
# Reference distance 1 meters
RSU_REFERENCE_DISTANCE = 1e-3
# Path loss at reference distance (( PL(d_0) )): 40 dB
RSU_PATH_LOSS_REFERENCE_DISTANCE = 40
# SNR threshold
RSU_SNR_THRESHOLD = 2e-3

# Vehicle Config
NUM_VEHICLES = 50
# Antenna Height
VEHICLE_ANTENNA_HEIGHT = 1.5

# Render Config

# Coordinate unit (meters)
COORDINATE_UNIT = 100
