import sys

sys.path.append("./")
from vanet_env import config
from vanet_env.entites import Rsu, Vehicle
from vanet_env import utils
import numpy as np

EPSILON = 1e-9


# Okumura-Hata path Loss another version (from github)
def power_loss(rsu: Rsu, vh: Vehicle):
    distance = rsu.real_distance(vh.position)

    ch = (
        0.8
        + (1.1 * np.log10(rsu.frequency) - 0.7) * rsu.height
        - 1.56 * np.log10(rsu.frequency)
    )
    tmp_1 = 69.55 - ch + 26.16 * np.log10(rsu.frequency) - 13.82 * np.log10(rsu.height)
    tmp_2 = 44.9 - 6.55 * np.log10(rsu.height)

    # add small epsilon to avoid log(0) if distance = 0
    return tmp_1 + tmp_2 * np.log10(distance + EPSILON)


# distance (km)
# Okumura-Hata path Loss
# Lb​=69.55+26.16log10​(f)−13.82log10​(hb​)−a(hm​)+(44.9−6.55log10​(hb​))log10​(d)
def okumura_hata_path_loss(rsu: Rsu, vh: Vehicle, city_type="small"):
    distance = rsu.real_distance(vh.position)

    if city_type == "large":
        if rsu.frequency <= 200:
            a_hm = 8.29 * (np.log10(1.54 * vh.height)) ** 2 - 1.1
        else:
            a_hm = 3.2 * (np.log10(11.75 * vh.height)) ** 2 - 4.97
    else:
        a_hm = (1.1 * np.log10(rsu.frequency) - 0.7) * vh.height - (
            1.56 * np.log10(rsu.frequency) - 0.8
        )

    path_loss = (
        69.55
        + 26.16 * np.log10(rsu.frequency)
        - 13.82 * np.log10(rsu.height)
        - a_hm
        + (44.9 - 6.55 * np.log10(rsu.height)) * np.log10(distance + EPSILON)
    )

    return path_loss


# deprecated
# distance (m)
#  A common path loss model is the log-distance path loss model:
# PL(d) = PL(d_0) + 10nlog_10(d/d_0)
def path_loss(distance):
    d_0 = config.RSU_REFERENCE_DISTANCE

    if distance <= d_0:
        return config.RSU_PATH_LOSS_REFERENCE_DISTANCE

    return path_loss(d_0) + 10 * config.RSU_PATH_LOSS_EXPONENT * np.log10(
        distance / d_0
    )


# winner + b1 model
class WinnerB1:
    def __init__(
        self,
        building_height=20,
        street_width=20,
        building_separation=50,
        street_orientation=30,
    ):
        self.building_height = building_height
        self.street_width = street_width
        self.building_separation = building_separation
        self.street_orientation = street_orientation
        pass

    def path_loss(self, rsu: Rsu, vh: Vehicle):
        distance = rsu.real_distance(vh.position)
        frequency = rsu.frequency
        base_station_height = rsu.height
        mobile_station_height = vh.height
        L0 = 32.4 + 20 * np.log10(distance) + 20 * np.log10(frequency)

        delta_hm = self.building_height - mobile_station_height
        Lrts = (
            -16.9
            - 10 * np.log10(self.street_width)
            + 10 * np.log10(frequency)
            + 20 * np.log10(delta_hm)
            + self.street_orientation_loss(self.street_orientation)
        )

        delta_hb = base_station_height - self.building_height
        if delta_hb > 0:
            Lbsh = -18 * np.log10(1 + delta_hb)
            ka = 54
            kd = 18
        else:
            Lbsh = 0
            kd = 18 - 15 * delta_hb / self.building_height
            if distance >= 0.5:
                ka = 54 - 0.8 * delta_hb
            else:
                ka = 54 - 1.6 * delta_hb * distance

        kf = -4 + 0.7 * (frequency / 925 - 1)
        Lmsd = (
            Lbsh
            + ka
            + kd * np.log10(distance)
            + kf * np.log10(frequency)
            - 9 * np.log10(self.building_separation)
        )

        L = L0 + Lrts + Lmsd
        return L

    def street_orientation_loss(self, alpha):
        if 0 <= alpha < 35:
            return -10 + 0.354 * alpha
        elif 35 <= alpha < 55:
            return 2.5 + 0.075 * (alpha - 35)
        else:
            return 4 - 0.114 * (alpha - 55)

    # def channel_capacity(self, rsu: Rsu, vh: Vehicle):
    #     distance = rsu.real_distance(vh.position)
    #     path_loss = path_loss()
    #     received_power_dbm = transmitted_power_dbm - path_loss
    #     received_power_w = 10 ** ((received_power_dbm - 30) / 10)
    #     snr = received_power_w / noise_power
    #     channel_capacity = bandwidth * np.log2(1 + snr)
    #     return channel_capacity / 1e6  # 转换为 Mbps


# convert dbm to watt
def dbm_to_watt(P_dbm):
    return 10 ** ((P_dbm - 30) / 10)


# P_r = P_t - PL(d)
# SNR = P_r / N
# P_r need to convert to watt
def snr(rsu: Rsu, vh: Vehicle, path_loss_func="winner_b1"):
    # noise_power = dbm_to_watt(rsu.noise_power)
    # return (
    #     dbm_to_watt(rsu.transmitted_power - okumura_hata_path_loss(rsu, vh))
    #     / rsu.noise_power
    # )
    path_loss = 0

    if path_loss_func == "winner_b1":
        path_loss = WinnerB1().path_loss(rsu, vh)
    else:
        path_loss = okumura_hata_path_loss(rsu, vh)

    snr = dbm_to_watt(rsu.transmitted_power - path_loss) / rsu.noise_power
    return snr


# inverse Okumura-Hata path loss to get max connection distance
def max_distance_oku(rsu: Rsu, city_type="small"):

    received_power_w = rsu.snr_threshold * rsu.noise_power
    received_power_dbm = 10 * np.log10(received_power_w) + 30

    path_loss = rsu.transmitted_power - received_power_dbm

    if city_type == "large":
        if rsu.frequency <= 200:
            a_hm = 8.29 * (np.log10(1.54 * config.VEHICLE_ANTENNA_HEIGHT)) ** 2 - 1.1
        else:
            a_hm = 3.2 * (np.log10(11.75 * config.VEHICLE_ANTENNA_HEIGHT)) ** 2 - 4.97
    else:
        a_hm = (1.1 * np.log10(rsu.frequency) - 0.7) * config.VEHICLE_ANTENNA_HEIGHT - (
            1.56 * np.log10(rsu.frequency) - 0.8
        )

    term1 = (
        path_loss
        - 69.55
        - 26.16 * np.log10(rsu.frequency)
        + 13.82 * np.log10(rsu.height)
        + a_hm
    )
    term2 = 44.9 - 6.55 * np.log10(rsu.height)
    distance = 10 ** (term1 / term2)

    return distance


def max_distance(rsu: Rsu):
    max_distance = 0
    step = 0.001  # km
    distance = utils.realDistanceToDistance(step)

    while True:
        vh = Vehicle(0, (rsu.position[0] + distance, rsu.position[1]))

        if snr(rsu, vh) < rsu.snr_threshold:
            break

        max_distance = distance
        distance += utils.realDistanceToDistance(step)

    return max_distance


def bpsToMbps(bps):
    return bps / 1e6


# Calculate connect speed:
# C=Blog2​(1+SNR)
def channel_capacity(rsu: Rsu, vh: Vehicle):
    return bpsToMbps(rsu.bw * np.log2(1 + snr(rsu, vh)))
