import sys

from shapely import Point

sys.path.append("./")
from vanet_env import config
from vanet_env.entites import Rsu, CustomVehicle, Vehicle
from vanet_env import utils
import numpy as np


class OkumuraHata:

    def __init__(self):
        self.EPSILON = 1e-9
        pass

    # Okumura-Hata path Loss another version (from github)
    # may have some bugs not fixed
    def power_loss(self, rsu: Rsu, vh: CustomVehicle):
        distance = rsu.real_distance(vh.position)

        ch = (
            0.8
            + (1.1 * np.log10(rsu.frequency) - 0.7) * rsu.height
            - 1.56 * np.log10(rsu.frequency)
        )
        tmp_1 = (
            69.55 - ch + 26.16 * np.log10(rsu.frequency) - 13.82 * np.log10(rsu.height)
        )
        tmp_2 = 44.9 - 6.55 * np.log10(rsu.height)

        # add small epsilon to avoid log(0) if distance = 0
        return tmp_1 + tmp_2 * np.log10(distance + self.EPSILON)

    # distance (km)
    # Okumura-Hata path Loss
    # Lb​=69.55+26.16log10​(f)−13.82log10​(hb​)−a(hm​)+(44.9−6.55log10​(hb​))log10​(d)
    # may have some bugs not fixed
    def okumura_hata_path_loss(self, rsu: Rsu, vh: CustomVehicle, city_type="small"):
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
            + (44.9 - 6.55 * np.log10(rsu.height)) * np.log10(distance + self.EPSILON)
        )

        return path_loss

    def max_distance_oku(rsu: Rsu, city_type="small"):

        received_power_w = rsu.snr_threshold * rsu.noise_power
        received_power_dbm = 10 * np.log10(received_power_w) + 30

        path_loss = rsu.transmitted_power - received_power_dbm

        if city_type == "large":
            if rsu.frequency <= 200:
                a_hm = (
                    8.29 * (np.log10(1.54 * config.VEHICLE_ANTENNA_HEIGHT)) ** 2 - 1.1
                )
            else:
                a_hm = (
                    3.2 * (np.log10(11.75 * config.VEHICLE_ANTENNA_HEIGHT)) ** 2 - 4.97
                )
        else:
            a_hm = (
                1.1 * np.log10(rsu.frequency) - 0.7
            ) * config.VEHICLE_ANTENNA_HEIGHT - (1.56 * np.log10(rsu.frequency) - 0.8)

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
        street_width=config.ROAD_WIDTH,
        building_separation=50,
        street_orientation=30,
    ):
        self.building_height = building_height
        self.street_width = street_width
        self.building_separation = building_separation
        self.street_orientation = street_orientation
        pass

    # deprecated
    def path_loss_deprecated(self, rsu: Rsu, vh: CustomVehicle):
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

    # deprecated
    def street_orientation_loss(self, alpha):
        if 0 <= alpha < 35:
            return -10 + 0.354 * alpha
        elif 35 <= alpha < 55:
            return 2.5 + 0.075 * (alpha - 35)
        else:
            return 4 - 0.114 * (alpha - 55)

    # Calculate breakpoint distance d'_BP, fc is central fq in Hz
    def breakpoint_distance(self, h_BS, h_MS, fc_hz):
        h_BS_eff = h_BS - 1.0
        h_MS_eff = h_MS - 1.0
        d_BP = (4 * h_BS_eff * h_MS_eff * fc_hz) / config.C
        return d_BP

    """
    where d is the distance between the transmitter and the receiver in [m], fc is the system frequency in [GHz], the fitting parameter A includes the path-loss exponent,
    parameter B is the intercept, parameter C describes the path loss frequency dependence, and X is an optional, environment-specific term (e.g., wall attenuation in the A1 NLOS scenario).
    PL = Alog_10(d[m])+B+Clog_10(fc[GHz]/5.0)+X
    
    """

    def path_loss_los(self, d1, fc_ghz, h_BS, h_MS):
        if d1 <= 10:
            d1 = 11

        # Calculate effective antenna heights
        h_BS_eff = h_BS - 1.0
        h_MS_eff = h_MS - 1.0

        # Calculate breakpoint distance d'_BP
        d_BP = self.breakpoint_distance(h_BS, h_MS, fc_ghz * 1e9)

        if d1 < d_BP:
            path_loss = 22.7 * np.log10(d1) + 41.0 + 20 * np.log10(fc_ghz / 5.0)
        else:
            path_loss = (
                40 * np.log10(d1)
                + 9.45
                - 17.3 * np.log10(h_BS_eff)
                - 17.3 * np.log10(h_MS_eff)
                + 2.7 * np.log10(fc_ghz / 5.0)
            )

        return path_loss

    # usually
    # return path loss in db
    def path_loss_nlos(self, d1, d2, fc_ghz, h_BS, h_MS):
        if d1 <= 10:
            d1 = 11

        w = self.street_width  # Street width in meters
        if d2 <= w / 2:
            return self.path_loss_los(d1, fc_ghz, h_BS, h_MS)

        def pl(d1, d2):
            nj = max(2.8 - 0.0024 * d2, 1.84)
            pl = (
                self.path_loss_los(d1, fc_ghz, h_BS, h_MS)
                + 20
                - 12.5 * nj
                + 10 * nj * np.log10(d2)
                + 3 * np.log10(fc_ghz / 5.0)
            )
            return pl

        if w / 2 < d2 < 2000:
            return min(pl(d1, d2), pl(d2, d1))
        else:
            raise ValueError("d2 is out of the valid range.")

    def test(self):
        # Parameters
        h_BS = 10  # Base station height in meters
        h_MS = 1.5  # Mobile station height in meters
        fc_ghz = 5.905  # Frequency in GHz (5905 MHz)

        # Example distances
        d1_values = [10, 100, 500]  # Distances for LOS path loss calculation
        d2_values = [
            50,
            100,
            200,
            300,
            400,
            500,
        ]  # Distances for NLOS path loss calculation
        # Calculate and print LOS path loss for example distances
        print("LOS Path Loss:")
        for d1 in d1_values:
            pl_los = self.path_loss_los(d1, fc_ghz, h_BS, h_MS)
            print(f"d1: {d1} m, Path Loss: {pl_los:.2f} dB")

        # Calculate and print NLOS path loss for example distances
        print("\nNLOS Path Loss:")
        for d2 in d2_values:
            pl_nlos = self.path_loss_nlos(d1_values[0], d2, fc_ghz, h_BS, h_MS)
            print(f"d1: {d1_values[0]} m, d2: {d2} m, Path Loss: {pl_nlos:.2f} dB")

    # def channel_capacity(self, rsu: Rsu, vh: Vehicle):
    #     distance = rsu.real_distance(vh.position)
    #     path_loss = path_loss()
    #     received_power_dbm = transmitted_power_dbm - path_loss
    #     received_power_w = 10 ** ((received_power_dbm - 30) / 10)
    #     snr = received_power_w / noise_power
    #     channel_capacity = bandwidth * np.log2(1 + snr)
    #     return channel_capacity / 1e6  # 转换为 Mbps


# P_r = P_t - PL(d)
# SNR = P_r / N
# P_r need to convert to watt
def snr(rsu: Rsu, vh: Vehicle, path_loss_func="winner_b1"):
    # noise_power = dbm_to_watt(rsu.noise_power)
    # return (
    #     dbm_to_watt(rsu.transmitted_power - okumura_hata_path_loss(rsu, vh))
    #     / rsu.noise_power
    # )

    if path_loss_func == "winner_b1":
        d1, d2 = rsu.get_d1_d2(vh.get_position(), vh.get_angle())
        path_loss = WinnerB1().path_loss_nlos(
            d1, d2, rsu.frequency * 1e-3, rsu.height, vh.height
        )
    else:
        path_loss = OkumuraHata().okumura_hata_path_loss(rsu, vh)

    snr = dbm_to_watt(rsu.transmitted_power - path_loss) / rsu.noise_power
    return snr


# inverse Okumura-Hata path loss to get max connection distance


# universe max_distance algorithm
# not available for winner + b1 model
def max_distance(rsu: Rsu):
    max_distance = 0
    step = 0.001  # km
    distance = utils.real_distance_to_distance(step)

    while True:
        vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

        if snr(rsu, vh) < rsu.snr_threshold:
            break

        max_distance = distance
        distance += utils.real_distance_to_distance(step)

    return max_distance


def max_distance_mbps(rsu: Rsu, rate_tr=config.DATA_RATE_TR):
    max_distance = 0
    step = 1  # m
    distance = step

    while True:
        vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

        rate = channel_capacity(rsu, vh)
        if rate <= rate_tr:
            break

        max_distance = distance
        distance += step

    return max_distance


# not sure
def max_rate(rsu: Rsu):
    distance = 1
    vh = CustomVehicle(0, Point((rsu.position.x + distance, rsu.position.y)))

    max_rate = channel_capacity(rsu, vh)

    return max_rate


# convert db to dbm
def db_to_dbm(P_db):
    return P_db + 30


# convert dbm to watt
def dbm_to_watt(P_dbm):
    return 10 ** ((P_dbm - 30) / 10)


def bpsToMbps(bps):
    return bps / 1e6


# ofdm+mimo single channel
# Calculate data rate:
# C=Blog2​(1+SNR)
def channel_capacity(rsu: Rsu, vh: Vehicle, bw=config.RSU_MAX_TRANSMITTED_BANDWIDTH):

    return bpsToMbps(bw * np.log2(1 + snr(rsu, vh)))
