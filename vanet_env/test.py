import io
import os
import pstats
import sys

import numpy as np
from shapely import Point

sys.path.append("./")

import cProfile
from pettingzoo.test import parallel_api_test
from vanet_env.gym_env_sumo import Env
from vanet_env.entites import Rsu, CustomVehicle
from vanet_env import utils
from vanet_env import network
from network import channel_capacity
import osmnx as ox
import matplotlib.pyplot as plt


def run_time_test():
    cProfile.run("channel_capacity(d, 20)", "restats_channel_capacity.restats")
    cProfile.run("render_test()", "restats_render_test.restats")


def print_stats():
    import pstats

    p = pstats.Stats("restats_channel_capacity.restats")
    p.sort_stats("cumulative").print_stats(10)
    p = pstats.Stats("restats_render_test.restats")
    p.sort_stats("cumulative").print_stats(10)


def network_test():
    # real distance (km)
    step = 0.001
    rsu = Rsu(1, (0, 0))
    while step <= 0.5:
        vh = CustomVehicle(1, Point((utils.real_distance_to_distance(step), 0)))
        print(f"real_distance: {step*1000:.2f} m, {channel_capacity(rsu, vh):.2f} Mbps")
        step += 0.01
    pass


def path_loss_test():
    winnerb1 = network.WinnerB1()
    winnerb1.test()


def render_test():
    env = Env(3)
    for i in range(5):
        env.render()


def test():
    env = Env(3)
    env.test()


def osmx_test():

    file_path = os.path.join(os.path.dirname(__file__), "assets", "seattle", "map.osm")
    G = ox.graph_from_xml(file_path)

    fig, ax = ox.plot_graph(G, node_size=5, edge_linewidth=0.5)
    plt.show()


# 3600s takes 25 seconds if render_mode = None
# 112,293,666 function calls in 97.557 seconds if render_mode = None when take _manage_rsu_vehicle_connections()
# 13864794 function calls in 6.782 seconds if render_mode = None without _manage_rsu_vehicle_connections()
# 3600s takes 105.441 seconds if render_mode = "human"
# 3600s takes 137.505 seconds if render lines by logic
# 3600s takes 136.182 seconds if using kdTree
# 3600s: 93277269 function calls in 125.288 seconds if using kdTree
# 3600s: 135685748 function calls in 111.391 seconds using new logic and min window
# 500s takes 16.412 seconds if render lines by logic
# 500s takes 16.939 seconds if using kdTree
# 500s takes 17.127 seconds using new logic
# 500s takes 11 seconds new render
# + queue list
# 11,405,122 function calls (11404904 primitive calls) in 15.304 seconds
# + lots of logics
# 12,107,925 function calls (12107707 primitive calls) in 14.846 seconds
# + lots of logics
# 12,041,754 function calls (12041532 primitive calls) in 11.103 seconds
# 100 sim_step * fps(10)
# 683965 function calls (683735 primitive calls) in 2.919 seconds
# None render
# 500 step-normal: 1,920,955 function calls in 1.502 seconds
# 500 step-getPos: 2,725,563 function calls in 4.650 seconds
# 500 step-getPos-logic: 12,153,777 function calls in 10.417 seconds
# 500 step-getPos-hasTree-logic: 8,218,740 function calls in 7.415 seconds
# 500 step-getPos-hasTree-logic-delete-render(): 3,926,358 function calls in 4.180 seconds
# 500 step-getPos-hasTree-logic-render()-init_all(): 3,516,416 function calls in 4.055 seconds
# 500 step-logic: 14,373,235 function calls in 12.490 seconds
# 500 step-getPos-hasTree-logic-render()-init_all() + Simulation version 1.21.0 started via libsumo with time: 0.00.
# 1,681,262 function calls in 1.604 seconds
# + orderd rsu conn list
# 1,995,522 function calls (1995304 primitive calls) in 2.147 seconds
# + orderd rsu conn list logic 2
# 2,213,830 function calls (2213612 primitive calls) in 1.877 seconds
# + queue list
# 2,152,464 function calls (1973712 primitive calls) in 1.983 seconds
# no empty determine
# 2,364,018 function calls (2363800 primitive calls) in 2.033 seconds
# + veh logics
# 2,652,298 function calls (2652080 primitive calls) in 2.224 seconds
# + frame step
# 1,867,325 function calls (1867095 primitive calls) in 1.448 seconds
# 3600 steps
# 13,356,345 function calls in 11.738 seconds


# fps 144?
def sumo_env_test():
    fps = 10
    # render_mode="human", None
    env = Env(None)
    obs, infos = env.reset()

    for i in range(100 * fps):
        env.step({})


if __name__ == "__main__":
    # cProfile.run("sumo_env_test()", sort="time")
    # sumo_env_test()
    is_full = np.array([True, False, False])
    is_in = np.array([False, True, False])
    update_mask = is_full & is_in  # 需要更新的 RSU
    store_mask = ~is_full & ~is_in  # 需要存入的 RSU
    valid_mask = update_mask | store_mask
    print(update_mask)
    print(store_mask)
    print(valid_mask)

    pass
