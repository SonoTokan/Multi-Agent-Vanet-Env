import io
import os
import pstats
import sys

sys.path.append("./")

import cProfile
from pettingzoo.test import parallel_api_test
from env_sumo import Env
from vanet_env.entites import Rsu, Vehicle
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
        vh = Vehicle(1, (utils.realDistanceToDistance(step), 0))
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
# fps 144?
def sumo_env_test():
    # render_mode="human", None
    env = Env(None)
    for i in range(3600):
        env.step([])
        env.render()


if __name__ == "__main__":
    cProfile.run("sumo_env_test()")
    pass
