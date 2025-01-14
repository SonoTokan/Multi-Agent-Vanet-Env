import sys

sys.path.append("./")

import cProfile
from pettingzoo.test import parallel_api_test
from vanet_env.env import Env
from vanet_env.entites import Rsu, Vehicle
from vanet_env import utils
from network import channel_capacity


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
    step = 0.01
    rsu = Rsu(1, (0, 0))
    while step <= 0.5:
        vh = Vehicle(1, (utils.realDistanceToDistance(step), 0))
        print(f"real_distance: {step:.2f} km, {channel_capacity(rsu, vh):.2f} Mbps")
        step += 0.05
    pass


def render_test():
    env = Env(3)
    for i in range(5):
        env.render()


def test():
    env = Env(3)
    env.test()


if __name__ == "__main__":
    test()
    network_test()

    # print_stats()
    pass
    # env = Env()
    # parallel_api_test(env, num_cycles=1_000_000)

    # env = Env()
    # parallel_api_test(env, num_cycles=1_000_000)
