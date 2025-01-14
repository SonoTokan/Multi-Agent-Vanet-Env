import sys

sys.path.append("./")

import os
from pettingzoo.utils.env import AgentID, ParallelEnv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np

from vanet_env import config, network
from vanet_env.utils import RSU_MARKER, VEHICLE_MARKER
from vanet_env.entites import Rsu, Vehicle


class Env(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_pause_time=0.001):
        self.num_rsu = config.NUM_RSU
        self.num_vh = config.NUM_VEHICLES / 2
        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=config.RSU_POSITIONS[i],
            )
            for i in range(self.num_rsu)
        ]
        # init vhs
        self.vehicles = [
            Vehicle(
                id=i,
                position=(0, 0),
            )
            for i in range(int(self.num_vh))
        ]
        # init roads
        road1 = plt.Rectangle((0, 4), 29, 2, linewidth=1, facecolor="gray")
        road2 = plt.Rectangle((0, 11.25), 29, 2, linewidth=1, facecolor="gray")
        road3 = plt.Rectangle((6.25, 0), 2, 18.5, linewidth=1, facecolor="gray")
        road4 = plt.Rectangle((20.75, 0), 2, 18.5, linewidth=1, facecolor="gray")
        self.roads = [road1, road2, road3, road4]

        # exsample, random vehicle position using random_point_within_rectangle
        vehicle_positions = [
            self.random_point_within_rectangle(
                self.roads[np.random.randint(0, len(self.roads))]
            )
            for _ in range(int(self.num_vh // 2))
        ]

        # copy to object, may take some performance issue
        self.vehicles = [
            Vehicle(id=i, position=pos) for i, pos in enumerate(vehicle_positions)
        ]

        # for convience
        self.rsu_positions = [rsu.position for rsu in self.rsus]

        # canvas max distance
        self.max_distance = network.max_distance(self.rsus[0])

        # render parameters
        self.pause_time = render_pause_time
        self.fig, self.ax = plt.subplots()

        # Set limits
        self.ax.set_xlim(0, 30)
        self.ax.set_ylim(0, 18)
        self.ax.set_aspect("equal")
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    # mobile rsu render, show road, vehicle and rsu, the wireless links with vehicle and rsu, the communicate range of rsu
    # matplotlib render
    def render(self, mode="human"):
        self.ax.clear()

        # rsu and road may not draw everytime, can set when step > 1 not draw road and rsu

        self._render_road(self.fig, self.ax, self.roads)
        self._render_rsu(self.fig, self.ax, self.rsus)
        self._render_vehicle(self.fig, self.ax, self.vehicles, self.rsus)

        plt.draw()
        plt.pause(self.pause_time)
        pass

    # random point within rectangle, and the range like this code
    def random_point_within_rectangle(self, rect):
        x = np.random.uniform(rect.get_x(), rect.get_x() + rect.get_width())
        y = np.random.uniform(rect.get_y(), rect.get_y() + rect.get_height())
        return (x, y)

    # render the vehicle and the wireless links with rsu
    def _render_vehicle(self, fig, ax, vhs: list, rsus: list):
        # Draw the vehicles
        for vh in vhs:
            print(f"Drawing vehicle at position: {vh.position}")

            ax.plot(
                vh.position[0],
                vh.position[1],
                marker=VEHICLE_MARKER,
                markersize=40,
                markeredgewidth=0.1,
                zorder=3,
                color="blue",
            )
            vh_id = f"VH {vh.id}"
            ax.annotate(
                vh_id,
                xy=(vh.position[0], vh.position[1]),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )

            # logic modify to step later
            # O(n^2) may take performance issue
            # Draw the wireless links if connected (example condition)
            # link color shows QoE, green for good, yellow for normal, red for poor
            for idx, rsu in enumerate(rsus):
                # canvas distance
                distance = np.linalg.norm(vh.position - np.array(rsu.position))
                if distance <= self.max_distance:  # If within communication range
                    self.rsus[idx].connected_vehicles.append(vh)
                    plt.plot(
                        [vh.position[0], rsu.position[0]],
                        [vh.position[1], rsu.position[1]],
                        "g--",
                    )

    # render the road
    def _render_road(self, fig, ax, roads):
        # Draw the roads (four intersections)
        # removed edgecolor="black"

        for road in roads:
            ax.add_patch(road)

    # render the rsus
    def _render_rsu(self, fig, ax, rsus: list):
        # RSU positions

        # Draw the RSUs and their communication ranges
        for idx, rsu in enumerate(rsus):
            ax.plot(
                rsu.position[0],
                rsu.position[1],
                marker=RSU_MARKER,
                markersize=30,
                markeredgewidth=0.1,
            )
            # Draw the RSU ID
            bs_id = f"RSU {idx+1}"
            ax.annotate(
                bs_id,
                xy=(rsu.position[0], rsu.position[1]),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )
            # rsu = plt.Circle(pos, 0.3, color="red")
            # ax.add_patch(rsu)
            comm_range = plt.Circle(
                rsu.position, self.max_distance, color="red", fill=False, linestyle="--"
            )
            ax.add_patch(comm_range)

    def test(self):
        self.render()
        # channel capacity between rsu and vehicle
        for rsu in self.rsus:
            for vh in rsu.connected_vehicles:
                print(
                    f"rsu {rsu.id} - vh {vh.id} distance {rsu.real_distance(vh.position)} km, channel capacity {network.channel_capacity(rsu, vh) } Mbps"
                )
        pass
