import sys

sys.path.append("./")

import os
from pettingzoo.utils.env import AgentID, ParallelEnv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np

from vanet_env.utils import RSU_MARKER, VEHICLE_MARKER


class Env(ParallelEnv):

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    # mobile rsu render, show road, vehicle and rsu, the wireless links with vehicle and rsu, the communicate range of rsu
    # matplotlib render
    def render(self):
        fig, ax = plt.subplots()

        # Draw the roads (four intersections)
        road1 = plt.Rectangle(
            (0, 4), 29, 2, linewidth=1, edgecolor="black", facecolor="gray"
        )
        road2 = plt.Rectangle(
            (0, 11.25), 29, 2, linewidth=1, edgecolor="black", facecolor="gray"
        )
        road3 = plt.Rectangle(
            (6.25, 0), 2, 18.5, linewidth=1, edgecolor="black", facecolor="gray"
        )
        road4 = plt.Rectangle(
            (20.75, 0), 2, 18.5, linewidth=1, edgecolor="black", facecolor="gray"
        )
        ax.add_patch(road1)
        ax.add_patch(road2)
        ax.add_patch(road3)
        ax.add_patch(road4)

        # RSU positions
        rsu_positions = [
            (6.25, 4),
            (6.25, 11.25),  # Intersection 1
            (13.5, 4),
            (20.75, 4),
            (20.75, 11.25),  # Intersection 2
            (13.5, 11.25),
        ]
        index = 0
        # Draw the RSUs and their communication ranges
        for pos in rsu_positions:
            index += 1
            ax.plot(
                pos[0],
                pos[1],
                marker=RSU_MARKER,
                markersize=30,
                markeredgewidth=0.1,
            )
            # Draw the RSU ID
            bs_id = f"RSU {index}"
            ax.annotate(
                bs_id,
                xy=(pos[0], pos[1]),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )
            # rsu = plt.Circle(pos, 0.3, color="red")
            # ax.add_patch(rsu)
            comm_range = plt.Circle(pos, 5, color="red", fill=False, linestyle="--")
            ax.add_patch(comm_range)

        # Draw the vehicle (example position)
        # vehicle positions (example positions for multiple vehicles)
        # the second vehicle may disappear, because the color of the vehicle simliar to the road
        vehicle_positions = [
            np.array([7.25, 5]),
            np.array([7.35, 8]),
            np.array([10.35, 5]),
            np.array([13.25, 12]),
            np.array([15.25, 12]),
            np.array([15.25, 5]),
        ]

        # Draw the vehicles
        # O(n^2) may take performance issue
        index = 0
        for vehicle_position in vehicle_positions:
            index += 1
            print(f"Drawing vehicle at position: {vehicle_position}")

            ax.plot(
                vehicle_position[0],
                vehicle_position[1],
                marker=VEHICLE_MARKER,
                markersize=40,
                markeredgewidth=0.1,
                zorder=3,
                color="blue",
            )
            vh_id = f"VH {index}"
            ax.annotate(
                vh_id,
                xy=(vehicle_position[0], vehicle_position[1]),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )

            # Draw the wireless links if connected (example condition)
            for pos in rsu_positions:
                distance = np.linalg.norm(vehicle_position - np.array(pos))
                if distance <= 5:  # If within communication range
                    plt.plot(
                        [vehicle_position[0], pos[0]],
                        [vehicle_position[1], pos[1]],
                        "r--",
                    )

        # Set limits and show plot
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 18)
        ax.set_aspect("equal")

        plt.show()
        pass


def test():
    env = Env()
    env.render()
    pass


test()
