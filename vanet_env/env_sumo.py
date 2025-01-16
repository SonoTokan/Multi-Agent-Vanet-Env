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
import traci
from sumolib import checkBinary


class Env(ParallelEnv):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode="human"):
        self.num_rsu = config.NUM_RSU
        self.num_vh = config.NUM_VEHICLES / 2
        self.max_size = config.MAP_SIZE
        self.road_width = config.ROAD_WIDTH
        self.render_mode = render_mode

        cfg_file_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "osm.sumocfg"
        )

        gui_settings_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "gui_hide_all.xml"
        )

        self.icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=config.RSU_POSITIONS[i],
            )
            for i in range(self.num_rsu)
        ]

        # for convience
        self.rsu_positions = [rsu.position for rsu in self.rsus]

        # canvas max distance
        self.max_distance = network.max_distance_mbps(self.rsus[0], 4)

        # start sumo sim
        if self.render_mode is not None:
            traci.start(["sumo-gui", "-c", cfg_file_path, "--step-length", "1"])
            # paint rsu
            for rsu in self.rsus:
                poi_id = f"rsu_icon_{rsu.id}"
                # add icon
                traci.poi.add(
                    poi_id,
                    rsu.position[0],
                    rsu.position[1],
                    (255, 0, 0, 255),
                    width=20,
                    height=20,
                    imgFile=self.icon_path,
                    layer=20,
                )
            # paint range
            for rsu in self.rsus:
                num_segments = 36
                for i in range(num_segments):
                    angle1 = 2 * np.pi * i / num_segments
                    angle2 = 2 * np.pi * (i + 1) / num_segments
                    x1 = rsu.position[0] + self.max_distance * np.cos(angle1)
                    y1 = rsu.position[1] + self.max_distance * np.sin(angle1)
                    x2 = rsu.position[0] + self.max_distance * np.cos(angle2)
                    y2 = rsu.position[1] + self.max_distance * np.sin(angle2)
                    traci.polygon.add(
                        f"circle_segment_rsu{rsu.id}_{i}",
                        [(x1, y1), (x2, y2)],
                        color=(255, 0, 0, 255),
                        fill=False,
                        lineWidth=0.2,
                        layer=20,
                    )
        else:
            # check SUMO bin
            sumoBinary = checkBinary("sumo")

            # start sumo no gui (for train)
            traci.start([sumoBinary, "-c", cfg_file_path, "--step-length", "1"])
        pass

    def reset(self, seed=None, options=None):
        traci.close()
        pass

    def step(self, action):
        # sumo sim step
        traci.simulationStep()
        pass

    def render(self, mode="human"):
        # human
        if mode is not None:
            # get veh ID
            vehicle_ids = traci.vehicle.getIDList()

            # clear all links
            for polygon_id in traci.polygon.getIDList():
                if polygon_id.startswith("line_rsu"):
                    traci.polygon.remove(polygon_id)

            for vehicle_id in vehicle_ids:
                # get veh pos
                vehicle_x, vehicle_y = traci.vehicle.getPosition(vehicle_id)

                for rsu in self.rsus:
                    rsu_x = rsu.position[0]
                    rsu_y = rsu.position[1]
                    # cal dis between rsu and veh
                    distance = np.sqrt(
                        (vehicle_x - rsu_x) ** 2 + (vehicle_y - rsu_y) ** 2
                    )

                    # linking if in rsu range
                    if distance <= self.max_distance:
                        traci.polygon.add(
                            f"line_rsu{rsu.id}_to_{vehicle_id}",
                            [(rsu_x, rsu_y), (vehicle_x, vehicle_y)],
                            color=(0, 255, 0, 255),
                            fill=False,
                            lineWidth=0.3,
                            layer=30,
                        )
            return
        else:
            return
