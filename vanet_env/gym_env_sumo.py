import sys

from shapely import Point

sys.path.append("./")

import os
from pettingzoo.utils.env import AgentID, ParallelEnv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
from scipy.spatial import KDTree

from vanet_env import config, network
from vanet_env.utils import RSU_MARKER, VEHICLE_MARKER, interpolate_color, sumo_detector
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle
import traci
from sumolib import checkBinary
import multiprocessing


class Env(ParallelEnv):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode="human"):
        # SUMO detector
        sumo_detector()

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
                position=Point(config.RSU_POSITIONS[i]),
            )
            for i in range(self.num_rsu)
        ]

        # max data rate
        self.max_data_rate = network.max_rate(self.rsus[0])
        print(f"single atn max_data_rate:{self.max_data_rate:.2f}")

        # for convience
        self.rsu_positions = [rsu.position for rsu in self.rsus]
        rsu_coords = np.array(
            [
                (self.rsu_positions[rsu.id].x, self.rsu_positions[rsu.id].y)
                for rsu in self.rsus
            ]
        )
        self.rsu_tree = KDTree(rsu_coords)

        # network
        self.connections = []

        # canvas max distance
        self.max_distance = network.max_distance_mbps(self.rsus[0], 4)

        # start sumo sim
        if self.render_mode is not None:
            traci.start(
                ["sumo-gui", "-c", cfg_file_path, "--step-length", "1", "--start"]
            )
            # paint rsu
            for rsu in self.rsus:
                poi_id = f"rsu_icon_{rsu.id}"
                # add rsu icon
                traci.poi.add(
                    poi_id,
                    rsu.position.x,
                    rsu.position.y,
                    (205, 254, 194, 255),  # green
                    width=10,
                    height=10,
                    imgFile=self.icon_path,
                    layer=20,
                )
            # paint range
            for rsu in self.rsus:
                num_segments = 36
                for i in range(num_segments):
                    angle1 = 2 * np.pi * i / num_segments
                    angle2 = 2 * np.pi * (i + 1) / num_segments
                    x1 = rsu.position.x + self.max_distance * np.cos(angle1)
                    y1 = rsu.position.y + self.max_distance * np.sin(angle1)
                    x2 = rsu.position.x + self.max_distance * np.cos(angle2)
                    y2 = rsu.position.y + self.max_distance * np.sin(angle2)
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
            traci.start([sumoBinary, "-c", cfg_file_path])
        pass

    def reset(self, seed=None, options=None):
        traci.close()
        pass

    def step(self, action):
        # sumo sim step
        traci.simulationStep()

        self.vehicle_ids = traci.vehicle.getIDList()
        # veh objs
        self.vehicles = [Vehicle(vehicle_id) for vehicle_id in self.vehicle_ids]

        self._manage_rsu_vehicle_connections()
        pass

    # def _connection(self):

    # peformance issue
    def _manage_rsu_vehicle_connections(self, has_tree=True):
        """
        connection logical
        """
        # clear connections
        for rsu in self.rsus:
            rsu.connections = []
            self.connections = []

        # KDtree
        if has_tree == False:
            for veh in self.vehicles:
                vehicle_x, vehicle_y = veh.position.x, veh.position.y

                for rsu in self.rsus:

                    rsu_x, rsu_y = (
                        self.rsu_positions[rsu.id].x,
                        self.rsu_positions[rsu.id].y,
                    )
                    distance = np.sqrt(
                        (vehicle_x - rsu_x) ** 2 + (vehicle_y - rsu_y) ** 2
                    )

                    if distance <= self.max_distance:
                        conn = Connection(rsu, veh)
                        rsu.connections.append(conn)
                        self.connections.append(conn)
        else:

            # connections update
            for veh in self.vehicles:
                vehicle_x, vehicle_y = veh.position.x, veh.position.y
                vehicle_coord = np.array([vehicle_x, vehicle_y])

                indices = self.rsu_tree.query_ball_point(
                    vehicle_coord, self.max_distance
                )

                for idx in indices:
                    rsu = self.rsus[idx]
                    conn = Connection(rsu, veh)
                    rsu.connections.append(conn)
                    self.connections.append(conn)

        # bw policy here
        # not sure right or not
        # connections info update

        for conn in self.connections:
            rsu = conn.rsu
            veh = conn.veh
            data_rate = (
                network.channel_capacity(rsu, veh) * rsu.num_atn / len(rsu.connections)
            )

            conn.data_rate = data_rate

    # improve：返回polygons然后后面统一绘制？
    def _render_connections(self):
        # Batch add polygons
        polygons_to_add = []

        for conn in self.connections:
            color = interpolate_color(
                config.DATA_RATE_TR, self.max_data_rate, conn.data_rate
            )

            color_with_alpha = (*color, 255)

            polygons_to_add.append(
                (
                    f"dynamic_line_rsu{conn.rsu.id}_to_{conn.veh.vehicle_id}",
                    [
                        (conn.rsu.position.x, conn.rsu.position.y),
                        (conn.veh.position.x, conn.veh.position.y),
                    ],
                    color_with_alpha,
                )
            )

        # for rsu in self.rsus:
        #     num_connected = len(rsu.connected_vehicles)
        #     rsu_x, rsu_y = ...

        #     for veh in rsu.connected_vehicles:
        #         vehicle_x, vehicle_y = veh.get_position()

        #         color = interpolate_color(
        #             config.DATA_RATE_TR, self.max_data_rate, connection_data_rate
        #         )

        #         color_with_alpha = (*color, 255)

        #         polygons_to_add.append(
        #             (
        #                 f"dynamic_line_rsu{rsu.id}_to_{veh.vehicle_id}",
        #                 [(rsu_x, rsu_y), (vehicle_x, vehicle_y)],
        #                 color_with_alpha,
        #             )
        #         )

        # Add all polygons at once
        for polygon_id, points, color in polygons_to_add:
            traci.polygon.add(
                polygon_id, points, color=color, fill=False, lineWidth=0.3, layer=30
            )

    def render(self, mode=None):
        mode = self.render_mode if mode is None else mode
        # human
        if mode is not None:
            # get veh ID

            # clear all dynamic rendered polygon
            for polygon_id in traci.polygon.getIDList():
                if polygon_id.startswith("dynamic_"):
                    traci.polygon.remove(polygon_id)

            self._render_connections()

            return
        else:
            return
