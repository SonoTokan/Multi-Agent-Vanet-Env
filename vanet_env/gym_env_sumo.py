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

# gym
import gymnasium as gym
from gymnasium import spaces

# custom
from vanet_env import config, network
from vanet_env.utils import RSU_MARKER, VEHICLE_MARKER, interpolate_color, sumo_detector
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle
from test import multiprocess as mp_funcs

# sumo
import traci
import libsumo
from sumolib import checkBinary

# next performance improvement
import multiprocessing as mp


class Env(ParallelEnv):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode="human", multi_core=8):
        # SUMO detector
        sumo_detector()

        self.num_rsu = config.NUM_RSU
        self.num_vh = config.NUM_VEHICLES / 2
        self.max_size = config.MAP_SIZE
        self.road_width = config.ROAD_WIDTH
        self.render_mode = render_mode
        self.multi_core = multi_core

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
        self.sumo = traci

        # sapces
        self._space_init()

        # start sumo sim
        if self.render_mode is not None:

            self.sumo.start(
                ["sumo-gui", "-c", cfg_file_path, "--step-length", "1", "--start"]
            )
            # paint rsu
            for rsu in self.rsus:
                poi_id = f"rsu_icon_{rsu.id}"
                # add rsu icon
                self.sumo.poi.add(
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
                    self.sumo.polygon.add(
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
            self.sumo = libsumo
            libsumo.start(["sumo", "-c", cfg_file_path])
            # traci.start([sumoBinary, "-c", cfg_file_path])
        pass

    def _space_init(self, aggregated=False):

        compute_resources_low = 0.0
        compute_resources_high = 1.0
        bandwidth_low = 0.0
        bandwidth_high = 1.0
        job_state_low = 0.0
        job_state_high = 1.0
        connection_state_low = 0.0
        connection_state_high = 1.0
        # important
        self.max_connections = 10
        self.max_content = 100
        self.max_caching = 10
        # rsu range veh wait for connections
        self.max_queue_len = 10
        # every single weight
        self.max_weight = 100

        # state need veh、job info

        # aggregated
        if aggregated:
            self.single_state_space = spaces.Dict(
                {
                    "compute_resources": spaces.Box(
                        low=compute_resources_low,
                        high=compute_resources_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                    "bandwidth": spaces.Box(
                        low=bandwidth_low,
                        high=bandwidth_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                    "avg_job_state": spaces.Box(
                        low=job_state_low,
                        high=job_state_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                    "avg_connection_state": spaces.Box(
                        low=connection_state_low,
                        high=connection_state_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                }
            )

            initial_state = {
                "compute_resources": np.random.uniform(
                    compute_resources_low, compute_resources_high, self.num_rsu
                ),
                "bandwidth": np.random.uniform(
                    bandwidth_low, bandwidth_high, self.num_rsu
                ),
                "avg_job_state": np.random.uniform(
                    job_state_low, job_state_high, self.num_rsu
                ),
                "avg_connection_state": np.random.uniform(
                    connection_state_low, connection_state_high, self.num_rsu
                ),
            }

        else:
            self.single_state_space = spaces.Dict(
                {
                    "compute_resources": spaces.Box(
                        low=compute_resources_low,
                        high=compute_resources_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                    "bandwidth": spaces.Box(
                        low=bandwidth_low,
                        high=bandwidth_high,
                        shape=(self.num_rsu,),
                        dtype=np.float32,
                    ),
                    "job_state": spaces.Box(
                        low=job_state_low,
                        high=job_state_high,
                        shape=(self.num_rsu, self.max_connections),
                        dtype=np.float32,
                    ),
                    "connection_state": spaces.Box(
                        low=connection_state_low,
                        high=connection_state_high,
                        shape=(self.num_rsu, self.max_connections),
                        dtype=np.float32,
                    ),
                }
            )
            ...
        job_handling_space = spaces.MultiDiscrete(
            [self.num_rsu + 2] * self.max_connections
        )

        # connection policy
        connection_space = spaces.MultiBinary(self.max_connections)

        # compute policy
        compute_allocation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.max_connections,), dtype=np.float32
        )

        # bw policy
        bandwidth_allocation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.max_connections,), dtype=np.float32
        )

        # caching policy
        caching_decision_space = spaces.MultiDiscrete(
            [self.max_content] * self.max_caching
        )

        # tuple action space
        self.tuple_action_space = spaces.Tuple(
            (
                job_handling_space,
                connection_space,
                compute_allocation_space,
                bandwidth_allocation_space,
            )
        )

        # Dict action space
        self.dict_action_space = spaces.Dict(
            {
                "job_handling": job_handling_space,
                "connection": connection_space,
                "compute_allocation": compute_allocation_space,
                "bandwidth_allocation": bandwidth_allocation_space,
                "caching_decision": caching_decision_space,
            }
        )

        # MultiDiscrete action space
        # job_handling
        # Action selection is followed by normalization to ensure that the sum of allocated resources is 1
        md_jh_space_shape = spaces.MultiDiscrete([2] * self.max_connections)
        # computing power allocation, value means weight
        md_ca_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # bandwidth allocation
        md_bw_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # caching_decision
        md_cd_space = spaces.MultiDiscrete([self.max_content] * self.max_caching)

        # combined
        md_combined_space = spaces.MultiDiscrete(
            [2] * self.max_connections
            + [self.max_weight] * self.max_connections
            + [self.max_weight] * self.max_connections
            + [self.max_content] * self.max_caching
        )

        self.md_action_space = md_combined_space

        print(
            f"action sample{self.md_action_space.sample()}, obs sample{self.single_state_space.sample()}"
        )

    def reset(self, seed=None, options=None):
        # may not need
        self.sumo.close()
        pass

    def step(self, action):
        # sumo sim step
        self.sumo.simulationStep()

        self.vehicle_ids = self.sumo.vehicle.getIDList()
        # veh objs 25% time spent
        self.vehicles = [
            Vehicle(vehicle_id, self.sumo) for vehicle_id in self.vehicle_ids
        ]

        self._manage_rsu_vehicle_connections()
        self._manage_resource()
        pass

    # def _connection(self):

    def _manage_resource(self): ...

    # bw policy here
    # peformance issue
    def _manage_rsu_vehicle_connections(self, kdtree=True):
        """
        connection logical
        """
        # clear connections
        self.connections = []
        for rsu in self.rsus:
            rsu.connections = []

        # KDtree
        if kdtree == False:
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

        # with mp.Pool() as pool:
        #     self.connections = pool.map(mp_funcs.update_connection, self.connections)
        # with Pool(self.multi_core) as p:
        #     p.map(mp.update_connection, self.connections)

        # connections info update
        # bw policy here
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
            self.sumo.polygon.add(
                polygon_id, points, color=color, fill=False, lineWidth=0.3, layer=30
            )

    def render(self, mode=None):
        mode = self.render_mode if mode is None else mode
        # human
        if mode is not None:
            # get veh ID

            # clear all dynamic rendered polygon
            for polygon_id in self.sumo.polygon.getIDList():
                if polygon_id.startswith("dynamic_"):
                    self.sumo.polygon.remove(polygon_id)

            self._render_connections()

            return
        else:
            return
