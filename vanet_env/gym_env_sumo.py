import functools
import sys

from shapely import Point

sys.path.append("./")

import os
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
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
from vanet_env.utils import (
    RSU_MARKER,
    VEHICLE_MARKER,
    interpolate_color,
    sumo_detector,
    is_empty,
)
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle

# sumo
import traci
import libsumo
from sumolib import checkBinary

# next performance improvement
import multiprocessing as mp


# def env(render_mode=None):
#     internal_render_mode = render_mode if render_mode != "ansi" else "human"
#     env = raw_env(render_mode=internal_render_mode)
#     # This wrapper is only for environments which print results to the terminal
#     if render_mode == "ansi":
#         env = wrappers.CaptureStdoutWrapper(env)
#     # this wrapper helps error handling for discrete action spaces
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     # Provides a wide vareity of helpful user errors
#     # Strongly recommended
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env


def raw_env(render_mode=None):
    env = Env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class Env(ParallelEnv):
    metadata = {"name": "sumo_vanet_environment_v0", "render_modes": ["human", None]}

    def __init__(
        self, render_mode="human", max_step=3600, multi_core=8, caching_step=100
    ):

        self.num_rsu = config.NUM_RSU
        self.num_vh = config.NUM_VEHICLES / 2
        self.max_size = config.MAP_SIZE  # deprecated
        self.road_width = config.ROAD_WIDTH
        self.render_mode = render_mode
        self.multi_core = multi_core
        self.max_step = max_step
        self.caching_step = caching_step

        # important
        self.max_connections = 10
        self.max_content = 100
        self.max_caching = 10
        # rsu range veh wait for connections
        self.max_queue_len = 10
        # every single weight
        self.max_weight = 100

        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=Point(config.RSU_POSITIONS[i]),
                max_connection=self.max_connections,
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
        self.connections_queue = []

        # rsu max connection distance
        self.max_distance = network.max_distance_mbps(self.rsus[0], 4)
        self.sumo = traci

        # pettingzoo init
        agents = ["rsu_" + str(i) for i in range(self.num_rsu)]
        self.possible_agents = agents[:]
        self.time_step = 0

        self._sumo_init()
        self._space_init()

    def _sumo_init(self):
        # SUMO detector
        sumo_detector()

        cfg_file_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "osm.sumocfg"
        )

        gui_settings_path = os.path.join(
            os.path.dirname(__file__), "assets", "seattle", "sumo", "gui_hide_all.xml"
        )

        self.icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

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

        net_boundary = self.sumo.simulation.getNetBoundary()
        self.map_size = net_boundary[1]

    def _space_init(self):

        # state need veh、job info

        # aggregated
        self.global_ag_state_space = spaces.Dict(
            {
                "compute_resources": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "avg_job_state": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "avg_connection_state": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "job_info": spaces.Box(  # normalized veh job info (size) in connection range
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu, self.max_connections),
                    dtype=np.float32,
                ),
                "veh_x": spaces.Box(  # vehicle x, normalize to 0-1, e.g. 200 -> 0.5, 400 -> 1
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu, self.max_connections),
                    dtype=np.float32,
                ),
                "veh_y": spaces.Box(  # vehicle y, normalize to 0-1
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu, self.max_connections),
                    dtype=np.float32,
                ),
            }
        )

        # not aggregated
        self.local_ng_state_space = spaces.Dict(
            {
                "compute_resource": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "job_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
                "connection_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
                "job_info": spaces.Box(  # normalized veh job info (size) in connection range
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
                "veh_x": spaces.Box(  # vehicle x, normalize to 0-1, e.g. 200 -> 0.5, 400 -> 1
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
                "veh_y": spaces.Box(  # vehicle y, normalize to 0-1
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
            }
        )

        # not aggregated multi discrete local space
        self.local_ng_md_state_space = spaces.MultiDiscrete(
            [self.max_weight] * 2 + [self.max_weight] * self.max_connections * 5
        )

        # aggregated multi discrete global space, veh and job info not aggregated
        self.global_ag_md_state_space = spaces.MultiDiscrete(
            [self.max_weight] * self.num_rsu * 4
            + [self.max_weight] * self.max_connections * self.num_rsu * 3
        )

        # not aggregated
        self.global_ng_state_space = spaces.Dict(
            {
                "compute_resources": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu,),
                    dtype=np.float32,
                ),
                "job_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu, self.max_connections),
                    dtype=np.float32,
                ),
                "connection_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsu, self.max_connections),
                    dtype=np.float32,
                ),
            }
        )

        # not aggregated
        self.local_ng_state_space = spaces.Dict(
            {
                "compute_resource": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "job_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
                "connection_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_connections,),
                    dtype=np.float32,
                ),
            }
        )

        # only this space tested
        # mixed: for every single agent, global aggregated but local not
        # every rsu's avg bw, cp, job, conn state and veh pos, job info in this rsu's range
        self.mixed_md_observation_space = spaces.MultiDiscrete(
            [self.max_weight] * self.num_rsu * 4
            + [self.max_weight] * self.max_connections
            + [int(self.map_size[0])] * self.max_connections
            + [int(self.map_size[1])] * self.max_connections
        )

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

        # # MultiDiscrete action space
        # # job_handling
        # # Action selection is followed by normalization to ensure that the sum of allocated resources is 1
        # md_jh_space_shape = spaces.MultiDiscrete([2] * self.max_connections)
        # # computing power allocation, value means weight
        # md_ca_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # # bandwidth allocation
        # md_bw_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # # caching_decision
        # md_cd_space = spaces.MultiDiscrete([self.max_content] * self.max_caching)

        # combined
        md_combined_space = spaces.MultiDiscrete(
            [2] * self.max_connections
            + [self.max_weight] * self.max_connections
            + [self.max_weight] * self.max_connections
            + [self.max_content] * self.max_caching
        )

        self.md_single_action_space = md_combined_space

        print(
            f"action sample{self.md_single_action_space.sample()}, obs sample{self.mixed_md_observation_space.sample()}"
        )

    def reset(self, seed=None, options=None):

        self.agents = np.copy(self.possible_agents)
        self.time_step = 0
        # every rsu's avg bw, cp, job, conn state and veh, job info in this rsu's range
        self.mixed_md_observation_space = spaces.MultiDiscrete(
            [self.max_weight] * self.num_rsu * 4
            + [self.max_weight] * self.max_connections * 3
        )

        observation = np.zeros(
            self.num_rsu * 4
            + self.max_connections
            + self.max_connections
            + self.max_connections
        ).tolist()  # not sure need or not

        # do not take any action except caching at start
        observations = {
            agent: {
                "observation": observation,
                "action_mask": self._single_action_mask(idx),
            }
            for idx, agent in enumerate(self.agents)
        }

        # parallel_to_aec conversion may needed
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        self.vehicle_ids = self.sumo.vehicle.getIDList()
        # veh objs 25% time spent
        self.vehicles = [
            Vehicle(vehicle_id, self.sumo) for vehicle_id in self.vehicle_ids
        ]

        self._update_connections_queue()

        # take action
        self._take_actions(actions)

        # sumo sim step
        self.sumo.simulationStep()

        # update observation space
        self._update_observations()

    def _single_action_mask(self, rsu_idx):
        connection_action_mask = []
        resource_manager_action_mask = []

        # for c in self.rsus[rsu_idx].connections:
        #     if c is None:
        #         connection_action_mask.append(0)
        #         resource_manager_action_mask.append(0)
        #         continue

        #     if c.connected:
        #         connection_action_mask.append(0)
        #         resource_manager_action_mask.append(1)
        #     else:
        #         resource_manager_action_mask.append(1)
        #         resource_manager_action_mask.append(0)

        # if None or connected do not need connect
        connection_action_mask = [
            0 if c is None or c.connected else 1 for c in self.rsus[rsu_idx].connections
        ]
        # if None or not connected do not need resource manage
        # so only connected need
        resource_manager_action_mask = [
            0 if c is None or not c.connected else 1
            for c in self.rsus[rsu_idx].connections
        ]

        caching_action_mask = [0] * self.max_caching

        if self.time_step % self.caching_step == 0:
            caching_action_mask = [1] * self.max_caching

        # 1 * connction + 2 * resourceman + caching
        return (
            connection_action_mask
            + resource_manager_action_mask * 2
            + caching_action_mask
        )

    def _take_actions(self, actions):
        # action rsu 0 -- n
        # action mask based on veh and rsu state
        for idx, action in enumerate(list(actions.values())):
            rsu = self.rsus[idx]

        self._manage_resources()
        pass

    def _update_observations(self):

        pass

    def _manage_resources(self):
        # connections info update
        # bw policy here
        for conn in self.connections_queue:
            rsu = conn.rsu
            veh = conn.veh
            data_rate = (
                network.channel_capacity(rsu, veh)
                * rsu.num_atn
                / len(rsu.connections.olist)
            )

            conn.data_rate = data_rate
        pass

    # peformance issue
    def _update_connections_queue(self, kdtree=True):
        """
        connection logical
        """
        # clear connections
        self.connections_queue = []
        # rsu not clear, just update
        # for rsu in self.rsus:
        #     rsu.connections = []

        # KDtree
        # disordered issue
        if kdtree == False:
            # not implement!
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
                        self.connections_queue.append(conn)
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
                    # update rsu conn queue
                    matching_connection = [c for c in rsu.connections if c == conn]
                    # if exist, update veh info else append
                    if matching_connection:
                        matching_connection[0].veh = veh
                    else:
                        rsu.connections.append(conn)

                    self.connections_queue.append(conn)

                # conn remove logical #1
                # check if veh out

                # for idx, rsu in enumerate(self.rsus):
                #     # empty
                #     if not rsu.connections:
                #         continue

                #     # veh is out of range
                #     if idx not in indices:
                #         rsu.connections = [
                #             c
                #             for c in rsu.connections
                #             if c.veh.vehicle_id != veh.vehicle_id
                #         ]

            # conn remove logical #2
            for rsu in self.rsus:
                for idx, c in enumerate(rsu.connections):
                    if c is None:
                        continue

                    if c not in self.connections_queue:
                        rsu.connections.remove(idx)
            # check if veh out
            # issue
            # for rsu in self.rsus:
            #     for i in range(len(rsu.connections) - 1, -1, -1):
            #         conn = rsu.connections[i]
            #         # may not right
            #         indices = self.rsu_tree.query_ball_point(
            #             np.array([conn.veh.position.x, conn.veh.position.y]),
            #             self.max_distance,
            #         )
            #         # not in range
            #         if len(indices) == 0:
            #             del rsu.connections[i]
        # with mp.Pool() as pool:
        #     self.connections = pool.map(mp_funcs.update_connection, self.connections)
        # with Pool(self.multi_core) as p:
        #     p.map(mp.update_connection, self.connections)

    # improve：返回polygons然后后面统一绘制？
    def _render_connections(self):
        # Batch add polygons
        polygons_to_add = []

        # not correct now
        for conn in self.connections_queue:
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

    def close(self):
        self.sumo.close()
        return super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        return self.mixed_md_observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.md_single_action_space
