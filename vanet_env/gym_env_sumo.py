import functools
import itertools
import random
import sys

sys.path.append("./")
import pandas as pd
from shapely import Point
from sklearn.preprocessing import MinMaxScaler

from vanet_env import data_preprocess

sys.path.append("./")

import os
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import poisson

# gym
import gymnasium as gym
from gymnasium import spaces

# custom
from vanet_env import config, network, caching, utility
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
        self,
        render_mode="human",
        multi_core=8,
        caching_fps=10,  # time split to 10 pieces
        fps=10,
        max_step=36000,  # need focus on fps
    ):

        self.num_rsus = config.NUM_RSU
        self.num_vh = config.NUM_VEHICLES / 2
        # self.max_size = config.MAP_SIZE  # deprecated
        self.road_width = config.ROAD_WIDTH
        self.seed = config.SEED
        self.render_mode = render_mode
        self.multi_core = multi_core
        self.max_step = max_step
        self.caching_fps = caching_fps
        self.caching_step = max_step // caching_fps
        print(f"caching_step:{self.caching_step}")

        random.seed(self.seed)

        # important
        self.max_connections = config.MAX_CONNECTIONS
        self.num_cores = config.NUM_CORES
        self.max_content = config.NUM_CONTENT
        self.max_caching = config.RSU_CACHING_CAPACITY
        # rsu range veh wait for connections
        self.max_queue_len = 10
        # every single weight
        self.max_weight = 5
        self.qoe_weight = 10
        # fps means frame per second(frame per sumo timestep)
        self.fps = fps

        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=Point(config.RSU_POSITIONS[i]),
                max_connections=self.max_connections,
                max_cores=self.num_cores,
            )
            for i in range(self.num_rsus)
        ]
        self.rsu_ids = [i for i in range(self.num_rsus)]
        self.avg_u_global = 0
        self.avg_u_local = 0

        # max data rate
        self.max_data_rate = network.max_rate(self.rsus[0])
        print(f"single atn max_data_rate:{self.max_data_rate:.2f}")

        # for convience
        self.rsu_positions = [rsu.position for rsu in self.rsus]
        self.rsu_coords = np.array(
            [
                (self.rsu_positions[rsu.id].x, self.rsu_positions[rsu.id].y)
                for rsu in self.rsus
            ]
        )
        self.rsu_tree = KDTree(self.rsu_coords)

        # network
        self.connections_queue = []
        self.connections = []
        self.rsu_network = network.network(self.rsu_coords, self.rsu_tree)

        # rsu max connection distance
        self.max_distance = network.max_distance_mbps(self.rsus[0])
        print(f"max_distance:{self.max_distance}")
        self.sumo = traci

        # pettingzoo init
        self.agents = ["rsu_" + str(i) for i in range(self.num_rsus)]
        self.possible_agents = self.agents[:]

        self.timestep = 0

        self.content_loaded = False
        self._space_init()

    def _content_init(self):
        self.cache = caching.Caching(
            self.caching_fps, self.max_content, self.max_caching, self.seed
        )
        self.content_list, self.aggregated_df_list, self.aggregated_df = (
            self.cache.get_content_list()
        )
        self.cache.get_content(self.timestep // self.caching_step)

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
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "avg_job_state": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "avg_connection_state": spaces.Box(  # ratio
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "job_info": spaces.Box(  # normalized veh job info (size) in connection range
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus, self.max_connections),
                    dtype=np.float32,
                ),
                "veh_x": spaces.Box(  # vehicle x, normalize to 0-1, e.g. 200 -> 0.5, 400 -> 1
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus, self.max_connections),
                    dtype=np.float32,
                ),
                "veh_y": spaces.Box(  # vehicle y, normalize to 0-1
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus, self.max_connections),
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
            [self.max_weight] * self.num_rsus * 4
            + [self.max_weight] * self.max_connections * self.num_rsus * 3
        )

        # not aggregated
        self.global_ng_state_space = spaces.Dict(
            {
                "compute_resources": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "bandwidth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus,),
                    dtype=np.float32,
                ),
                "job_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus, self.max_connections),
                    dtype=np.float32,
                ),
                "connection_state": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_rsus, self.max_connections),
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
        # every rsu's avg bw, cp, job, conn state and veh pos (x, y) in range
        # self.mixed_md_observation_space = spaces.MultiDiscrete(
        #     [self.max_weight] * self.num_rsus * 2  # cp(usage), job(qoe) status per rsu
        #     + [100] * self.num_rsus  # cp remaining per rsu
        #     + [self.max_weight] * self.num_cores  # self job status (qoe)
        #     + [2] * self.num_cores  # self job arrival 0 means none job arrival
        #     + [2] * self.num_cores  # self job handling 0 means not handling
        #     + [2] * self.max_connections  # self connection queue status 0 means none
        #     + [2] * self.max_connections  # self connected status 0 means none
        #     + [int(self.map_size[0])]
        #     * self.max_connections  # self range veh x 0 means none
        #     + [int(self.map_size[1])]
        #     * self.max_connections  # self range veh y 0 means none
        # )

        job_handling_space = spaces.MultiDiscrete(
            [self.num_rsus + 2] * self.max_connections
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
        # job handling to which rsu id (include self, need 0 to specify self?), num_rsus means cloud, num_rsus + 1 means not handling
        # md_jh_space_shape = spaces.MultiDiscrete([self.num_rsus + 2] * self.max_connections)
        # received handling job handle or not （necessary?）
        #
        # # computing power allocation, value means weight
        # md_ca_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # # bandwidth allocation
        # md_bw_space = spaces.MultiDiscrete([self.max_weight] * self.max_connections)
        # # caching_decision
        # md_cd_space = spaces.MultiDiscrete([self.max_content] * self.max_caching)

        # combined
        md_combined_space = spaces.MultiDiscrete(
            [self.num_rsus + 2]
            * self.max_connections  # job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
            + [2] * self.num_cores  # handling arrival job? 0 for not, 1 for yes
            + [self.max_weight] * self.num_cores  # computing power alloc
            + [self.max_weight] * self.num_cores  # bw alloc
            + [100]  # computing power usage  %
            + [self.max_content] * self.max_caching  # caching policy
        )

        # frame job Schelling
        dict_frame_combined_action_space = spaces.Dict(
            {
                # this job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
                "connection": spaces.Discrete(self.num_rsus + 2),
                "handling_job": spaces.Discrete(
                    2
                ),  # handling arrival job? 0 for not, 1 for yes
                "computing_power_alloc": spaces.Box(
                    low=0.0, high=1.0, shape=(1,)
                ),  # computing power alloc
                "bw_alloc": spaces.Box(low=0, high=1, shape=(1,)),  # bw alloc
                "cp_usage": spaces.Box(low=0, high=1, shape=(1,)),
                "caching": spaces.MultiBinary(self.max_content),  # caching choose
            }
        )

        # frame job Schelling
        self.md_frame_combined_action_space = spaces.MultiDiscrete(
            [
                self.num_rsus + 2
            ]  # this job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
            + [2]  # handling arrival job? 0 for not, 1 for yes
            + [self.max_weight]  # computing power alloc
            + [self.max_weight]  # bw alloc
            + [self.max_weight]  # cp usage
            + [self.max_content]  # caching choose
        )

        dict_combined_action_space = spaces.Dict(
            {
                # job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
                "connection": spaces.MultiDiscrete([self.num_rsus + 2]),
                "handling_jobs": spaces.MultiBinary(
                    self.num_cores
                ),  # handling arrival job? 0 for not, 1 for yes
                "computing_power_alloc": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_cores,)
                ),  # computing power alloc
                "bw_alloc": spaces.Box(
                    low=0, high=1, shape=(self.num_cores,)
                ),  # bw alloc
                "cp_usage": spaces.Box(
                    low=0, high=1, shape=(1,)
                ),  # computing power usage  %
                "caching": spaces.MultiBinary(self.max_content),
            }
        )

        # only this space tested
        # mixed: for every single agent, global aggregated but local not
        # every rsu's avg bw, cp, job, conn state and veh pos (x, y) in range
        self.mixed_dict_observation_space = spaces.Dict(
            {
                "cpu_usage_per_rsu": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_rsus,)
                ),
                "avg_jobs_qoe_per_rsu": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_rsus,)
                ),
                "self_jobs_qoe": spaces.Box(low=0.0, high=1.0, shape=(self.num_cores,)),
                "self_arrival_jobs": spaces.MultiBinary(self.num_cores),
                "self_handling_jobs": spaces.MultiBinary(self.num_cores),
                "self_connection_queue": spaces.MultiBinary(self.max_connections),
                "self_connected": spaces.MultiBinary(self.max_connections),
            }
        )

        self.mixed_frame_md_observation_space = spaces.MultiDiscrete(
            [self.max_weight]  # rsus: avg usage
            + [self.num_cores]  # rsus: avg handling jobs count
            + [self.qoe_weight]  # self: avg_job_qoe
            + [self.num_cores]  # self: num of arrival_jobs
            + [self.num_cores]  # self: num of handling_jobs
            + [self.max_connections]  # self: num of queue_connections
            + [self.max_connections]  # self: num of connected
        )

        self.mixed_frame_dict_observation_space = spaces.Dict(
            {
                "avg_usage": spaces.Box(low=0.0, high=1.0, shape=(1,)),
                "avg_num_handling_jobs": spaces.Discrete(self.num_cores),
                # "avg_jobs_qoe_per_rsu": spaces.Box(
                #     low=0.0, high=1.0, shape=(self.num_rsus,)
                # ),
                "self_avg_jobs_qoe": spaces.Box(low=0.0, high=1.0, shape=(1,)),
                "self_num_arrival_jobs": spaces.Discrete(
                    self.num_cores
                ),  # num of arrival_jobs
                "self_num_handling_jobs": spaces.Discrete(
                    self.num_cores
                ),  # num of handling_jobs
                "self_num_connection_queue": spaces.Discrete(
                    self.max_connections
                ),  # num of queue_connections
                "self_num_connected": spaces.Discrete(
                    self.max_connections
                ),  # num of connected
            }
        )

        self.md_single_action_space = md_combined_space
        self.dict_combined_space = dict_combined_action_space

        print(
            f"action sample{self.md_frame_combined_action_space.sample()}, obs sample{self.mixed_frame_md_observation_space.sample()}"
        )

    def reset(self, seed=config.SEED, options=None):
        if not self.content_loaded:
            self._content_init()
            self.content_loaded = True

        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        self._sumo_init()
        # self._space_init()

        self.vehicle_ids = self.sumo.vehicle.getIDList()

        self.vehicles = [
            Vehicle(
                vehicle_id,
                self.sumo,
                self.timestep,
                self.cache.get_content(self.timestep // self.caching_step),
            )
            for vehicle_id in self.vehicle_ids
        ]
        self.pending_job_vehicles = self.vehicles

        self._update_connections_queue()

        self.agents = list.copy(self.possible_agents)
        self.timestep = 0
        # every rsu's avg bw, cp, job, conn state and veh, job info in this rsu's range
        # self.mixed_md_observation_space = spaces.MultiDiscrete(
        #     [self.max_weight] * self.num_rsus * 4
        #     + [self.max_weight] * self.max_connections * 3
        # )

        observation = {
            "cpu_usage_per_rsu": np.zeros(shape=(self.num_rsus,), dtype=np.float32),
            "avg_jobs_qoe_per_rsu": np.zeros(shape=(self.num_rsus,), dtype=np.float32),
            "self_jobs_qoe": np.zeros(shape=(self.num_cores,), dtype=np.float32),
            "self_arrival_jobs": np.zeros(
                shape=(self.num_cores,), dtype=np.int64
            ),  # MultiDiscrete 默认返回 int64
            "self_handling_jobs": np.zeros(shape=(self.num_cores,), dtype=np.int64),
            "self_connection_queue": np.zeros(
                shape=(self.max_connections,), dtype=np.int64
            ),
            "self_connected": np.zeros(shape=(self.max_connections,), dtype=np.int64),
        }

        # do not take any action except caching at start
        observations = {
            agent: {
                "observation": observation,
                # "action_mask": self._single_dict_action_mask(idx, next_time_step=0),
            }
            for idx, agent in enumerate(self.agents)
        }
        # parallel_to_aec conversion may needed
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):

        # take action
        self._take_frame_actions(actions)

        # caculate rewards
        # dev tag: calculate per timestep? or per fps?
        rewards = self._calculate_rewards()

        # sumo simulation every 10 time steps
        if self.timestep % self.fps == 0:
            self.sumo.simulationStep()
            # update veh status(position and job type) after sim step
            self._update_vehicles()
            # update connections queue, very important
            self._update_connections_queue()
            # remove deprecated rsu connections remove deprecated jobs
            self._update_rsus()

        # update observation space
        observations = self._update_frame_observations()
        # time up or sumo done

        # self.sumo.simulation.getMinExpectedNumber() <= 0
        if self.timestep >= self.max_step:
            terminations = {a: True for a in self.agents}
        else:
            terminations = {a: False for a in self.agents}

        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.timestep += 1

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _update_rsus(self):
        for rsu in self.rsus:
            rsu.update_conn_list()

        for rsu in self.rsus:
            rsu.update_job_handling_list()
        # may not necessary
        # for rsu in self.rsus:
        #     rsu.job_clean()
        ...

    def _update_vehicles(self):
        current_vehicle_ids = set(self.sumo.vehicle.getIDList())
        previous_vehicle_ids = set(self.vehicle_ids)

        # find new veh in map
        new_vehicle_ids = current_vehicle_ids - previous_vehicle_ids
        for vehicle_id in new_vehicle_ids:
            self.vehicles.append(
                Vehicle(
                    vehicle_id,
                    self.sumo,
                    self.timestep,
                    self.cache.get_content(self.timestep // self.caching_step),
                )
            )

        # find leaving veh
        removed_vehicle_ids = previous_vehicle_ids - current_vehicle_ids
        self.vehicles = [
            vehicle
            for vehicle in self.vehicles
            if vehicle.vehicle_id not in removed_vehicle_ids
        ]

        # update vehicle_ids
        self.vehicle_ids = list(current_vehicle_ids)

        # update every veh's position and direction
        for vehicle in self.vehicles:
            vehicle.update_pos_direction()
            # dev tag: update content?
            vehicle.update_job_type(
                self.cache.get_content(self.timestep // self.caching_step)
            )

        # vehs need pending job
        self.pending_job_vehicles = [veh for veh in self.vehicles if not veh.job.done()]

    def _single_frame_discrete_action_mask(self, rsu_idx, next_time_step):
        """
        # frame job Schelling
        self.md_frame_combined_action_space = spaces.MultiDiscrete(
            [
                self.num_rsus + 2
            ]  # this job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
            + [2]  # handling arrival job? 0 for not, 1 for yes
            + [self.max_weight]  # computing power alloc
            + [self.max_weight]  # bw alloc
            + [self.max_weight]  # cp usage
            + [self.max_content]  # caching choose
        )
        """
        frame = next_time_step % self.fps

        if 0 <= frame <= self.max_connections:
            m = []
            for i in range(self.rsus):
                if i == 0:
                    rsu_temp = self.rsus[rsu_idx]
                else:
                    rsu_temp = self.rsu_ids[i] - 1 if i <= rsu_idx else self.rsu_ids[i]
                    rsu_temp = self.rsus[id]

                if (
                    rsu_temp.handling_job_queue.size()
                    == rsu_temp.handling_job_queue.max_size
                ):
                    m.append(0)
                else:
                    m.append(1)
            m.append(1)  # cloud
            m.append(1)  # refuse
        else:
            m = [0] * self.num_rsus + 2

        rsu = self.rsus[rsu_idx]
        if 1 <= frame <= self.max_connections + 1:
            h = []
            conn = rsu.connections_queue[0]
            if conn is not None:
                h = [1] * 2
            else:
                h = [0] * 2
        else:
            h = [0] * 2

        if 2 <= frame <= self.max_connections + 2:
            # cp alloc
            if rsu.handling_jobs[0] is not None:
                ca = [1] * self.max_weight

            else:
                ca = [0] * self.max_weight

            # bw alloc
            if rsu.connections[0] is not None:
                ba = [1] * self.max_weight
            else:
                ba = [0] * self.max_weight

            cp_u = [1] * self.max_weight
        else:
            ca = [0] * self.max_weight
            ba = [0] * self.max_weight
            cp_u = [0] * self.max_weight

        if self.timestep % self.caching_step == 0:
            c = []
        else:
            c = [0] * self.max_content

        act_mask = m + h + ca + ba + cp_u + c
        return act_mask
        pass

    def _single_dict_action_mask(self, rsu_idx, next_time_step):
        """
        dict_combined_action_space = spaces.Dict(
            {
                # job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
                "connection": spaces.Box(low=-1, high=1, shape=(self.num_rsus,)),
                "handling_jobs": spaces.MultiBinary(
                    self.num_cores
                ),  # handling arrival job? 0 for not, 1 for yes
                "computing_power_alloc": spaces.Box(
                    low=0, high=1, shape=(self.num_cores,)
                ),  # computing power alloc
                "bw_alloc": spaces.Box(
                    low=0, high=1, shape=(self.num_cores,)
                ),  # bw alloc
                "cp_usage": spaces.Box(
                    low=0, high=1, shape=(1,)
                ),  # computing power usage  %
                "caching": spaces.Box(low=0, high=1, shape=(self.max_caching,)),
            }
        )
        """
        # per fps need connection policy
        if next_time_step % self.fps == 0:
            connection_action_mask = [
                0 if c is None else 1 for c in self.rsus[rsu_idx].connections_queue
            ]
        else:
            connection_action_mask = [0] * self.max_connections

        # dev tag: Maybe split the frame gap, e.g. sumo's timestep is 1s, our own time step is set to 1/10 i.e. 0.1s means 10fps
        # every frame beginning
        if next_time_step % self.fps == 1:
            job_handle_action_mask = [1] * self.num_cores
        else:
            job_handle_action_mask = [0] * self.num_cores

        # only queued job need
        resource_manager_action_mask = [
            0 if hconn is None else 1 for hconn in self.rsus[rsu_idx].handling_jobs
        ]

        caching_action_mask = [0] * self.max_caching

        # dev tag: would caching_step eq fps?
        if next_time_step % self.caching_step == 0:
            caching_action_mask = [1] * self.max_caching

        # 1 * connction + 2 * resourceman + caching
        return (
            connection_action_mask
            + job_handle_action_mask
            + resource_manager_action_mask * 2
            + caching_action_mask
            + [1]
        )
        ...

    def _single_action_mask(self, rsu_idx, next_time_step):
        """
        space:
        md_combined_space = spaces.MultiDiscrete(
            [self.num_rsus + 2]
            * self.max_connections  # job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
            + [2] * self.num_cores  # handling arrival job? 0 for not, 1 for yes
            + [self.max_weight] * self.num_cores  # computing power alloc
            + [self.max_weight] * self.num_cores  # bw alloc
            + [self.max_content] * self.max_caching  # caching policy
            + [100]  # computing power usage  %
            + [100]  # bw ratio % may not need
        )
        """
        # per fps need connection policy
        if next_time_step % self.fps == 0:
            connection_action_mask = [
                0 if c is None else 1 for c in self.rsus[rsu_idx].connections_queue
            ]
        else:
            connection_action_mask = [0] * self.max_connections

        # dev tag: Maybe split the frame gap, e.g. sumo's timestep is 1s, our own time step is set to 1/10 i.e. 0.1s means 10fps
        # every frame beginning
        if next_time_step % self.fps == 1:
            job_handle_action_mask = [1] * self.num_cores
        else:
            job_handle_action_mask = [0] * self.num_cores

        # only queued job need
        resource_manager_action_mask = [
            0 if hconn is None else 1 for hconn in self.rsus[rsu_idx].handling_jobs
        ]

        caching_action_mask = [0] * self.max_caching

        # dev tag: would caching_step eq fps?
        if next_time_step % self.caching_step == 0:
            caching_action_mask = [1] * self.max_caching

        # 1 * connction + 2 * resourceman + caching
        return (
            connection_action_mask
            + job_handle_action_mask
            + resource_manager_action_mask * 2
            + caching_action_mask
            + [1]
        )

    def _take_frame_actions(self, actions):
        """
        frame job Schelling
        self.md_frame_combined_action_space = spaces.MultiDiscrete(
            [
                self.num_rsus + 2
            ]  # this job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
            + [2]  # handling arrival job? 0 for not, 1 for yes
            + [self.max_weight]  # computing power alloc
            + [self.max_weight]  # bw alloc
            + [self.max_weight]  # cp usage
            + [self.max_content]  # caching choose
        )
        hyper:
        # 在条件判断前预先计算状态码
        state = (h_conn << 2) | (h2_conn << 1) | m_conn

        # 定义处理规则映射表
        action_rules = {
            # (h_conn, h2_conn, m_conn) : slice
            0b000: (0, 1),    # 无连接 -> action[0]
            0b001: (3, None), # 仅m_conn -> action[3:]
            0b010: (1, None), # 仅h2_conn -> action[1:]
            0b011: (4, None), # h2_conn + m_conn -> action[4:]
            0b100: (1, None), # 仅h_conn -> action[1:]
            0b101: (4, None), # h_conn + m_conn -> action[4:]
            0b110: (2, None), # h_conn + h2_conn -> action[2:]
            0b111: (5, None)  # 全连接 -> action[5:]
        }

        # 统一处理逻辑
        start_idx, end_idx = action_rules.get(state, (0, 1))
        a = action[start_idx:end_idx]
        """
        frame = self.timestep % self.fps
        h_conn = False
        h2_conn = False
        m_conn = False

        for idx, action in enumerate(list(actions.values())):
            rsu = self.rsus[idx]

            if 0 <= frame <= self.max_connections:
                # time: 0->max_connections handle connections (mig to where)
                conn = rsu.connections_queue[idx]
                if conn is not None:
                    h_conn = True
                    a_conn = action[0]
                    if a_conn == self.num_rsus + 1:
                        continue  # not handling or continue pre policy
                    elif a_conn == self.num_rsus:
                        # cloud handling
                        # direct connect to cloud
                        # conn.connect(rsu, is_cloud=True)
                        rsu.frame_queuing_job(conn, cloud=True)
                        rsu.connect(conn, jumping=True)
                        ...
                    elif a_conn == 0:
                        rsu.frame_queuing_job(conn)
                    else:
                        # conn.connect(rsu)
                        self.rsus[
                            (
                                self.rsu_ids[a_conn] - 1
                                if a_conn <= idx
                                else self.rsu_ids[a_conn]
                            )
                        ].frame_queuing_job(conn)

            if 1 <= frame <= self.max_connections + 1:

                # arrival job choose handle?
                # do not skip first action
                if h_conn:
                    rsu.frame_handling_job(action[1:2])

                else:
                    rsu.frame_handling_job(action[0])

                h2_conn = True

            if 2 <= frame <= self.max_connections + 2:
                m_conn = True
                #    + [self.max_weight]  # computing power alloc
                #    + [self.max_weight]  # bw alloc
                #    + [self.max_weight]  # cp usage
                if h_conn and h2_conn:
                    a = action[2:5]
                elif h_conn or h2_conn:
                    a = action[1:4]
                else:
                    a = action[0:3]
                cp_a = a[0]
                bw_a = a[1]
                cp_u = a[2]
                rsu.frame_allocate_computing_power(cp_a, cp_u)
                rsu.frame_allocate_bandwidth(bw_a)

            if self.timestep % self.caching_step == 0:
                if h_conn and h2_conn and m_conn:
                    a = action[5:]
                elif (h_conn and m_conn) or (h2_conn and m_conn):
                    a = action[4:]
                elif m_conn:
                    a = action[3:]
                elif h_conn and h2_conn:
                    a = action[2:]
                elif h_conn or h2_conn:
                    a = action[1:]
                else:
                    a = action[0]
                rsu.frame_cache_content(a)
        pass

    def _take_dict_actions(self, actions):
        """
            dict_combined_action_space = spaces.Dict(
            {
                # job connect and to rsu.id or cloud, 0 means self, num_rsu + 1 means not connect
                "connection": spaces.MultiDiscrete(
                    [self.num_rsus + 2] * self.max_connections
                ),
                "handling_jobs": spaces.MultiBinary(
                    self.num_cores
                ),  # handling arrival job? 0 for not, 1 for yes
                "computing_power_alloc": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_cores,)
                ),  # computing power alloc
                "bw_alloc": spaces.Box(
                    low=0, high=1, shape=(self.num_cores,)
                ),  # bw alloc
                "cp_usage": spaces.Box(
                    low=0, high=1, shape=(1,)
                ),  # computing power usage  %
                "caching": spaces.Box(low=0.0, high=1.0, shape=(self.max_caching,)),
            }
        )
        """
        # first, connections/job handling, after first time step(first sim), take actions
        if self.timestep % self.fps == 0:
            for idx, action in enumerate(list(actions.values())):
                rsu = self.rsus[idx]
                # decode action
                connections_handling = action["connection"]

                # excute job_handling
                for conn_idx, jh_action in enumerate(connections_handling):
                    conn = rsu.connections_queue[conn_idx]
                    if conn is None:
                        continue

                    if jh_action == self.num_rsus + 1:
                        continue  # not handling or continue pre policy
                    elif jh_action == self.num_rsus:
                        # cloud handling
                        # direct connect to cloud
                        # conn.connect(rsu, is_cloud=True)
                        rsu.queuing_job(conn, cloud=True)
                        ...
                    else:
                        # conn.connect(rsu)
                        self.rsus[jh_action].queuing_job(conn)
                    # # dev tag: if 0 means self then
                    # elif jh_action == 0:
                    #     # self handling
                    #     conn.connect(rsu)
                    #     rsu.queuing_job(conn)
                    #     ...
                    # else:
                    #     conn.connect(rsu)
                    #
                    #     # if jh_action ≤ idx, ids-1, for exclude idx
                    #     self.rsus[
                    #         (
                    #             self.rsu_ids[jh_action] - 1
                    #             if jh_action <= idx
                    #             else self.rsu_ids[jh_action]
                    #         )
                    #     ].queuing_job(conn)
        else:
            # seconde, resources handling
            for idx, action in enumerate(list(actions.values())):
                rsu = self.rsus[idx]
                # Take job at the beginning of each frame step. dev tag: or every frame step?
                if self.timestep % self.fps == 1:
                    arrival_job_handling_action = action["handling_jobs"]
                    # arrival job handling
                    rsu.handling_job(arrival_job_handling_action)

                computing_power_allocation = action["computing_power_alloc"]
                bandwidth_allocation = action["bw_alloc"]

                cp_usage = action["cp_usage"]
                # bw_ratio = action[
                #     self.num_cores * 3
                #     + self.max_caching
                #     + 1 : self.max_connections
                #     + self.num_cores * 3
                #     + self.max_caching
                #     + 2
                # ]

                # allocate computing power
                total_computing_power = sum(computing_power_allocation)
                if total_computing_power > 0:
                    normalized_computing_power = [
                        cp / total_computing_power for cp in computing_power_allocation
                    ]
                    rsu.allocate_computing_power(normalized_computing_power, cp_usage)

                # allocate bw
                total_bandwidth = sum(bandwidth_allocation)
                if total_bandwidth > 0:
                    normalized_bandwidth = [
                        bw / total_bandwidth for bw in bandwidth_allocation
                    ]
                    rsu.allocate_bandwidth(normalized_bandwidth, 1.0)

        # excuete caching policy
        if self.timestep % self.caching_step == 0:
            caching_decision = action["caching"]
            rsu.cache_content(caching_decision)
        ...

    def _take_actions(self, actions):
        print(actions["rsu_0"])
        print(actions["rsu_1"])
        # action rsu 0 -- n
        # first, connections/job handling, after first time step(first sim), take actions
        if self.timestep % self.fps == 0:
            for idx, action in enumerate(list(actions.values())):
                rsu = self.rsus[idx]
                # decode action
                connections_handling = action[: self.max_connections]

                # excute job_handling
                for conn_idx, jh_action in enumerate(connections_handling):
                    conn = rsu.connections_queue[conn_idx]

                    if jh_action == self.num_rsus + 1:
                        continue  # not handling or continue pre policy
                    elif jh_action == self.num_rsus:
                        # cloud handling
                        # direct connect to cloud
                        # conn.connect(rsu, is_cloud=True)
                        rsu.queuing_job(conn, cloud=True)
                        ...
                    else:
                        # conn.connect(rsu)
                        self.rsus[
                            (
                                self.rsu_ids[jh_action] - 1
                                if jh_action <= idx
                                else self.rsu_ids[jh_action]
                            )
                        ].queuing_job(conn)

        else:
            # seconde, resources handling
            for idx, action in enumerate(list(actions.values())):
                # Take job at the beginning of each frame step. dev tag: or every frame step?
                if self.timestep % self.fps == 1:
                    arrival_job_handling_action = action[: self.num_cores]
                    # arrival job handling
                    self.rsus[idx].handling_job(arrival_job_handling_action)

                computing_power_allocation = action[
                    self.num_cores : self.max_connections + self.num_cores * 2
                ]
                bandwidth_allocation = action[
                    self.num_cores * 2 : self.max_connections + self.num_cores * 3
                ]

                cp_usage = action[
                    self.num_cores * 3
                    + self.max_caching : self.max_connections
                    + self.num_cores * 3
                    + self.max_caching
                    + 1
                ]
                # bw_ratio = action[
                #     self.num_cores * 3
                #     + self.max_caching
                #     + 1 : self.max_connections
                #     + self.num_cores * 3
                #     + self.max_caching
                #     + 2
                # ]

                # allocate computing power
                total_computing_power = sum(computing_power_allocation)
                if total_computing_power > 0:
                    normalized_computing_power = [
                        cp / total_computing_power for cp in computing_power_allocation
                    ]
                    rsu.allocate_computing_power(normalized_computing_power, cp_usage)

                # allocate bw
                total_bandwidth = sum(bandwidth_allocation)
                if total_bandwidth > 0:
                    normalized_bandwidth = [
                        bw / total_bandwidth for bw in bandwidth_allocation
                    ]
                    rsu.allocate_bandwidth(normalized_bandwidth, 1.0)

        # excuete caching policy
        if self.timestep % self.caching_step == 0:
            caching_decision = action[-self.max_caching :]
            rsu.cache_content(caching_decision)
        pass

    def _calculate_rewards(self):
        rewards = {}
        avg_u_global, avg_u_local = utility.calculate_utility_all_optimized(
            self.vehicles,
            self.rsus,
            self.rsu_network,
            self.timestep,
            self.fps,
            self.max_weight,
            self.qoe_weight,
        )

        self.avg_u_global = avg_u_global
        self.avg_u_local = avg_u_local

        for idx, a in enumerate(self.agents):
            rsu = self.rsus[idx]
            u_local = rsu.avg_u
            # dev tag: factor may need specify
            rewards[a] = avg_u_global * 0.7 + u_local * 0.3

        return rewards

    def _update_frame_observations(self):
        """
        self.mixed_frame_md_observation_space = spaces.MultiDiscrete(
            + [10]  # self: avg_job_qoe
            + [self.num_cores]  # self: num of arrival_jobs
            + [self.num_cores]  # self: num of handling_jobs
            + [self.max_connections]  # self: num of queue_connections
            + [self.max_connections]  # self: num of connected
            + [self.max_weight]  # rsus: avg usage
            + [self.num_cores]  # rsus: avg handling jobs count
        )
        """
        observations = {}

        usage_all = 0
        count = 0

        for rsu in self.rsus:
            usage_all += rsu.cp_usage
            count += 1

        avg_usage = 0 if count == 0 else usage_all // count

        for idx, a in enumerate(self.agents):
            rsu = self.rsus[idx]
            obs = []
            obs.append(int(self.avg_u_local), avg_usage)

            qoe_all = 0
            count = 0
            for hconn in rsu.handling_jobs:
                if hconn is not None:
                    qoe_all += hconn.qoe
                    count += 1

            avg_qoe = 0 if count == 0 else qoe_all / count
            arrival_jobs = rsu.handling_job_queue.size()
            handling = count
            num_conn_queue = rsu.connections_queue.size()
            num_connected = rsu.connections.size()

            obs.append(avg_qoe, arrival_jobs, handling, num_conn_queue, num_connected)
            act_mask = self._single_frame_discrete_action_mask(idx, self.timestep + 1)
            observations[a] = {"observation": obs, "action_mask": act_mask}

        return observations
        pass

    def _update_dict_observations(self):
        """
        only this space tested
        mixed: for every single agent, global aggregated but local not
        every rsu's avg bw, cp, job, conn state and veh pos (x, y) in range
        self.mixed_dict_observation_space = spaces.Dict(
            {
                "cpu_usage_per_rsu": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_rsus,)
                ),
                "avg_jobs_qoe_per_rsu": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_rsus,)
                ),
                "self_jobs_qoe": spaces.Box(low=0.0, high=1.0, shape=(self.num_cores,)),
                "self_arrival_jobs": spaces.MultiDiscrete([2] * self.num_cores),
                "self_handling_jobs": spaces.MultiDiscrete([2] * self.num_cores),
                "self_connection_queue": spaces.MultiDiscrete([2] * self.connections),
                "self_connected": spaces.MultiDiscrete([2] * self.connections),
            }
        )
        """
        observations = {}

        usage_list = []
        qoe_list = []
        for rsu in self.rsus:
            usage_list.append(rsu.cp_usage)
            qoe = 0
            qoe_count = 0

            for hconn in rsu.handling_jobs:
                if hconn is not None:
                    qoe += int(hconn.qoe)
                    qoe_count += 1

            avg_qoe = 0.0 if qoe_count == 0 else qoe / qoe_count
            qoe_list.append(avg_qoe)

        for idx, a in enumerate(self.agents):
            rsu = self.rsus[idx]
            self_qoe_list = []

            for hconn in rsu.handling_jobs:
                if hconn is None:
                    self_qoe_list.append(0.0)
                else:
                    self_qoe_list.append(hconn.qoe)

            """
            + [2] * self.num_cores  # self job arrival 0 means none job arrival
            + [2] * self.num_cores  # self job handling 0 means not handling
            + [2] * self.max_connections  # self connection queue status 0 means none
            + [2] * self.max_connections  # self connected status 0 means none
            """
            arrival = [1 if x is not None else 0 for x in rsu.handling_job_queue]
            handling = [1 if x is not None else 0 for x in rsu.handling_jobs]

            connections_queue = [
                1 if x is not None else 0 for x in rsu.connections_queue
            ]
            connected = [1 if x is not None else 0 for x in rsu.connections]

            obs = {
                "cpu_usage_per_rsu": usage_list,
                "avg_jobs_qoe_per_rsu": qoe_list,
                "self_jobs_qoe": self_qoe_list,
                "self_arrival_jobs": arrival,
                "self_handling_jobs": handling,
                "self_connection_queue": connections_queue,
                "self_connected": connected,
            }

            action_mask = self._single_dict_action_mask(
                idx, next_time_step=self.timestep + 1
            )

            observations[a] = {"observation": obs}

        return observations

    def _update_observations(self):
        observations = {}
        """
        # space:
        # mixed: for every single agent, global aggregated but local not
        # every rsu's avg bw, cp, job, conn state and veh pos (x, y) in range
        self.mixed_md_observation_space = spaces.MultiDiscrete(
            [self.max_weight] * self.num_rsus * 2  # cp(usage), job(qoe) status per rsu
            + [100] * self.num_rsus  # cp remaining per rsu
            + [self.max_weight] * self.num_cores  # self job status (avg qoe)
            + [2] * self.num_cores  # self job arrival 0 means none job arrival
            + [2] * self.num_cores  # self job handling 0 means not handling
            + [2] * self.max_connections  # self connection queue status 0 means none
            + [2] * self.max_connections  # self connected queue status 0 means none
            + [int(self.map_size[0])]
            * self.max_connections  # self range veh x 0 means none
            + [int(self.map_size[1])]
            * self.max_connections  # self range veh y 0 means none
        )
        """
        usage_list = []
        qoe_list = []
        for rsu in self.rsus:
            usage_list.append(rsu.cp_usage)
            qoe = 0
            qoe_count = 0

            for hconn in rsu.handling_jobs:
                if hconn is not None:
                    qoe += int(hconn.qoe * self.max_weight)
                    qoe_count += 1

            avg_qoe = 0 if qoe_count == 0 else qoe // qoe_count
            qoe_list.append(avg_qoe)

        for idx, a in enumerate(self.agents):
            rsu = self.rsus[idx]
            self_qoe_list = []

            for hconn in rsu.handling_jobs:
                if hconn is None:
                    self_qoe_list.append(0)
                else:
                    self_qoe_list.append(hconn.qoe * self.max_weight)

            """
            + [2] * self.num_cores  # self job arrival 0 means none job arrival
            + [2] * self.num_cores  # self job handling 0 means not handling
            + [2] * self.max_connections  # self connection queue status 0 means none
            + [2] * self.max_connections  # self connected status 0 means none
            """
            arrival = [1 if x is not None else 0 for x in rsu.handling_job_queue]
            handling = [1 if x is not None else 0 for x in rsu.handling_jobs]

            connections_queue = [
                1 if x is not None else 0 for x in rsu.connections_queue
            ]
            connected = [1 if x is not None else 0 for x in rsu.connections]
            """
            + [int(self.map_size[0])]
            * self.max_connections  # self range veh x 0 means none
            + [int(self.map_size[1])]
            * self.max_connections  # self range veh y 0 means none
            """
            x_list = []
            y_list = []

            for conn in rsu.connections_queue:
                if conn is None:
                    x_list.append(0)
                    y_list.append(0)
                else:
                    x_list.append(int(conn.veh.position.x))
                    y_list.append(int(conn.veh.position.y))
            obs = (
                usage_list
                + qoe_list
                + self_qoe_list
                + arrival
                + handling
                + connections_queue
                + connected
                + x_list
                + y_list
            )

            action_mask = self._single_action_mask(
                idx, next_time_step=self.timestep + 1
            )

            observations[a] = {"observation": obs, "action_mask": action_mask}
        pass

    def _manage_resources(self):
        # connections info update
        # bw policy here
        # for conn in self.connections_queue:
        #     rsu = conn.rsu
        #     veh = conn.veh
        #     data_rate = (
        #         network.channel_capacity(rsu, veh)
        #         * rsu.num_atn
        #         / len(rsu.connections_queue.olist)
        #     )

        #     conn.data_rate = data_rate
        pass

    # peformance issue
    def _update_connections_queue(self, kdtree=True):
        """
        connection logic
        """
        # clear connections
        self.connections_queue = []
        # rsu not clear, just update
        # for rsu in self.rsus:
        #     rsu.connections = []

        if kdtree == False:
            # not implement!
            for veh in self.pending_job_vehicles:
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
                        rsu.connections_queue.append(conn)
                        self.connections_queue.append(conn)
        else:
            # connections update
            for veh in self.pending_job_vehicles:
                # if veh.job.done():
                #     continue

                vehicle_x, vehicle_y = veh.position.x, veh.position.y
                vehicle_coord = np.array([vehicle_x, vehicle_y])

                # find vehicle_coord range has rsu?
                indices = self.rsu_tree.query_ball_point(
                    vehicle_coord, self.max_distance
                )

                # remove deprecated veh.connections which veh.connections[i].rsu.id not in indices
                for i in range(veh.connections.max_size):
                    if (
                        veh.connections[i] is not None
                        and veh.connections[i].rsu.id not in indices
                    ):
                        veh.connections[i].disconnect()
                        veh.connections[i] = None

                # is there need choose one to prefer connect?
                for idx in indices:
                    rsu = self.rsus[idx]
                    conn = Connection(rsu, veh)

                    # update rsu conn queue, only one here
                    matching_connection = [
                        c for c in rsu.connections_queue if c == conn
                    ]

                    # if rsu--veh exist, update veh info else append
                    if matching_connection:
                        matching_connection[0].veh = veh
                        conn = matching_connection[0]
                    else:
                        rsu.connections_queue.append(conn)

                    # update veh conn queue, only one here
                    # need update to new logic, i.e. conn replace rsu?
                    veh_matching_connection = [c for c in veh.connections if c == conn]
                    # if veh--rsu exist, update veh info else append
                    if veh_matching_connection:
                        veh_matching_connection[0].veh = veh
                        conn = veh_matching_connection[0]
                    else:
                        veh.connections.append(conn)

                    self.connections_queue.append(conn)

                # conn remove logic #1
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

            # rsu conn remove logic #2 i.e. connection loss
            for rsu in self.rsus:
                for idx, c in enumerate(rsu.connections_queue):
                    if c is None:
                        continue

                    if c not in self.connections_queue:
                        rsu.connections_queue[idx].disconnect()
                        rsu.connections_queue.remove(idx)

            # job disconnected logic

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

        # render QoE
        # not correct now
        for conn in self.connections_queue:
            color = interpolate_color(0, 1, conn.qoe)

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
        # only render after sim
        if self.timestep % self.fps == 0:
            mode = self.render_mode if mode is None else mode
            # human
            if mode is not None:
                # get veh ID

                # clear all dynamic rendered polygon
                # draw QoE
                for polygon_id in self.sumo.polygon.getIDList():
                    if polygon_id.startswith("dynamic_"):
                        self.sumo.polygon.remove(polygon_id)

                self._render_connections()

                return

        return

    def close(self):
        self.sumo.close()
        return super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.mixed_frame_md_observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.mixed_frame_dict_observation_space
