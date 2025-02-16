import collections
import functools
import itertools
import math
import random
import sys

from vanet_env import utility_light
from vanet_env import env_config

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
from gym import spaces as gym_spaces

# custom
from vanet_env import network, caching, handler
from vanet_env.utils import (
    RSU_MARKER,
    VEHICLE_MARKER,
    interpolate_color,
    sumo_detector,
    is_empty,
    discrete_to_multi_discrete,
    multi_discrete_space_to_discrete_space,
)
from vanet_env.entites import Connection, Rsu, CustomVehicle, Vehicle

# sumo
import traci
import libsumo
from sumolib import checkBinary

# next performance improvement
import multiprocessing as mp

import datetime


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
        seed=env_config.SEED,
        is_discrete=True,
        handler=handler.TrajectoryHandler,
        map="seattle",
    ):

        self.is_discrete = is_discrete
        self.num_rsus = env_config.NUM_RSU
        # self.max_size = config.MAP_SIZE  # deprecated
        self.road_width = env_config.ROAD_WIDTH
        self.seed = seed
        self.render_mode = render_mode
        self.multi_core = multi_core
        self.max_step = max_step
        self.caching_fps = caching_fps
        self.caching_step = max_step // caching_fps
        self.caching_ratio_dict = []
        self.ava_rewards = []
        self.map = map

        random.seed(self.seed)

        # important rsu info
        self.max_connections = env_config.MAX_CONNECTIONS
        self.num_cores = env_config.NUM_CORES
        self.max_content = env_config.NUM_CONTENT
        self.max_caching = env_config.RSU_CACHING_CAPACITY
        # rsu range veh wait for connections
        self.max_queue_len = 10
        # every single weight
        self.max_weight = 10
        self.qoe_weight = 10
        self.bins = 5
        # fps means frame per second(frame per sumo timestep)
        self.fps = fps

        # init rsus
        self.rsus = [
            Rsu(
                id=i,
                position=Point(env_config.RSU_POSITIONS[i]),
                max_connections=self.max_connections,
                max_cores=self.num_cores,
            )
            for i in range(self.num_rsus)
        ]
        self.rsu_ids = [i for i in range(self.num_rsus)]
        self._agent_ids = [f"rsu_{i}" for i in range(self.num_rsus)]
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
        self.max_distance = network.max_distance_mbps(
            self.rsus[0], rate_tr=env_config.DATA_RATE_TR * 2
        )
        print(f"max_distance:{self.max_distance}")
        self.sumo = traci

        # pettingzoo init
        list_agents = list(self._agent_ids)
        self.agents = list.copy(list_agents)
        self.possible_agents = list.copy(list_agents)

        self.timestep = 0
        # 异常监测
        # 监测无法编排次数
        self.full_count = 0
        # 监测云次数
        self.cloud_times = 0

        self.content_loaded = False
        self.handler_class = handler

        self._handler_init()
        self._space_init()
        self._sumo_init()

    def _handler_init(self):
        self.handler = self.handler_class(self)

    def _content_init(self):
        self.cache = caching.Caching(
            self.caching_fps, self.max_content, self.max_caching, self.seed
        )
        self.content_list, self.aggregated_df_list, self.aggregated_df = (
            self.cache.get_content_list()
        )
        self.cache.get_content(min(self.timestep // self.caching_step, 9))

    def _sumo_init(self):
        # SUMO detector
        # sumo_detector()
        print("sumo init")

        self.cfg_file_path = os.path.join(
            os.path.dirname(__file__), "assets", self.map, "sumo", "osm.sumocfg"
        )

        # gui_settings_path = os.path.join(
        #     os.path.dirname(__file__), "assets", self.map, "sumo", "gui_hide_all.xml"
        # )

        self.icon_path = os.path.join(os.path.dirname(__file__), "assets", "rsu.png")

        # start sumo sim and plot
        if self.render_mode is not None:
            # 初始化数据
            self.steps = []
            self.utilities = []
            self.caching_ratios = []

            # 创建图表
            plt.ion()  # 开启交互模式
            plt.rcParams["toolbar"] = "None"

            self.fig, (self.ut_ax, self.ca_ax) = plt.subplots(
                2, 1, figsize=(8, 6)
            )  # 创建两个子图，上下排列
            (self.ut_line,) = self.ut_ax.plot(
                self.steps, self.utilities, "r-", label="Utility"
            )  # 创建初始的线
            self.ut_ax.set_xlabel("Step")
            self.ut_ax.set_ylabel("Averge Utility")
            self.ut_ax.set_title("Utility over Steps")
            self.ut_ax.legend()

            # 初始化第二个图表（caching ratio vs step）
            (self.ca_line,) = self.ca_ax.plot(
                self.steps, self.caching_ratios, "b-", label="Hit Ratio"
            )
            self.ca_ax.set_xlabel("Step")
            self.ca_ax.set_ylabel("Averge Caching Hit Ratio")
            self.ca_ax.set_title("Caching Ratio over Steps")
            self.ca_ax.legend()

            # 调整子图间距
            plt.tight_layout()
            # 设置窗口位置和大小
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, "window"):
                fig_manager.window.attributes("-topmost", 1)  # 置顶窗口
                # fig_manager.window.geometry(
                #     "200x500+1300+300"
                # )  # 宽度x高度+水平偏移+垂直偏移

            self.sumo.start(
                ["sumo-gui", "-c", self.cfg_file_path, "--step-length", "1", "--start"]
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

            # paint resource capicity
            # for rsu in self.rsus:
            #     max_ee = env_config.MAX_EE
            #     offset = 2
            #     width = 2
            #     height = 1
            #     x1 = rsu.position.x + offset
            #     y1 = rsu.position.y
            #     x2 = rsu.position.x + offset
            #     y2 = rsu.position.y + height
            #     self.sumo.polygon.add(
            #         f"resource_rsu{rsu.id}",
            #         [(x1, y1), (x2, y2)],
            #         color=(205, 254, 194, 255),
            #         fill=False,
            #         lineWidth=2,
            #         layer=40,
            #     )

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
            libsumo.start(["sumo", "-c", self.cfg_file_path])
            # traci.start([sumoBinary, "-c", cfg_file_path])

        net_boundary = self.sumo.simulation.getNetBoundary()
        self.map_size = net_boundary[1]
        self.sumo_has_init = True

    def _space_init(self):
        self.handler.spaces_init()

    def reset(self, seed=env_config.SEED, options=None):
        self.start_time = datetime.datetime.now()
        return self.handler.reset(seed)

    def step(self, actions):

        return self.handler.step(actions)

    # improve：返回polygons然后后面统一绘制？
    def _render_connections(self):
        # Batch add polygons
        polygons_to_add = []

        """
            for rsu in self.rsus:
                max_ee = env_config.MAX_EE
                offset = 2
                width = 2
                height = 1
                x1 = rsu.position.x + offset
                y1 = rsu.position.y
                x2 = rsu.position.x + offset
                y2 = rsu.position.y + height
                self.sumo.polygon.add(
                    f"resource_rsu{rsu.id}",
                    [(x1, y1), (x2, y2)],
                    color=(205, 254, 194, 255),
                    fill=False,
                    lineWidth=2,
                    layer=40,
                )
        """

        # render ee
        for rsu in self.rsus:
            rsu: Rsu
            max_ee = env_config.MAX_EE
            offset = 10
            width = 6
            height = 10

            # jobs_x1 = rsu.position.x - offset
            # jobs_y1 = rsu.position.y
            # jobs_x2 = rsu.position.x - offset

            # norm_self_handling_ratio = (
            #     sum([v[1] for v in rsu.handling_jobs.olist if v is not None])
            #     / rsu.handling_jobs.max_size
            # )
            # jobs_y2 = rsu.position.y + height * max(norm_self_handling_ratio, 0.1)

            x1 = rsu.position.x + offset
            y1 = rsu.position.y
            x2 = rsu.position.x + offset
            y2 = rsu.position.y + height * max(rsu.cp_usage, 0.1)

            if not rsu.idle:
                color = interpolate_color(
                    0, 1, rsu.cp_usage, is_reverse=True
                )  # 越高越红，越低越绿
                # job_color = interpolate_color(
                #     0, 1, norm_self_handling_ratio, is_reverse=True
                # )  # 越高越红，越低越绿
            else:
                y2 = rsu.position.y + height * 0.1
                color = interpolate_color(
                    0, 1, 0.1, is_reverse=True
                )  # 越高越红，越低越绿

                # job_color = interpolate_color(
                #     0, 1, 0.1, is_reverse=True
                # )  # 越高越红，越低越绿
                # jobs_y2 = rsu.position.y + height * 0.1

            color_with_alpha = (*color, 255)
            # job_color_with_alpha = (*job_color, 255)

            polygons_to_add.append(
                (
                    f"dynamic_resource_rsu{rsu.id}",
                    [
                        (
                            x1,
                            y1,
                        ),
                        (x2, y2),
                    ],
                    color_with_alpha,
                    False,
                    width,
                )
            )

            # polygons_to_add.append(
            #     (
            #         f"dynamic_job_rsu{rsu.id}",
            #         [
            #             (
            #                 jobs_x1,
            #                 jobs_y1,
            #             ),
            #             (jobs_x2, jobs_y2),
            #         ],
            #         job_color_with_alpha,
            #         False,
            #         width,
            #     )
            # )

        # render QoE
        # not correct now
        for rsu in self.rsus:
            rsu: Rsu
            for veh in rsu.connections:
                if veh is None:
                    continue

                veh: Vehicle

                if False:
                    pass
                    # 需要拉上所有鄰居嗎

                    # if rsu.connections.size() == 0:
                    #     # 不可能会到这里
                    #     max_trans_qoe = 1e-6
                    # else:
                    #     max_trans_qoe = (
                    #         env_config.MAX_QOE
                    #         * self.max_data_rate
                    #         * rsu.num_atn
                    #         / env_config.JOB_DR_REQUIRE
                    #     ) / rsu.connections.size()

                    # if rsu.handling_jobs.size() == 0:
                    #     # 不知道为什么有可能会到这里
                    #     max_proc_qoe = 1e-6
                    # else:

                    #     max_proc_qoe = (
                    #         env_config.MAX_QOE
                    #         * rsu.computation_power
                    #         * 1
                    #         / env_config.JOB_CP_REQUIRE
                    #     ) / rsu.handling_jobs.size()

                    # max_qoe = min(max_proc_qoe, max_trans_qoe)

                max_qoe = max(env_config.MAX_QOE * 0.7, veh.job.qoe)

                color = interpolate_color(0, max_qoe * 0.7, veh.job.qoe)
                color_with_alpha = (*color, 255)

                polygons_to_add.append(
                    (
                        f"dynamic_line_rsu{rsu.id}_to_{veh.vehicle_id}",
                        [
                            (
                                rsu.position.x,
                                rsu.position.y,
                            ),
                            (veh.position.x, veh.position.y),
                        ],
                        color_with_alpha,
                        False,
                        0.3,
                    )
                )

        for polygon_id, points, color, is_fill, line_width in polygons_to_add:
            self.sumo.polygon.add(
                polygon_id,
                points,
                color=color,
                fill=is_fill,
                lineWidth=line_width,
                layer=41,
            )

    def render(self, mode=None):
        # only render after sim
        if self.timestep % self.fps == 0:
            mode = self.render_mode if mode is None else mode
            # human
            if mode is not None:
                # get veh ID

                # draw utility
                # 模拟utility的计算
                mean_ut = np.nanmean(self.ava_rewards)
                cas = []

                for rsu in self.rsus:
                    cas = np.nanmean(rsu.hit_ratios)

                # self.ava_rewards = []

                if not np.isnan(mean_ut):
                    self.utilities.append(mean_ut)

                # 更新数据
                if self.utilities:
                    self.steps.append(self.timestep)

                    self.caching_ratios.append(np.mean(cas))

                    # 更新图表
                    self.ut_line.set_xdata(self.steps)
                    self.ut_line.set_ydata(self.utilities)
                    self.ut_ax.relim()  # 重新计算坐标轴范围
                    self.ut_ax.autoscale_view()  # 自动调整坐标轴范围

                    self.ca_line.set_xdata(self.steps)
                    self.ca_line.set_ydata(self.caching_ratios)
                    self.ca_ax.relim()  # 重新计算坐标轴范围
                    self.ca_ax.autoscale_view()  # 自动调整坐标轴范围
                    plt.draw()
                    self.fig.canvas.flush_events()

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
        self.sumo_has_init = False
        plt.ioff()
        plt.close()
        self.endtime = datetime.datetime.now()

        return super().close()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.local_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def global_observation_space(self, agent):
        return self.global_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def local_observation_space(self, agent):
        return self.local_neighbor_obs_space

    @functools.lru_cache(maxsize=None)
    def multi_discrete_action_space(self, agent):
        return self.md_discrete_action_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.box_neighbor_action_space

    # def get_agent_ids(self):
    #     return self._agent_ids
