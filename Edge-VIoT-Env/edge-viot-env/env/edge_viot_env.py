# doc: https://pettingzoo.farama.org/content/environment_creation/
import functools
from pettingzoo import AECEnv, ParallelEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector, wrappers, parallel_to_aec
import numpy as np

# ignore
# def env(**kwargs):
#     render_mode = kwargs.get("render_mode")
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


def raw_env(render_mode="ansi"):
    env = EdgeVIoTEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


# def get_aec_env() -> AECEnv:
#     parallel_env = MyParallelEnv()
#     aec_env = parallel_to_aec(parallel_env)
#     return aec_env


class EdgeVIoTEnv(ParallelEnv):
    metadata = {
        "name": "edge_viot_env_v0",
    }

    def __init__(
        self,
        num_rsu=4,
        num_jobs=5,
        max_content=100,
        max_cache=10,
        current_time=0,
        render_mode="ansi",
    ) -> None:
        # Init
        # self.num_rsu = 4
        # self.num_jobs = 5
        # self.max_cache = 5
        # max cache content
        # self.max_content = 100

        self.num_rsu = num_rsu
        self.num_jobs = num_jobs
        self.max_content = max_content
        self.max_cache = max_cache
        self.current_time = current_time
        self.time_step = 0

        self.render_mode = render_mode
        # self.act_dims = [1, 1]
        # self.n_act_agents = self.act_dims[0]

        # agents
        # num_rsu = kwargs.get("num_rsu")
        self.agents = ["rsu_" + str(i) for i in range(self.num_rsu)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_rsu))))
        self._agent_selector = agent_selector(self.agents)

        # self.possible_agents = ["test"]

        # Spaces
        # Define action space for each RSU
        # Change in any stage
        # self.action_space = spaces.Tuple((
        #     spaces.MultiDiscrete([num_rsu + 1] * num_jobs),  # 1. Target RSU or core network, 0 for not migrate
        #     spaces.Box(low=0, high=1, shape=(num_jobs,), dtype=np.float32),  # 2. Migration ratio
        #     spaces.Discrete([max_content] * max_cache),  # 3. Cache content selection (select 5 different content to cache), this maybe duplicate selection, need to be handled
        #     spaces.MultiDiscrete([num_rsu + 1])  # 4. Target RSU or core network for rejected job, 0 for accept
        # ))

        # T = 0, 5, 10, ..., choose caching content
        # self.action_spaces = {
        #     agent: spaces.Discrete([max_content] * max_cache) if self.current_time % 5 == 0 else spaces.Tuple((
        #         spaces.MultiDiscrete([num_rsu + 1] * num_jobs),  # 1. Target RSU or core network, 0 for not migrate
        #         spaces.Box(low=0, high=1, shape=(num_jobs,), dtype=np.float32),  # 2. Migration ratio
        #         spaces.MultiDiscrete([num_rsu + 1])  # 3. Target RSU or core network for rejected job, 0 for accept
        #     )) for agent in range(num_rsu)
        # }

        # No Caching
        self.action_spaces = {
            agent: spaces.Tuple(
                (
                    spaces.Discrete(
                        self.max_content * self.max_cache
                    ),  # 1. Cache content selection (select 5 different content to cache), this maybe duplicate selection, need to be handled
                    spaces.MultiDiscrete(
                        [num_rsu + 1] * num_jobs
                    ),  # 2. Target RSU or core network, 0 for not migrate
                    spaces.Box(
                        low=0, high=1, shape=(num_jobs,), dtype=np.float32
                    ),  # 3. Migration ratio
                    spaces.MultiDiscrete(
                        [num_rsu + 1]
                    ),  # 4. Target RSU or core network for rejected job, 0 for accept
                )
            )
            for agent in range(num_rsu)
        }

        # Define observation space
        observation_space = spaces.Dict(
            {
                "jobs": spaces.Box(
                    low=0, high=1, shape=(num_rsu, num_jobs), dtype=np.float32
                ),  # Job status for all RSUs
                "compute_capacity": spaces.Box(
                    low=0, high=1, shape=(num_rsu,), dtype=np.float32
                ),  # Compute capacity for all RSUs
                "bandwidth": spaces.Box(
                    low=0, high=1, shape=(num_rsu,), dtype=np.float32
                ),  # Bandwidth status for all RSUs
                "cache": spaces.Box(
                    low=0, high=1, shape=(num_rsu, max_cache), dtype=np.float32
                ),  # Cache status for all RSUs, convert to discrete by * 100, eg 0.1 * 100 = 10, means cache id = 10
                "user_trajectory": spaces.Box(
                    low=0, high=1, shape=(num_jobs, 2), dtype=np.float32
                ),  # User trajectory data after processing
            }
        )

        # self.action_spaces = dict(zip(self.agents, self.action_space))
        self.observation_spaces = {agent: observation_space for agent in self.agents}

        self.steps = 0
        self.closed = False
        pass

    def reset(self, seed=None, options=None):
        """
        Init agents attr
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.time_step = 0

        # Reset the state of the environment to an initial state
        # Need Modify
        observation_space = spaces.Dict(
            {
                "jobs": spaces.Box(
                    low=0, high=1, shape=(self.num_rsu, self.num_jobs), dtype=np.float32
                ),  # Job status for all RSUs
                "compute_capacity": spaces.Box(
                    low=0, high=1, shape=(self.num_rsu,), dtype=np.float32
                ),  # Compute capacity for all RSUs
                "bandwidth": spaces.Box(
                    low=0, high=1, shape=(self.num_rsu,), dtype=np.float32
                ),  # Bandwidth status for all RSUs
                "cache": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.num_rsu, self.max_cache),
                    dtype=np.float32,
                ),  # Cache status for all RSUs, convert to discrete by * 100, eg 0.1 * 100 = 10, means cache id = 10
                "user_trajectory": spaces.Box(
                    low=0, high=1, shape=(self.num_jobs, 2), dtype=np.float32
                ),  # User trajectory data after processing
            }
        )

        observations = {agent: observation_space for agent in self.agents}

        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # actions = {agent: action for agent, action in actions.items()}
        # Env Simulate

        # T = 0, 5, 10, ... choose caching content
        if self.current_time % 5 == 0:
            pass
        else:
            # T = 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, ...

            pass
        # Capacity
        # 1.Storage capacity changes

        # 2.Bandwidth capacity changes

        # 3.Computing capacity changes

        # QoE
        # 1.Transmission Latency
        # 2.Computational Latency
        # 3.Weighted Sum
        self.current_time += 1

        # observation sample
        observations = {
            agent: {
                "jobs": np.random.random((self.num_rsu, self.num_jobs)),
                "compute_capacity": np.random.random(self.num_rsu),
                "bandwidth": np.random.random(self.num_rsu),
                "cache": np.random.random((self.num_rsu, self.max_cache)),
                "user_trajectory": np.random.random((self.num_jobs, 2)),
            }
            for agent in self.agents
        }

        # terminations = {agent: np.random.choice([True, False], p=[0.5, 0.5]) for agent in self.agents}

        # Sample reward
        rewards = {agent: np.random.random() for agent in self.agents}

        # Sample truncations
        # truncations = {agent: np.random.choice([True, False], p=[0.5, 0.5]) for agent in self.agents}

        # Sample infos
        infos = {agent: {} for agent in self.agents}
        self.time_step += 1
        # test, delete later
        print("Time Step: ", self.time_step)

        # Sample, modify later
        if self.time_step == 100:
            terminations = {agent: True for agent in self.agents}
            self.agents = []
        else:
            terminations = {agent: False for agent in self.agents}

        return (
            observations,
            rewards,
            terminations,
            {agent: False for agent in self.agents},
            infos,
        )

    def render(self):
        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # current_time: 0 for caching, 1 for migration,
    # here we only consider caching,
    # future use of hierarchical reinforcement learning separation cache
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Tuple(
            (
                spaces.Discrete(
                    self.max_content * self.max_cache
                ),  # 1. Cache content selection (select 5 different content to cache), this maybe duplicate selection, need to be handled
                spaces.MultiDiscrete(
                    [self.num_rsu + 1] * self.num_jobs
                ),  # 2. Target RSU or core network, 0 for not migrate
                spaces.Box(
                    low=0, high=1, shape=(self.num_jobs,), dtype=np.float32
                ),  # 3. Migration ratio
                spaces.MultiDiscrete(
                    [self.num_rsu + 1]
                ),  # 4. Target RSU or core network for rejected job, 0 for accept
            )
        )
