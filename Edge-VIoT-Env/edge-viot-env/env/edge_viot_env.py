# doc: https://pettingzoo.farama.org/content/environment_creation/
from pettingzoo import AECEnv
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

# # ignore
# def raw_env(**kwargs):
#     env = EdgeVIoTEnv(render_mode=kwargs.get("render_mode"))
#     env = parallel_to_aec(env)
#     return env

class EdgeVIoTEnv(AECEnv):
    metadata = {
        "name": "edge_viot_env_v0",
    }
    
    def __init__(self, *args, **kwargs):
        # Init
        self.num_agents = 4
        self.num_jobs = 5
        self.max_cache = 5
        self.max_content = 10
        self.render_mode = "ansi"
        # self.act_dims = [1, 1]
        # self.n_act_agents = self.act_dims[0]

        num_jobs = self.num_jobs
        num_agents = self.num_agents
        max_cache = self.max_cache
        max_content = self.max_content
        
        # agents
        # num_agents = kwargs.get("num_agents")
        self.agents = ["rsu_" + str(i) for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # self.possible_agents = ["test"]
        
        # Spaces
        # Define action space for each RSU
        # Change in any stage
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([num_agents + 1] * num_jobs),  # 1. Target RSU or core network, 0 for not migrate
            spaces.Box(low=0, high=1, shape=(num_jobs,), dtype=np.float32),  # 2. Migration ratio
            spaces.Discrete([max_content] * max_cache),  # 3. Cache content selection (select 5 different content to cache), this maybe duplicate selection, need to be handled
            spaces.MultiDiscrete([num_agents + 1])  # 4. Target RSU or core network for rejected job, 0 for accept
        ))
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'jobs': spaces.Box(low=0, high=1, shape=(num_agents, num_jobs), dtype=np.float32),  # Job status for all RSUs
            'compute_capacity': spaces.Box(low=0, high=1, shape=(num_agents,), dtype=np.float32),  # Compute capacity for all RSUs
            'bandwidth': spaces.Box(low=0, high=1, shape=(num_agents, num_jobs), dtype=np.float32),  # Bandwidth between RSUs and users
            'cache': spaces.MultiDiscrete([max_content] * max_cache * num_agents),  # Cache status for all RSUs
            'user_trajectory': spaces.Box(low=0, high=1, shape=(num_jobs, 2), dtype=np.float32)  # User trajectory data after processing  
        })
        
        self.action_spaces = dict(zip(self.agents, self.action_space))
        self.observation_spaces = dict(zip(self.agents, self.observation_space))
        self.steps = 0
        self.closed = False
        pass

    def reset(self, seed=None, options=None):
        """
        Init agents attr
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        
        # Reset the state of the environment to an initial state
        observations = {
            agent: {
            'jobs': np.zeros((self.num_rsus, self.num_jobs), dtype=np.float32),
            'compute_capacity': np.zeros((self.num_rsus,), dtype=np.float32),
            'bandwidth': np.zeros((self.num_rsus, self.num_users), dtype=np.float32),
            'cache': np.zeros((self.num_rsus, self.max_cache), dtype=np.float32),
            'user_trajectory': np.zeros((self.num_jobs, 2), dtype=np.float32)
        } for agent in self.agents}
        
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
        # Env Simulate
        
        # Capacity
        # 1.Storage capacity changes
        
        # 2.Bandwidth capacity changes
        
        # 3.Computing capacity changes
        
        
        # QoE 
        # 1.Transmission Latency
        # 2.Computational Latency
        # 3.Weighted Sum
        
        pass

    def render(self):
        pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]