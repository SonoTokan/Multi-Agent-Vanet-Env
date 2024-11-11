# doc: https://pettingzoo.farama.org/content/environment_creation/
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector, wrappers, parallel_to_aec

def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    """
    render_mode = kwargs.get("render_mode")
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(**kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = EdgeVIoTEnv(render_mode=kwargs.get("render_mode"))
    env = parallel_to_aec(env)
    return env

class EdgeVIoTEnv(ParallelEnv):
    metadata = {
        "name": "edge_viot_env_v0",
    }
    
    def __init__(self, *args, **kwargs):
        # init
        self.num_agents = 5
        self.render_mode = "ansi"
        self.act_dims = [1, 1]
        self.action_space = ...
        self.observation_space = ...
        
        # agents
        # num_agents = kwargs.get("num_agents")
        self.agents = ["rsu" + str(a) for a in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        # self.possible_agents = ["test"]
        
        # spaces
        self.n_act_agents = self.act_dims[0]
        self.action_spaces = dict(zip(self.agents, self.action_space))
        self.observation_spaces = dict(zip(self.agents, self.observation_space))
        self.steps = 0
        self.closed = False
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        # self.num_moves = 0
        # None need trans to real init value
        observations = {agent: None for agent in self.agents}
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