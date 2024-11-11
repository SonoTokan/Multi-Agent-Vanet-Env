from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete

class PoiEdgeEnv(ParallelEnv):
    metadata = {
        "name": "edge_viot_env_v0",
    }

    def __init__(self):
        self.possible_agents = ["prisoner"]
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]