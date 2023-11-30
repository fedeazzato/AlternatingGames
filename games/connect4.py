from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from pettingzoo.classic import connect_four_v3 as connect4
from base.game import AlternatingGame, AgentID, ActionType
import numpy as np
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

class Connect4(AlternatingGame):
    def __init__(self, render_mode=''):
        super().__init__()
        self.env = connect4.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.render_mode = render_mode
        self.played_actions = []

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def observe(self, agent: AgentID) -> ObsType:
        state = self.env.env.game.get_state(self.env._name_to_int(agent))
        obs = state['observation']
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.played_actions = []
        self.env.reset(seed, options)
        self._update()

    def render(self) -> ndarray | str | list | None:
        return self.env.render()

    def step(self, action: ActionType) -> None:
        self.played_actions.append(action)
        self.env.step(action)
        self._update()

    def available_actions(self):
        return list(self.env._legal_moves())

    def close(self):
        self.env.close()

    def clone(self):
        # Create a new game (initial state is always the same)
        cloned_game = Connect4()
        cloned_game.reset()

        # replay all actions
        for action in self.played_actions:
            cloned_game.step(action)

        # return the replayed game        
        return cloned_game