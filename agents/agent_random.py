from base.game import AlternatingGame, AgentID
from base.agent import Agent
import numpy as np

class RandomAgent(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, seed=None) -> None:
        super().__init__(game=game, agent=agent)

    def action(self):
        return self.game.action_space(self.agent).sample()
    
    def policy(self):
        return np.full(self.game.available_actions(), 1/self.game.available_actions())