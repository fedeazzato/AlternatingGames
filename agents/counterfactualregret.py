import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        self.game = game
        self.agent = agent
        self.obs = obs
        self.num_actions = self.game.num_actions(agent)
        self.cumulative_regrets = np.zeros(self.num_actions)
        self.curr_policy = self.uniform_policy()
        self.sum_policy = self.uniform_policy()
        self.learned_policy = self.uniform_policy()
        self.agent_num = self.game.agent_name_mapping[self.agent]

    def uniform_policy(self):
        return np.ones(self.num_actions) / self.num_actions

    def normalize_policy(self, non_normalized_policy):
        return non_normalized_policy / np.sum(non_normalized_policy)

    def update(self, utility, node_utility: int, probability) -> None:
        # Calcular P_p
        new_prob = probability.copy()
        new_prob[self.agent_num] = 1
        P_p = np.prod(new_prob)

        # Calcular regrets
        regrets = utility - node_utility

        # Actualizar regrets acumulados
        self.cumulative_regrets += (P_p * regrets)

        # Caluclar Pp
        Pp = probability[self.agent_num]
        self.sum_policy += Pp * self.curr_policy
        self.learned_policy = self.normalize_policy(self.sum_policy)

        # Regret matching
        positive_regrets = np.maximum(self.cumulative_regrets, 0)
        if np.sum(positive_regrets) == 0:
            self.curr_policy = self.uniform_policy()
        else:
            self.curr_policy = self.normalize_policy(positive_regrets)

    def policy(self) -> ndarray[float]:
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            raise ValueError('Train agent before calling action()')
    
    def train(self, niter=1000):
        for _ in range(niter):
            self.cfr()

    def cfr(self):
        game = self.game.clone()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            self.cfr_rec(game=game, agent=agent, probability=probability)

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
    
        node_agent = game.agent_selection

        # base cases

        if game.done():
            return game.reward(agent)
        
        # recursive call

        # get observation node
        obs = game.observe(node_agent)
        try:
            node = self.node_dict[obs]
        except:
            node = Node(game=game, agent=node_agent, obs=obs)
            self.node_dict[obs] = node

        assert(node_agent == node.agent)

        # compute expected (node) utility
        utility = np.zeros(game.num_actions(node_agent))
        agent_num = game.agent_name_mapping[node_agent]
        for a in game.action_iter(node_agent):
            # update probability vector
            new_prob = probability.copy()
            new_prob[agent_num] *= node.curr_policy[a]  # Probabilidad de ejecutar 'a'

            # play the game
            new_game = game.clone()
            new_game.step(a)

            # call cfr recursively on updated game with new probability and update node utility
            utility[a] = self.cfr_rec(new_game, agent, new_prob)

        node_utility = np.sum(utility * node.curr_policy)

        # update node cumulative regrets using regret matching
        if node_agent == agent:
            node.update(utility=utility, node_utility=node_utility, probability=probability)

        return node_utility
        
