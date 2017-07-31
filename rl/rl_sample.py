import numpy as np
import pandas as pd
import unittest

def encode(state3):
    """
    convert_matrix = np.array([[1,2,3,4,5,6,7,8,9],
                      [3,2,1,6,5,4,9,8,7],
                      [7,4,1,8,5,2,9,6,3],
                      [1,4,7,2,5,8,3,6,9],
                      [9,8,7,6,5,4,3,2,1],
                      [7,8,9,4,5,6,1,2,3],
                      [3,6,9,2,5,8,1,4,7],
                      [9,6,3,8,5,2,7,4,1]])


    results = np.zeros(convert_matrix.share[0])
    for i in range(convert_matrix.shape[1]):
        results[i] =
    """
    power = np.array([3**8, 3**7, 3**6, 3**5, 3**4, 3**3, 3**2, 3**1, 3**0])
    result = np.dot(state3, power)
    return result
"""
def montecarlo_policy_iteration(policy_iterations, episodes, options):
"""
"""Documentation for main function
:settings:
- n_states : all states as integer
- n_actions : in each turn, player is able to choose a cell which he checks
- step : all 3 * 3 cells are checked in 5 turn
- q : value function table

:argument
- policy_iterations : L
- episodes : M

:return:
policy

"""
"""
# states table
# each cell has 3 states.
n_states = 3 ** 9
# in each turn, player is able to choose a cell which he checks
n_actions = 9
# all 3 * 3 cells are checked in 5 turn
steps = 5
# initialize value function table
q  = np.zeros((n_states, n_actions))
print(q)


for l in range(policy_iterations):
    visits = np.ones((n_states, n_actions))
    results = np.zeros((episodes, 1))
    #np.rand()

    # for each episode
    for episode in range(episodes):
        state3 = np.zeros((1, n_actions))
        for step in range(steps):
            state = encode(state3)
            policy = np.zeros(1, n_actions)
            if (options['mode'] == 0):
                q = Q[state]

    pass
"""

class MonteCarloPolicyIteration:
    n_states = 3**9
    n_actions = 9 #行動数(最大9箇所)
    T = 5 # 1 episode(ゲーム)あたりのターン数(敵味方合計)

    visits = None # ？？
    states = None
    actions = None
    rewards = None
    discounted_rewards = None
    results = None
    rates = []

    convert_matrix = np.array([[1,2,3,4,5,6,7,8,9],
                      [3,2,1,6,5,4,9,8,7],
                      [7,4,1,8,5,2,9,6,3],
                      [1,4,7,2,5,8,3,6,9],
                      [9,8,7,6,5,4,3,2,1],
                      [7,8,9,4,5,6,1,2,3],
                      [3,6,9,2,5,8,1,4,7],
                      [9,6,3,8,5,2,7,4,1]])

    power = np.array([3**i for i in range(8, -1, -1)], dtype=np.float64)

    def __init__(self, policy_iterations, episodes, options):
        self.policy_iterations = policy_iterations
        self.episodes = episodes
        self.Q = self.init_Q()
        self.options = options
        np.random.seed(100)

    def init_Q(self):
        return np.zeros((self.n_states, self.n_actions)) # row 7, p56

    def train(self):
        for policy_index in range(self.policy_iterations):
            self.visits = self.init_visits()
            self.states = self.init_matrix() # 盤面？
            self.actions = self.init_matrix() # 各episodeの、各stepごとの行動
            self.rewards = self.init_matrix() # 各episodeの、各stepごとの報酬
            self.drewards = self.init_matrix() # 各episodeの、各stepごとの割引報酬？？
            self.results = self.init_results()

            for episode_index in range(self.episodes):
                state3 = self.init_state3()

                for step_index in range(self.steps):
                    state = encode(state3)
                    policy = self.generate_policy()
                    policy = self.improve_policy(policy, state)

                    # row 44, in p.56
                    action, reward, state3, fin = self.action_train(policy, step_index, state3)

                    self.update(episode_index, step_index, state, action, reward)

                    if self.is_finished(fin):
                        #今ゲームの結果を格納
                        self.results[episode_index] = fin

                        # 今ゲームの割引報酬和(各episodeの各stepごとに格納)
                        self.discounted_rewards = self.calculate_discounted_rewards(episode_index, step_index)
                        break

            self.Q = self.calculate_state_action_value_function()
            self.rates.append(self.calculate_win_ratio())


    def init_visits(self):
        return np.ones(self.n_states,self.n_actions)

    def init_matrix(self):
        return np.ones(self.episodes, self.steps)

    def init_return(self):
        return np.zeros(self.episodes)

    def init_state3(self):
        return np.zeros(self.n_actions)

    def generate_policy(self):
        return np.zeros(self.n_actions)

    def improve_policy(self, policy, state):
        # if self.options["pmode"]==0:
        return 0

    def action_train(self, policy, step_index, state3):
        return 0,0,0,0

    def update(self, episode_index, step_index, state, action, reward):
        pass

    def is_finished(self, fin):
        return True

    def calculate_discounted_rewards(self, episode_index, step_index):
        return 0

    def calculate_state_action_value_function(self):
        return 0

    def calculate_win_ratio(self):
        return 0.5



def main():
    ################# arguments #################
    # L
    policy_iterations = 10
    # M
    episodes = 10

    options = {'gamma': 0.9, 'tau': 0.3, 'epsilon' : 0.2, 'pmode' : 1}

    #action = montecarlo_policy_iteration(policy_iterations, episodes, options)
    #print(action)
    mcpi = MonteCarloPolicyIteration(policy_iterations, episodes, options)
    mcpi.train()

class test_encode(unittest.TestCase):
    def test_encode_3_to_10(self):
        test_input = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2])
        expected = 6563
        self.assertAlmostEqual(expected, encode(test_input), delta=10e-12)

if __name__ == '__main__':
    main()
    unittest.main()

