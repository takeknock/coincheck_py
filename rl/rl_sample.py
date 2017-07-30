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

def montecarlo_policy_iteration(policy_iterations, episodes, options):

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


    for i in range(policy_iterations):
        visits = np.ones((n_states, n_actions))
        results = np.zeros((episodes, 1))
        #np.rand()

        # for each episode
        for episode in range(episodes):
            state3 = np.zeros((n_states, n_actions))
            for step in range(steps):
                state = encode(state3)
                policy = np.zeros(1, n_actions)
                if (options['mode'] == 1):
                    
        pass


def main():
    ################# arguments #################
    # L
    policy_iterations = 10
    # M
    episodes = 10

    options = {'gamma': 0.9, 'tau': 0.3, 'epsilon' : 0.2, 'mode' : 1}
    action = montecarlo_policy_iteration(policy_iterations, episodes, options)
    print(action)

class test_encode(unittest.TestCase):
    def test_encode_3_to_10(self):
        test_input = np.array([1, 0, 0, 0, 0, 0, 0, 0, 2])
        expected = 6563
        self.assertAlmostEqual(expected, encode(test_input), delta=10e-12)

if __name__ == '__main__':
    main()
    unittest.main()

