import numpy as np
import pandas as pd
import unittest
from numpy.random import *

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
            self.discounted_rewards = self.init_matrix() # 各episodeの、各stepごとの割引報酬？？
            self.results = self.init_results() # ゲームの結果報酬 results.len <=> episodes

            for episode_index in range(self.episodes):
                state3 = self.init_state3()

                for step_index in range(self.steps):
                    state = encode(state3)
                    policy = self.generate_policy()
                    # ここまでの政策反復で得たQ関数を用いて政策改善
                    # 政策改善はあるQ関数の下では同じQ関数(評価関数)を
                    # 用いて、政策を出力し続ける
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
            # 行動価値関数(評価関数)を生成(政策評価)
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
        if self.options["pmode"]==0:
            q = self.Q[state]
            v = max(q)
            a = np.where(q==v)[0][0]
            policy[a] = 1
        elif self.options["pmode"]==1:
            q = self.Q[state]
            v = max(q)
            a = np.where(v==q)[0][0]
            policy = np.ones(self.n_actions) * self.options["epsilon"] / self.n_actions
            policy[a] = 1 - self.options["epsilon"] + self.options["epsilon"] / self.n_actions
        elif self.options["pmode"] == 2:
            policy = np.exp(self.Q[state] / self.options["tau"]) / sum(np.exp(self.Q[state] / self.options["tau"]))

        return policy

    # play 1 game
    def action_train(self, policy, step_index, state3):
        # assumption : training player move first

        # まずはplayerの行動選択
        npc_action = self.select_npc_action(policy, state3)
        # 2は学習プレイヤーの行動した印
        state3[npc_action] = 2
        checked_result = self.judge(state3)
        reward = self.calculate_reward(checked_result)

        if reward is not None:
            return npc_action, reward, state3, checked_result

        # 次にenemyの行動選択
        enemy_action = self.select_enemy_action(step_index, state3)
        state3[enemy_action] = 1
        checked_result = self.judge(state3)
        reward = self.calculate_reward(checked_result)

        return npc_action, reward, state3, checked_result

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

    # 印をつけるセルを返す
    def select_npc_action(self, step_index, policy, state3):
        a = None
        if step_index == 0:
            a = 0
        else:
            while 1:
                random = np.random.rand()
                cumulative_probability = 0
                for a in range(self.n_actions):
                    cumulative_probability += policy[a]
                    if cumulative_probability > random :
                        break

                # 既に0埋めされてないか確認
                # 0以外の場合、既に行動済み
                # 行動済みの場合、行動を重ねられないため
                # 別の行動が選ばれるよう、もう一回行動ガチャを回す
                if state3[a] == 0:
                    break
        # 第1step なら 0のマスに印をつける
        return a

    def judge(self, state3):
        fin_positions = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
        # fin_positionに含まれるマスの組のうち、全てが1もしくは2のものが、state3に含まれていた場合、
        # ゲーム終了
        #for position in fin_positions:
        #    for i in position:
        for i in range(len(fin_positions)):
            state_i = state3[fin_positions[i]]
            val_npc = sum(state_i == 1)
            val_enemy = sum(state_i == 2)
            if val_npc == 3:
                # win player
                return 1
            if val_enemy == 3:
                # win enemy
                return 2
        is_actioned_all_cell = sum(state3 == 0) == 0
        if (is_actioned_all_cell):
            # tie
            return 3
        # game is continued
        return 0

    def calculate_reward(self, finished_state):
        # assumption: finished_state is contained in {0, 1, 2, 3} space
        if finished_state == 1:
            return 10
        if finished_state == 2:
            return -10
        if finished_state == 3:
            return None
        if finished_state == 0:
            return 0

    def select_enemy_action(self, step_index, state3):
        # あと1つでenemy winとなるならそのセルを埋めに行く
        fin_positions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6]]
        for i in range(len(fin_positions)):
            # ゲーム終了条件に関係するセルの状態のみ抽出
            state_i = state3[fin_positions[i]]
            print("state_i : ", state_i)
            # val \in [0, 6]
            val = sum(state_i)
            # どちらも印をつけてないセルの数
            num_of_no_action_cells = len(state_i[state_i == 0])
            if val == 2 and num_of_no_action_cells == 1:
                # [1, 1, 0]のようなstate_iの場合のみ、この分岐
                # idx : 0となっている(まだ印のついていないインデックス
                idx = state_i.index(0)
                a = fin_positions[i][idx]
                # この場合これ以上良い行動はないので、
                # 決めたセルを行動として返す
                return a

        # 上のような1手詰みの状況でなければ、
        # ランダムに手を選ぶ
        while 1:
            a = np.random.randint(10)
            if state3[a] == 0:
                # 選んだセルにまだ印がついていなければ
                # そのセルに印をつけることに決定。
                # 選んだセルを行動として返す
                return a


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

