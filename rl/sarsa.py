import numpy as np
from matplotlib import pyplot as plt

class SARSAPolicyIteration:
    # game information
    n_states = 3**9
    n_actions = 9 #行動数(最大9箇所)
    steps = 5

    Q = None
    visits = None
    states = None
    actions = None
    rewards = None
    discounted_rewards = None
    results = None
    options = None
    rates = []
    # 盤面を回転させると同じ状態になる
    # その変換を行い状態数を減らす
    convert = [[0,1,2,3,4,5,6,7,8], # 元の状態
               [2,1,0,5,4,3,8,7,6], # 変換(2)
               [6,3,0,7,4,1,8,5,2], # 変換(3)
               [0,3,8,1,4,7,2,5,8], # 変換(4)
               [8,7,6,5,4,3,2,1,0], # 変換(5)
               [6,7,8,3,4,5,0,1,2], # 変換(6)
               [2,5,8,1,4,7,0,3,6], # 変換(7)
               [8,5,2,7,4,1,6,3,0]  # 変換(8)
               ]

    power = np.array([3**i for i in range(8, -1, -1)], dtype=np.float64)

    def __init__(self, policy_iterations, episodes, options):
        self.policy_iterations = policy_iterations
        self.episodes = episodes
        self.Q = self.init_q()
        self.options = options
        np.random.seed(555)

    def init_q(self):
        return np.zeros((self.n_states, self.n_actions))


    def train(self):
        for policy_index in range(self.policy_iterations):
            self.visits = self.init_visits()
            self.states = self.init_matrix()
            self.actions = self.init_matrix()
            self.rewards = self.init_matrix()
            new_q = self.init_q()
            self.discounted_rewards = self.init_matrix()
            self.results = self.init_results() # ゲームの結果報酬 results.len <=> episodes

            for episode_index in range(self.episodes):
                state3 = self.init_state3()
                p_state = None
                p_action = None
                for step_index in range(self.steps):
                    state = self.encode(state3)

                    policy = self.generate_policy()
                    # ここまでの政策反復で得たQ関数を用いて政策改善
                    # 政策改善はあるQ関数の下では同じQ関数(評価関数)を
                    # 用いて、政策を出力し続ける
                    policy = self.improve_policy(policy, state)

                    # row 44, in p.56
                    action, reward, state3, fin = self.action_train(policy, step_index, state3)

                    ###違うのココから
                    if step_index > 0:
                        # 1ステップ前の状態、行動のQ値を更新
                        q = new_q[p_state]
                        v = max(q)
                        new_q[p_state][p_action] += \
                            self.options["alpha"] * (reward - new_q[p_state][p_action])  \
                            + self.options["gamma"] * v

                    if self.is_finished(fin):
                        self.results[episode_index] = fin
                        break

                    p_state = state
                    p_action = action
                    ### ココまで

            ##各エピソードで生成したQ関数で更新
            self.Q = new_q
            self.rates.append(self.calculate_win_ratio())
            self.output_results(policy_index)
            #print("Q state value function : ", self.Q)

    # play 1 turn
    def action_train(self, policy, step_index, state3):
        # assumption : training player move first

        # まずはplayerの行動選択
        npc_action = self.select_npc_action(step_index, policy, state3)
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


    def init_visits(self):
        return np.ones((self.n_states,self.n_actions))

    def init_matrix(self):
        return np.ones((self.episodes, self.steps))

    def init_return(self):
        return np.zeros(self.episodes)

    def init_results(self):
        return np.zeros(self.episodes)

    def init_state3(self):
        return np.zeros(self.n_actions)

    def generate_policy(self):
        return np.zeros(self.n_actions)

    def output_results(self, l):
        print('l=%d: Win=%d/%d, Draw=%d/%d, Lose=%d/%d\n' % (l, \
                len(self.results[self.results == 2]), self.episodes, \
                len(self.results[self.results == 3]), self.episodes, \
                len(self.results[self.results == 1]), self.episodes))

    def encode(self, state3):
        # stateに(2)～(8)の8種類の変換を加えた後、10進数へ変換
        cands = [ sum(state3[self.convert[i]]*self.power) # indexを入れ替えて、10進数に変換
                    for i in range(len(self.convert))]
        # 8個の候補のうち一番小さいものを選ぶ
        return min(cands)+1

    def is_finished(self, fin):
        return fin > 0

    def calculate_win_ratio(self):
        return float(len(self.results[self.results == 2])) / float(self.episodes)

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
            val_npc = sum(state_i == 2)
            val_enemy = sum(state_i == 1)
            if val_npc == 3:
                # win player
                return 2
            if val_enemy == 3:
                # win enemy
                return 1
        is_actioned_all_cell = sum(state3 == 0) == 0
        if (is_actioned_all_cell):
            # tie
            return 3
        # game is continued
        return 0

    def calculate_reward(self, finished_state):
        # assumption: finished_state is contained in {0, 1, 2, 3} space
        if finished_state == 2:
            return 10
        if finished_state == 1:
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



def main():
    ################# arguments #################
    # L
    policy_iterations = 100
    # M
    episodes = 100

    options = {"alpha": 0.5, "gamma": 0.9, "tau": 2, "epsilon": 0.05, "pmode": 1}

    #action = montecarlo_policy_iteration(policy_iterations, episodes, options)
    #print(action)
    sarsapi = SARSAPolicyIteration(policy_iterations, episodes, options)
    sarsapi.train()
    plt.plot(range(len(sarsapi.rates)), sarsapi.rates)
    plt.show()

if __name__ == '__main__':
    main()