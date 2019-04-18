import numpy as np


# [Stationary|NonStationary]BanditProblemのための抽象クラス
class BanditProblem:
    def __init__(self, k_arms, reward_std):
        self.k_arms = k_arms
        self.n_states = 1
        self._true_qs = None  # 真の価値
        self._reward_std = reward_std
        # ContextualBanditProblemとの整合性のためself._stateも持っておく（常に0）
        self._state = 0

    def try_an_arm(self, k):
        if self._true_qs is None:
            raise NotImplementedError
        true_q = self._true_qs[k]
        # 報酬は真の価値を平均とし, reward_stdを分散とした正規分布からサンプルされる
        reward = np.random.normal(true_q, self._reward_std)
        return reward, self._state

    def _get_optimal_action(self):
        if self._true_qs is None:
            raise NotImplementedError
        return np.argmax(self._true_qs)


# 定常なバンディット問題
# 初めに平均0, 標準偏差problem_stdで生成した価値をずっと使う
class StationaryBanditProblem(BanditProblem):
    def __init__(self, k_arms, reward_std, problem_mean, problem_std):
        super().__init__(k_arms, reward_std)
        self._true_qs = np.random.normal(problem_mean, problem_std, k_arms)


# 非定常なバンディット問題
# 初めに価値を0で初期化し, 以後毎ステップ平均0, 標準偏差change_stdが加算されて変化する
class NonStationaryBanditProblem(BanditProblem):
    def __init__(self, k_arms, reward_std, change_std):
        super().__init__(k_arms, reward_std)
        self._true_qs = np.zeros(k_arms)
        self._change_std = change_std

    def try_an_arm(self, k):
        true_q = self._true_qs[k]
        reward = np.random.normal(true_q, self._reward_std)
        self._true_qs += np.random.normal(0, self._change_std, self.k_arms)
        return reward, self._state


# 文脈付きバンディット問題
# 内部にn_states個のStationaryBanditProblemを持ち, ランダムにその中の1つが状態として選ばれる
# state_infoによって状態の情報を返すかを選べる. Falseなら常に0が返る
class ContextualBanditProblem:
    def __init__(self, k_arms, reward_std, problem_mean, problem_std, n_states, state_info):
        self.k_arms = k_arms
        self._reward_std = reward_std
        self._problems = [StationaryBanditProblem(k_arms, reward_std, problem_mean, problem_std)
                          for _ in range(n_states)]
        self.n_states = n_states
        self.state_info = state_info
        self._state = 0  # 初期状態は0

    def try_an_arm(self, k):
        problem = self._problems[self._state]
        reward, _ = problem.try_an_arm(k)
        next_state = np.random.randint(self.n_states)
        self._state = next_state
        if state_info:
            return reward, next_state
        else:
            return reward, 0

    def _get_optimal_action(self, state):
        return self._problems[state]._get_optimal_action()
