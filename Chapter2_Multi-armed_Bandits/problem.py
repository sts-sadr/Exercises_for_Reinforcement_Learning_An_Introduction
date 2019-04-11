import numpy as np


class BanditProblem:
    def __init__(self, k_arms, reward_std):
        self.k_arms = k_arms
        self.n_states = 1
        self._true_qs = None
        self._reward_std = reward_std

    def try_an_arm(self, k):
        if self._true_qs is None:
            raise NotImplementedError
        true_q = self._true_qs[k]
        reward = np.random.normal(true_q, self._reward_std)
        # ContextualBanditProblemとの整合性のためstate(always 0)も返す
        return reward, 0

    def _get_optimal_action(self):
        if self._true_qs is None:
            raise NotImplementedError
        return np.argmax(self._true_qs)


class StationaryBanditProblem(BanditProblem):
    def __init__(self, k_arms, reward_std=1.0, problem_std=1.0):
        super().__init__(k_arms, reward_std)
        self._true_qs = np.random.normal(0, problem_std, k_arms)


class NonStationaryBanditProblem(BanditProblem):
    def __init__(self, k_arms, reward_std=1.0, change_std=0.01):
        super().__init__(k_arms, reward_std)
        self._true_qs = np.zeros(k_arms)
        self._change_std = change_std

    def try_an_arm(self, k):
        true_q = self._true_qs[k]
        reward = np.random.normal(true_q, self._reward_std)
        self._true_qs += np.random.normal(0, self._change_std, self.k_arms)
        return reward, 0


class ContextualBanditProblem:
    def __init__(self, k_arms, reward_std=1.0, problem_std=1.0,
                 n_states=2, state_info=True):
        self.k_arms = k_arms
        self._reward_std = reward_std
        self.problems = [StationaryBanditProblem(k_arms, reward_std, problem_std)
                         for _ in range(n_states)]
        self.n_states = n_states
        self.state_info = state_info
        self._state = 0 # 初期状態は0

    def try_an_arm(self, k):
        problem = self.problems[self._state]
        reward, _ = problem.try_an_arm(k)
        next_state = np.random.randint(self.n_states)
        self._state = next_state
        if state_info:
            return reward, next_state
        else:
            return reward, 0

    def _get_optimal_action(self, state):
        return self.problems[state]._get_optimal_action()
