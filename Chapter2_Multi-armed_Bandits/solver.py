import numpy as np


class Solver:
    def __init__(self, n_states, k_arms, step_size):
        self.n_states = n_states
        self.k_arms = k_arms
        self.step_size = step_size
        self.step_count = 0

    def __call__(self, state):
        # self.step_countのインクリメントを実装すること
        raise NotImplementedError

    def update(self, state, action, reward):
        raise NotImplementedError

    # step_sizeが-1だと, (1/ステップ数)を学習率αに用いる
    def _get_alpha(self, step_size):
        if step_size == -1:
            alpha = 1 / (self.step_count + 1)
        else:
            alpha = step_size
        return alpha


class ValueIteration(Solver):
    def __init__(self, n_states, k_arms, step_size, initial_value):
        super().__init__(n_states, k_arms, step_size)
        self.pred_q_table = initial_value * np.ones((n_states, k_arms))

    def update(self, state, action, reward):
        alpha = self._get_alpha(self.step_size)
        self.pred_q_table[state][action] += alpha * (reward - self.pred_q_table[state][action])


class EpsilonGreedy(ValueIteration):
    def __init__(self, n_states, k_arms, step_size, initial_value, epsilon):
        super().__init__(n_states, k_arms, step_size, initial_value)
        self.epsilon = epsilon

    def __call__(self, state):
        self.step_count += 1
        is_greedy = np.random.rand() >= self.epsilon
        if is_greedy:
            # 行動価値が等しい行動が複数あるときは, その中からランダムに選ぶ
            pred_qs = self.pred_q_table[state]
            action_candidates = np.arange(self.k_arms)[pred_qs == np.max(pred_qs)]
        else:
            # 全ての行動からランダムに選ぶ
            action_candidates = np.arange(self.k_arms)
        action = np.random.choice(action_candidates)
        return action


class UCB1(ValueIteration):
    def __init__(self, n_states, k_arms, step_size, initial_value, conf_coeff):
        super().__init__(n_states, k_arms, step_size, initial_value)
        self.conf_coeff = conf_coeff
        self.action_counts = np.zeros((n_states, k_arms)) + 1e-6

    def __call__(self, state):
        self.step_count += 1
        confidence = np.sqrt(np.log(self.step_count) / self.action_counts[state])
        pred_qs = self.pred_q_table[state] + self.conf_coeff * confidence
        action_candidates = np.arange(self.k_arms)[pred_qs == np.max(pred_qs)]
        action = np.random.choice(action_candidates)
        self.action_counts[state][action] += 1
        return action


class PolicyGradient(Solver):
    def __init__(self, n_states, k_arms, step_size, baseline_step_size, with_baseline):
        super().__init__(n_states, k_arms, step_size)
        self.preferences = np.zeros((n_states, k_arms))
        self.baseline = 0
        self.baseline_step_size = baseline_step_size
        self.with_baseline = with_baseline

    def __call__(self, state):
        self.step_count += 1
        preference = self.preferences[state]
        probability = self._softmax(preference)
        action = np.random.choice(np.arange(self.k_arms), p=probability)
        return action

    def update(self, state, action, reward):
        baseline_alpha = self._get_alpha(self.baseline_step_size)
        self.baseline += baseline_alpha * (reward - self.baseline)

        preference = self.preferences[state]
        probability = self._softmax(preference)
        one_hot = self._indicator(action)
        alpha = self._get_alpha(self.step_size)
        self.preferences[state] += alpha * (reward - self.with_baseline*self.baseline) * (one_hot - probability)

    def _softmax(self, preference):
        return np.exp(preference) / np.sum(np.exp(preference))

    def _indicator(self, action):
        one_hot = np.zeros(self.k_arms)
        one_hot[action] = 1
        return one_hot
