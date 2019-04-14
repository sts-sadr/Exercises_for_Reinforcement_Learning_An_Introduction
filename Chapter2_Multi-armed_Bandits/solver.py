import numpy as np


class EpsilonGreedy:
    # step_sizeはデフォルトで-1で, この状態だと(1/step数)を学習率に用いる
    def __init__(self, n_states, k_arms, initial_value, epsilon, step_size):
        self.n_states = n_states
        self.k_arms = k_arms
        self.pred_q_table = initial_value * np.ones((n_states, k_arms))
        self.epsilon = epsilon
        self.step_size = step_size
        self.step_count = 0

    def __call__(self, state):
        is_greedy = np.random.rand() >= self.epsilon
        if is_greedy:
            # 行動価値が等しい行動が複数あるときは, その中からランダムに選ぶ
            pred_qs = self.pred_q_table[state]
            action_candidates = np.arange(self.k_arms)[pred_qs == np.max(pred_qs)]
        else:
            # 全ての行動からランダムに選ぶ
            action_candidates = np.arange(self.k_arms)
        action = np.random.choice(action_candidates)
        self.step_count += 1
        return action

    def update(self, state, action, reward):
        if self.step_size == -1:
            alpha = 1 / (1 + self.step_count)
        else:
            alpha = self.step_size

        self.pred_q_table[state][action] += alpha * (reward - self.pred_q_table[state][action])


class UCB1:
    def __init__(self, n_states, k_arms, initial_value, conf_coeff, step_size):
        self.n_states = n_states
        self.k_arms = k_arms
        self.pred_q_table = initial_value * np.ones((n_states, k_arms))
        self.conf_coeff = conf_coeff
        self.action_counts = np.zeros((n_states, k_arms)) + 1e-6
        self.step_size = step_size
        self.step_count = 0

    def __call__(self, state):
        confidence = np.sqrt(np.log(self.step_count) / self.action_counts[state])
        pred_qs = self.pred_q_table[state] + self.conf_coeff * confidence
        action_candidates = np.arange(self.k_arms)[pred_qs == np.max(pred_qs)]
        action = np.random.choice(action_candidates)
        self.action_counts[state][action] += 1
        self.step_count += 1
        return action

    def update(self, state, action, reward):
        if self.step_size == -1:
            alpha = 1 / (1 + self.step_count)
        else:
            alpha = self.step_size

        self.pred_q_table[state][action] += alpha * (reward - pred_q_table[state][action])


class PolicyGradient:
    def __init__(self, n_states, k_arms, initial_value):
        raise NotImplementedError

    def __call__(self, state):
        raise NotImplementedError

    def update(self, state, action, reward):
        raise NotImplementedError



