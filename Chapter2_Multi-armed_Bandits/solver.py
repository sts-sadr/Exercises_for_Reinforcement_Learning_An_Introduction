import numpy as np


class EpsilonGreedy:
    # step_sizeはデフォルトで-1で, この状態だと(1/step数)を学習率に用いる
    def __init__(self, n_states, k_arms, initial_value=0.0, epsilon=0.1, step_size=-1):
        self.n_states = n_states
        self.k_arms = k_arms
        self.pred_q_table = initial_value * np.ones((n_states, k_arms))
        self.epsilon = epsilon
        self.step_size = step_size
        self.step_count = 0

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

    def update(self, state, action, reward):
        if self.step_size == -1:
            alpha = 1 / (1 + self.step_count)
        else:
            alpha = self.step_size

        self.pred_q_table[state][action] += alpha * (reward - self.pred_q_table[state][action])


class UCB1:
    def __init__(self, n_states, k_arms, initial_value=0.0, conf_coeff=1.0):
        self.n_states = n_states
        self.k_arms = k_arms
        self.pred_q_table = initial_value * np.ones((n_states, k_arms))
        self.conf_coeff = conf_coeff
        self.action_count = np.zeros(k_arms)

    def __call__(self, state):
        confidence = np.sqrt(np.log(np.sum(self.action_count)) / (self.action_count + 1e-6))
        pred_qs = self.pred_q_table[state] + self.conf_coeff * confidence
        action_candidates = np.arange(self.k_arms)[pred_qs == np.max(pred_qs)]
        action = np.random.choice(action_candidates)
        return action

    def update(self, state, action, reward, alpha):
        self.pred_q_table[state][action] += alpha * (reward - pred_q_table[state][action])


class PolicyGradient:
    def __init__(self, n_states, k_arms, initial_value):
        1+1

    def __call__(self, state):
        raise NotImplementedError

    def update(self, state, action, reward, alpha):
        raise NotImplementedError

# step_sizeはデフォルトで-1で, この状態だと(1/step数)を用いる.
def run(problem, solver, n_steps, step_size=-1):
    k_arms = problem.k_arms
    n_states = problem.n_states

    optimal_action_count = 0
    optimal_action_rates = np.zeros(n_steps)
    cumulative_reward = 0
    average_rewards = np.zeros(n_steps)

    state = 0 # 初期状態が0なのは知っていることにする
    for step in range(n_steps):
        action = solver(state)
        optimal_action_count += (action == problem._get_optimal_action())

        reward, next_state = problem.try_an_arm(action)
        cumulative_reward += reward

        if step_size == -1:
            alpha = 1 / (step+1)
        else:
            alpha = step_size

        solver.update(action, state, reward, alpha)
        state = next_state

        optimal_action_rates[step] = optimal_action_count / (step+1)
        average_rewards[step] = cumulative_reward / (step+1)

    return solver.pred_q_table, optimal_action_rates, average_rewards

def epsilon_greedy(problem, n_steps, epsilon, initial_value=0.0, step_size=-1):
    k_arms = problem.k_arms
    n_states = problem.n_states
    pred_q_table = initial_value * np.ones((n_states, k_arms))

    optimal_action_count = 0
    optimal_action_rates = np.zeros(n_steps)
    cumulative_reward = 0
    average_rewards = np.zeros(n_steps)

    state = 0 # 初期状態が0なのは知っていることにする
    for step in range(n_steps):
        is_greedy = np.random.rand() >= epsilon
        if is_greedy:
            # 行動価値が等しい行動が複数あるときは, その中からランダムに選ぶ
            pred_qs = pred_q_table[state]
            action_candidates = np.arange(k_arms)[pred_qs == np.max(pred_qs)]
        else:
            # 全ての行動
            action_candidates = np.arange(k_arms)
        action = np.random.choice(action_candidates)
        optimal_action_count += (action == problem._get_optimal_action())

        reward, next_state = problem.try_an_arm(action)
        cumulative_reward += reward

        if step_size == -1:
            alpha = 1 / (step+1)
        else:
            alpha = step_size
        pred_q_table[state][action] += alpha * (reward - pred_q_table[state][action])
        state = next_state

        optimal_action_rates[step] = optimal_action_count / (step+1)
        average_rewards[step] = cumulative_reward / (step+1)

    return pred_q_table, optimal_action_rates, average_rewards

