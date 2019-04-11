import numpy as np


# step_sizeはデフォルトで-1で, この状態だと(1/step数)を用いる.
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


def UCB1(problem, n_steps, c=1.0):
    1+1

def gradient_bandit(problem, n_steps, step_size):
    1+1

