import numpy as np


# step_sizeはデフォルトで-1で, この状態だと(1/step数)を用いる.
def epsilon_greedy(problem, n_steps, epsilon, initial_value=0.0, step_size=-1):
    k_arms = problem.k_arms
    n_states = problem.n_states
    pred_q_table = initial_value * np.ones((n_states, k_arms))

    cumulative_reward = 0
    selected_actions = np.zeros(n_steps)
    average_rewards = np.zeros(n_steps)

    state = 0
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

        reward, state = problem.try_an_arm(action)
        cumulative_reward += reward

        if step_size == -1:
            alpha = 1 / (step+1)
        else:
            alpha = step_size
        pred_q_table[state][action] += alpha * (reward - pred_q_table[state][action])

        selected_actions[step] = action
        average_rewards[step] = cumulative_reward / (step+1)

        return pred_table, selected_actions, average_rewards


def UCB1(problem, n_steps, c=1.0):
    1+1

def gradient_bandit(problem, n_steps, step_size):
    1+1

