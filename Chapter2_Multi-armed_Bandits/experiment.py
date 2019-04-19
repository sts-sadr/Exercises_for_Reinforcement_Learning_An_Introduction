import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config as cfg
from problem import StationaryBanditProblem, NonStationaryBanditProblem, ContextualBanditProblem
from solver import EpsilonGreedy, UCB1, PolicyGradient


class Experiment:
    def __init__(self):
        # problem config
        self.problem_name = cfg.PROBLEM_NAME
        self.k_arms = cfg.K_ARMS
        self.reward_std = cfg.REWARD_STD
        self.n_states = cfg.N_STATES
        self.problem_mean = cfg.PROBLEM_MEAN
        self.problem_std = cfg.PROBLEM_STD
        self.change_std = cfg.CHANGE_STD
        self.state_info = cfg.STATE_INFO

        # solver config
        self.solver_name = cfg.SOLVER_NAME
        self.step_size = cfg.STEP_SIZE
        self.initial_value = cfg.INITIAL_VALUE
        self.epsilon = cfg.EPSILON
        self.conf_coeff = cfg.CONF_COEFF
        self.baseline_step_size = cfg.BASELINE_STEP_SIZE

        # other config
        self.n_runs = cfg.N_RUNS
        self.n_steps = cfg.N_STEPS

        # logging
        self.mean_average_rewards_record = {}
        self.mean_optimal_action_rates_record = {}

    def print_config(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

    def run(self, experiment_name):
        mean_average_rewards = np.zeros(self.n_steps)
        mean_optimal_action_rates = np.zeros(self.n_steps)

        for _ in tqdm(range(self.n_runs)):
            average_rewards, optimal_action_rates = self._trial()
            mean_average_rewards += average_rewards
            mean_optimal_action_rates += optimal_action_rates

        mean_average_rewards /= self.n_runs
        mean_optimal_action_rates /= self.n_runs
        self.mean_average_rewards_record[experiment_name] = mean_average_rewards
        self.mean_optimal_action_rates_record[experiment_name] = mean_optimal_action_rates
        return mean_average_rewards, mean_optimal_action_rates

    def show_results(self):
        plt.title('Average Reward over Step')
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        for experiment_name in self.mean_average_rewards_record:
            plt.plot(self.mean_average_rewards_record[experiment_name], label=experiment_name)
        plt.legend()
        plt.show()

        plt.title('Optimal Action Rate over Step')
        plt.xlabel('Step')
        plt.ylabel('Optimal Action Rate(%)')
        for experiment_name in self.mean_optimal_action_rates_record:
            plt.plot(100*self.mean_optimal_action_rates_record[experiment_name], label=experiment_name)
        plt.legend()
        plt.show()

    def _trial(self):
        problem = self._set_problem()
        solver = self._set_solver()

        cumulative_reward = 0
        optimal_action_count = 0
        average_rewards = np.zeros(self.n_steps)
        optimal_action_rates = np.zeros(self.n_steps)

        state = problem.get_initial_state()
        for step in range(self.n_steps):
            action = solver(state)
            reward, next_state = problem.try_an_arm(action)
            solver.update(state, action, reward)
            state = next_state

            cumulative_reward += reward
            optimal_action_count += (action == problem._get_optimal_action(state))
            average_rewards[step] = cumulative_reward / (step + 1)
            optimal_action_rates[step] = optimal_action_count / (step + 1)

        return average_rewards, optimal_action_rates

    def _set_problem(self):
        if self.problem_name == 'stationary':
            problem = StationaryBanditProblem(self.k_arms,
                                              self.reward_std,
                                              self.problem_mean,
                                              self.problem_std)
        elif self.problem_name == 'nonstationary':
            problem = NonStationaryBanditProblem(self.k_arms,
                                                 self.reward_std,
                                                 self.change_std)
        elif self.problem_name == 'contextual':
            problem = ContextualBanditProblem(self.k_arms,
                                              self.reward_std,
                                              self.problem_mean,
                                              self.problem_std,
                                              self.n_states,
                                              self.state_info)
        else:
            raise ValueError('problem_name should be "stationary" or "nonstationary" or "contextual".')
        return problem

    def _set_solver(self):
        if self.solver_name == 'epsilon-greedy':
            solver = EpsilonGreedy(self.n_states,
                                   self.k_arms,
                                   self.step_size,
                                   self.initial_value,
                                   self.epsilon)
        elif self.solver_name == 'UCB1':
            solver = UCB1(self.n_states,
                          self.k_arms,
                          self.step_size,
                          self.initial_value,
                          self.conf_coeff)
        elif self.solver_name == 'policygradient':
            solver = PolicyGradient(self.n_states,
                                    self.k_arms,
                                    self.step_size,
                                    self.baseline_step_size)
        else:
            raise ValueError('solver_name should be "epsilon-greedy" or "UCB1" or "policygradient".')
        return solver
