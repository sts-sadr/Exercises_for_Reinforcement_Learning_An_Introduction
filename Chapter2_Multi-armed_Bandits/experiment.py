import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config as cfg
from problem import StationaryBanditProblem, NonStationaryBanditProblem, ContextualBanditProblem
from solver import EpsilonGreedy, UCB1, PolicyGradient


class Experiment:
    def __init__(self):
        # config
        self.params = {'problem_name': cfg.PROBLEM_NAME,
                       'k_arms': cfg.K_ARMS,
                       'reward_std': cfg.REWARD_STD,
                       'n_states': cfg.N_STATES,
                       'problem_mean': cfg.PROBLEM_MEAN,
                       'problem_std': cfg.PROBLEM_STD,
                       'change_std': cfg.CHANGE_STD,
                       'state_info': cfg.STATE_INFO,
                       'solver_name': cfg.SOLVER_NAME,
                       'step_size': cfg.STEP_SIZE,
                       'initial_value': cfg.INITIAL_VALUE,
                       'epsilon': cfg.EPSILON,
                       'conf_coeff': cfg.CONF_COEFF,
                       'baseline_step_size': cfg.BASELINE_STEP_SIZE,
                       'with_baseline': cfg.WITH_BASELINE,
                       'n_runs': cfg.N_RUNS,
                       'n_steps': cfg.N_STEPS
                       }

        # logging
        self.mean_average_rewards_record = {}
        self.mean_optimal_action_rates_record = {}

    def print_config(self):
        for key in self.params:
            print('{}: {}'.format(key, self.params[key]))

    def set_config(self, param_name, value):
        if param_name not in self.params.keys():
            raise ValueError('Wrong key. Check key list by "print_config" method.')
        self.params[param_name] = value

    def run(self, experiment_name):
        mean_average_rewards = np.zeros(self.params['n_steps'])
        mean_optimal_action_rates = np.zeros(self.params['n_steps'])

        for _ in tqdm(range(self.params['n_runs'])):
            average_rewards, optimal_action_rates = self._trial()
            mean_average_rewards += average_rewards
            mean_optimal_action_rates += optimal_action_rates

        mean_average_rewards /= self.params['n_runs']
        mean_optimal_action_rates /= self.params['n_runs']
        self.mean_average_rewards_record[experiment_name] = mean_average_rewards
        self.mean_optimal_action_rates_record[experiment_name] = mean_optimal_action_rates
        return mean_average_rewards, mean_optimal_action_rates

    def show_results(self):
        plt.title('Average reward')
        plt.xlabel('Steps')
        plt.ylabel('Average reward')
        for experiment_name in self.mean_average_rewards_record:
            plt.plot(self.mean_average_rewards_record[experiment_name], label=experiment_name)
        plt.legend()
        plt.show()

        plt.title('Optimal action(%)')
        plt.xlabel('Steps')
        plt.ylabel('Optimal action(%)')
        for experiment_name in self.mean_optimal_action_rates_record:
            plt.plot(100*self.mean_optimal_action_rates_record[experiment_name], label=experiment_name)
        plt.legend()
        plt.show()

    def _trial(self):
        problem = self._set_problem()
        solver = self._set_solver()

        cumulative_reward = 0
        optimal_action_count = 0
        average_rewards = np.zeros(self.params['n_steps'])
        optimal_action_rates = np.zeros(self.params['n_steps'])

        state = problem.get_initial_state()
        for step in range(self.params['n_steps']):
            action = solver(state)
            reward, next_state = problem.try_an_arm(action)
            solver.update(state, action, reward)

            cumulative_reward += reward
            optimal_action_count += (action == problem._get_optimal_action(state))
            average_rewards[step] = cumulative_reward / (step + 1)
            optimal_action_rates[step] = optimal_action_count / (step + 1)

            state = next_state

        return average_rewards, optimal_action_rates

    def _set_problem(self):
        if self.params['problem_name'] == 'stationary':
            problem = StationaryBanditProblem(self.params['k_arms'],
                                              self.params['reward_std'],
                                              self.params['problem_mean'],
                                              self.params['problem_std'])
        elif self.params['problem_name'] == 'nonstationary':
            problem = NonStationaryBanditProblem(self.params['k_arms'],
                                                 self.params['reward_std'],
                                                 self.params['change_std'])
        elif self.params['problem_name'] == 'contextual':
            problem = ContextualBanditProblem(self.params['k_arms'],
                                              self.params['reward_std'],
                                              self.params['problem_mean'],
                                              self.params['problem_std'],
                                              self.params['n_states'],
                                              self.params['state_info'])
        else:
            raise ValueError('problem_name should be "stationary" or "nonstationary" or "contextual".')
        return problem

    def _set_solver(self):
        if self.params['solver_name'] == 'epsilon-greedy':
            solver = EpsilonGreedy(self.params['n_states'],
                                   self.params['k_arms'],
                                   self.params['step_size'],
                                   self.params['initial_value'],
                                   self.params['epsilon'])
        elif self.params['solver_name'] == 'UCB1':
            solver = UCB1(self.params['n_states'],
                          self.params['k_arms'],
                          self.params['step_size'],
                          self.params['initial_value'],
                          self.params['conf_coeff'])
        elif self.params['solver_name'] == 'policygradient':
            solver = PolicyGradient(self.params['n_states'],
                                    self.params['k_arms'],
                                    self.params['step_size'],
                                    self.params['baseline_step_size'],
                                    self.params['with_baseline'])
        else:
            raise ValueError('solver_name should be "epsilon-greedy" or "UCB1" or "policygradient".')
        return solver
