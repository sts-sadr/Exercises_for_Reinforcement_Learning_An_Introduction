import numpy as np

import config as cfg
import problem


class Experiment:
    def __init__(self):
        # problem config
        self.problem_name = cfg.PROBLEM_NAME
        self.k_arms = cfg.K_ARMS
        self.reward_std = cfg.REWARD_STD
        self.n_states = cfg.N_STATES
        self.problem_std = cfg.PROBLEM_STD
        self.change_std = cfg.CHANGE_STD
        self.state_info = cfg.STATE_INFO

        # solver config
        self.solver_name = cfg.SOLVER_NAME
        self.step_size = cfg.STEP_SIZE
        self.initial_value = cfg.INITIAL_VALUE
        self.epsilon = cfg.EPSILON
        self.conf_coeff = cfg.CONF_COEFF

        # other config
        self.n_runs = cfg.N_RUNS
        self.n_steps = cfg.N_STEPS

    def print_config(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

    def run(self):
        1+1

    def _trial(self):
        1+1

    def _set_problem(self):
        if self.problem_name == 'stationary':
            problem = problem.StationaryBanditProblem(self.k_arms,
                                                      self.reward_std,
                                                      self.problem_std)
        elif self.problem_name == 'nonstationary':
            problem = problem.NonStationaryBanditProblem(self.k_arms,
                                                         self.reward_std,
                                                         self.change_std)
        elif self.problem_name == 'contextual':
            problem = problem.ContextualBanditProblem(self.k_arms,
                                                      self.reward_std,
                                                      self.problem_std,
                                                      self.n_states,
                                                      self.state_info)
        else:
            raise ValueError('problem_name should be "stationary" or "nonstationary" or "contextual".')

    def _set_solver(self):
        if self.solver_name == 'epsilon-greedy':
            1+1
        elif self.solver_name == 'UCB1':
            1+1
        elif self.solver_name == 'policygradient':
            1+1

        else:
            raise ValueError('solver_name should be "epsilon-greedy" or "UCB1" or "policygradient".')


