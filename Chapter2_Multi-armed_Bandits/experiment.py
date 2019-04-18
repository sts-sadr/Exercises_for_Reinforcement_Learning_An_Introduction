import numpy as np

import config as cfg
import problem
import solver


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

    def print_config(self):
        for key, value in self.__dict__.items():
            print(key, ':', value)

    def run(self):
        1+1

    def _trial(self):
        problem = self._set_problem()
        solver = self._set_solver()

        

    def _set_problem(self):
        if self.problem_name == 'stationary':
            problem = problem.StationaryBanditProblem(self.k_arms,
                                                      self.reward_std,
                                                      self.problem_mean,
                                                      self.problem_std)
        elif self.problem_name == 'nonstationary':
            problem = problem.NonStationaryBanditProblem(self.k_arms,
                                                         self.reward_std,
                                                         self.change_std)
        elif self.problem_name == 'contextual':
            problem = problem.ContextualBanditProblem(self.k_arms,
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
            solver = solver.EpsilonGreedy(self.n_states,
                                          self.k_arms,
                                          self.initial_value,
                                          self.epsilon,
                                          self.step_size)
        elif self.solver_name == 'UCB1':
            solver = solver.UCB1(self.n_states,
                                 self.k_arms,
                                 self.initial_value,
                                 self.conf_coeff,
                                 self.step_size)
        elif self.solver_name == 'policygradient':
            solver = solver.PolicyGradient(self.n_states,
                                           self.k_arms,
                                           self.step_size,
                                           self.baseline_step_size)
        else:
            raise ValueError('solver_name should be "epsilon-greedy" or "UCB1" or "policygradient".')
        return solver

