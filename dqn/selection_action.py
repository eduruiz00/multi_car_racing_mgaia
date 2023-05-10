#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:58:12 2023

@author: fpinar
"""
# packages
import numpy as np
import torch
import gym
from collections import defaultdict
from simhash import Simhash
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelectionMethod:
    def __init__(self, method: str, epsilon: float = None, temp: float = None, num_episodes: float = None,
                 final: float = None, percentage: float = None, novelty: bool = False):
        self.method = method
        self.epsilon = epsilon
        self.temp = temp
        self.num_episodes = num_episodes
        self.final = final
        self.percentage = percentage
        self.novelty = novelty
        self.counts = {}
        self.rewards = {}

    # file with a selection functions
    def egreedy_pol(self, epsilon, state, dqn, env):
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = dqn(state_tensor)
            action = torch.argmax(q_values).item()

        return action

    def boltzmann_pol(self, temp, state, dqn, env):
        if temp is None:
            raise KeyError("Provide a temperature")

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = dqn(state_tensor)
        pi_a = torch.nn.Softmax(dim=1)(q_values / temp)
        # pi_a=softmax(q_values,temp)
        # action = torch.argmax(q_values).item()
        action = np.random.choice(list(range(env.action_space.n)), 1, p=pi_a.detach().to('cpu').numpy().reshape(-1))
        # torch.argmax(pi_a).item() # Replace this with correct action selection
        # action_tensor = torch.tensor(action.item(), dtype=torch.int64).item()

        return action.item()


    def linear_anneal(self, t, T, start, final, percentage):
        ''' Linear annealing scheduler
        t: current timestep
        T: total timesteps
        start: initial value
        final: value after percentage*T steps
        percentage: percentage of T after which annealing finishes
        '''
        final_from_T = int(percentage * T)
        if t > final_from_T:
            return final
        else:
            return final + (start - final) * (final_from_T - t) / final_from_T

    def explore(self, states_ep, beta=1):
        # states_ep=collection of states in an episode

        for state in states_ep:
            # Compute the SimHash fingerprint of the point
            s = " ".join(str(round(x, 1)) for x in state)
            k = Simhash(s).value
            # Increment the count for the fingerprint
            if k in self.counts.keys():
                self.counts[k] += 1
            else:
                self.counts[k] = 1

        # calculate rewards
        for key in self.counts.keys():
            self.rewards[key] = beta / np.sqrt(self.counts[key])

    def run(self, t, state, dqn, env):
        if self.method == 'egreedy':
            return self.egreedy_pol(self.epsilon, state, dqn, env)
        elif self.method == 'boltzmann':
            return self.boltzmann_pol(self.temp, state, dqn, env)
        elif self.method == 'egreedy_linear_anneal':
            new_epsilon = self.linear_anneal(t, self.num_episodes, self.epsilon, self.final, self.percentage)
            return self.egreedy_pol(new_epsilon, state, dqn, env)
        elif self.method == 'boltzmann_linear_anneal':
            new_temp = self.linear_anneal(t, self.num_episodes, self.temp, self.final, self.percentage)
            return self.boltzmann_pol(new_temp, state, dqn, env)
        else:
            raise KeyError("Unknown selection method")