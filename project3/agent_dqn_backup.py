#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math
from matplotlib import pyplot as plt 
import csv

import os
import sys


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from agent import Agent
from dqn_model import DQN
import time
"""
you can import any package and define any extrak function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Agent_DQN():
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        # Parameters for q-learning
        self.env = env
        self.GAMMA = 0.99

        self.EPSILON = 0.99
        self.EPS_START = self.EPSILON
        self.EPS_END = 0.1
        self.EPS_DECAY = 1000000

        self.ALPHA = 1e-4
        self.TARGET_UPDATE = 1000
        self.actionSpace = [0,1,2,3]

        # Parameters for Replay Buffer
        self.CAPACITY = 40000 # Memory size
        self.memory = deque(maxlen=self.CAPACITY) #namedtuple to be used
        self.batch_size = 32
        self.memCntr = 0 # Total sample stored, len(memory) 
        self.steps = 0
        self.storeEpsilon = []        
        self.learn_step_counter = 0

        super(Agent_DQN,self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # nAction = env.action_space.n
        
        # Setup device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: ', device)

        #Initial Q
        self.policy_net = DQN(self.ALPHA).to(device) # Behavior Q 
        self.target_net = DQN(self.ALPHA).to(device) # Target Q 

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.ALPHA)
        # self.loss = nn.MSELoss()

        print('hyperparameters and network initialized')
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            policy_net.load_state_dict(torch.load('test'))
            
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass    
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory.append(Transition(*args))
        # self.position = (self.position + 1) % self.CAPACITY #increment position to store next transitions
        self.memCntr = len(self.memory)

    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # generate batch of random sampled transitions from memory
        # return random.sample(self.memory, self.batch_size)

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states)
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # observation = observation.transpose(2,0,1)
        device = self.policy_net.device

        if np.random.random() < self.EPSILON:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([observation], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = self.policy_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        ###########################
        return action

    def optimize_model(self):

        device = self.policy_net.device

        states, actions, next_states, rewards, dones  = self.replay_buffer()

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done = torch.BoolTensor(dones).to(device)

        state_action_values = self.policy_net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = rewards_v.type(torch.cuda.FloatTensor) + next_state_values * self.GAMMA #+ rewards_v

        return nn.MSELoss()(state_action_values,expected_state_action_values)
        
        
    def train(self):
        nEpisodes = 2000

        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.ALPHA)

        # Fill the memory with experiences
        print('Gathering experiences ...')

        while self.memCntr != self.CAPACITY:
            state = self.env.reset()
            state = state.transpose(2,0,1)
            done = False

            while not done:

                action = self.env.get_random_action()
                    
                nextState, reward, done, info = self.env.step(action)
                nextState = nextState.transpose(2,0,1)
                
                self.push(state, action, nextState, reward, done)

                # Mpve to next state    
                state = nextState   
                
                if done:
                    break

        print('Ready to train model ... ') 
        self.scores = []
        AvgRewards = []
        meanScore = 0
        t = 1
        self.iEpisode = 0

        while meanScore < 50:
            self.iEpisode += 1
            # Initialize environment and state
            state = self.env.reset()
            state = state.transpose(2,0,1)
            done = False
            score = 0

            t1 = time.time()
            
            while not done:

                action = self.make_action(state)
                    
                nextState, reward, done, info = self.env.step(action)
                t+=1
                # state = state.transpose(2,0,1)
                nextState = nextState.transpose(2,0,1)

                # Updating memory with new experiences
                self.push(state, action, nextState, reward, done)

                # Mpve to next state    
                state = nextState   
                
                # Batch Optimization
                optimizer.zero_grad()
                loss = self.optimize_model()
                loss.backward()
                optimizer.step()
                score += reward
                
                if t % self.TARGET_UPDATE == 0:
                    print('Updating Target Network . . .')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Update exploration factor
                self.EPSILON = self.EPS_END + (self.EPS_START - self.EPS_END) *math.exp(-1 * self.steps/self.EPS_DECAY)
                self.storeEpsilon.append(self.EPSILON)
                self.steps += 1

            self.scores.append(score)
            meanScore = np.mean(self.scores[-100:])
            AvgRewards.append(meanScore)

            print('Episode: ', self.iEpisode, ' score:', score, ' Avg Score:',meanScore,' epsilon: ', self.EPSILON, ' t: ', time.time()-t1, ' loss:', loss.item())
            
            if self.iEpisode % 1000 == 0:
                torch.save({
                    'epoch': self.iEpisode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'AvgRewards': AvgRewards
                }, 'trainData')

            # if self.iEpisode % 500 == 0:
            #     torch.save(self.policy_net.state_dict(),'test')

        print('======== Complete ========')
        torch.save({
            'epoch': self.iEpisode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'AvgRewards': AvgRewards
        }, 'trainData')

        with open('Rewards.csv', mode='w') as dataFile:
            rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rewardwriter.writerow(AvgRewards)
