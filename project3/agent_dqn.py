#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math
from matplotlib import pyplot as plt 

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
        self.EPS_END = 0.02 
        self.EPS_DECAY = 100000
        self.ALPHA = 1e-4
        self.TARGET_UPDATE = 1000
        self.actionSpace = [0,1,2,3]

        # Parameters for Replay Buffer
        self.CAPACITY = 10000 # Memory size
        self.memory = deque(maxlen=self.CAPACITY) #namedtuple to be used
        self.position = 0
        self.memCntr = 0 # Total sample stored, len(memory) 
        self.steps = 0
        self.iEpisode = 0
        self.storeEpsilon = []
        
        self.learn_step_counter = 0
        self.batch_size = 32

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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.ALPHA)
        self.loss = nn.MSELoss()

        print('hyperparameters and network initialized')
        # self.target_net.load_state_dict(self.policy_net.state_dict())

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            
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
        # if len(self.memory) == self.CAPACITY:
        #     self.memory.popleft()
        self.memory.append(Transition(*args)) 
        self.position = (self.position + 1) % self.CAPACITY #increment position to store next transitions
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
        # rand = np.random.random()
        # observation = torch.Tensor(observation).unsqueeze(0)
        # actions = self.policy_net.forward(observation)

        # if rand < 1 - self.EPSILON:
        #     action = torch.argmax(actions[0]).item()
        # else:
        #     action = np.random.choice(self.actionSpace)

        device = self.policy_net.device

        if np.random.random() < self.EPSILON:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([observation], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = self.policy_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # Update exploration factor
        self.EPSILON = self.EPS_END + (self.EPS_START - self.EPS_END) *math.exp(-1 * self.steps/self.EPS_DECAY)
        self.storeEpsilon.append(self.EPSILON)
        self.steps += 1

        ###########################
        return action

    def optimize_model(self):        
        self.optimizer.zero_grad()

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

        # print(type(next_state_values))
        # print(type(self.GAMMA))
        # print(type(rewards_v))

        expected_state_action_values = rewards_v.type(torch.cuda.FloatTensor) + next_state_values * self.GAMMA #+ rewards_v

        # transitions[0][0] = state[1] = 4 images [4x84x84]
        # transitions = np.array(self.replay_buffer())

        # # Get Q values for State and Action [32x4] 
        # Qstate = self.policy_net.forward(list(transitions[:,0])).to(self.policy_net.device)

        # # Get Q values for State and Action [32x4]
        # QNextState = self.target_net.forward(list(transitions[:,2])).to(self.policy_net.device)    

        # # Find the action with max Q values at the next state
        # maxActions = torch.argmax(QNextState, dim=1).to(self.policy_net.device)

        # rewards = torch.reshape(torch.Tensor(list(transitions[:,3])), (self.batch_size,1)).to(self.policy_net.device)

        # # print('--------Qstate---------')
        # print(Qstate)

        # # Qtarget.data.copy(Qstate.data)
        # # Qtarget = torch.tensor.new_tensor(Qstate)
        # Qtarget = Qstate.clone()    .detach()

        # # Qtarget[:,maxActions] = rewards + self.GAMMA*torch.max(QNextState[1])
        # temp = torch.reshape(torch.tensor([self.GAMMA*torch.max(QNextState[i]) for i in range(self.batch_size)]), (self.batch_size,1)).to(self.policy_net.device)
        

        # print(temp)
        # print(rewards)
        # for i in range(self.batch_size):
        #     Qtarget[i,maxActions[i]] = rewards[i] + temp[i]

        # Qtarget[:, maxActions] = rewards + temp
        # print('--------QNextState---------')
        # print(QNextState)
        # print('--------Qtarget---------')
        # print(Qtarget)
        # print('--------Qstate---------')
        # print(Qstate)
        # print('$$$$$$$$$$$$$$$$$')


        # loss = self.policy_net.loss(Qtarget,Qstate).to(self.policy_net.device)
        loss = self.loss(expected_state_action_values,state_action_values).to(device)
        print('loss:', loss)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        
    def train(self):
        nEpisodes = 100_000

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
        t = 1

        for self.iEpisode in range(nEpisodes):
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
                nextState = nextState.transpose(2,0,1)

                # Updating memory with new experiences
                self.push(state, action, nextState, reward, done)

                # Mpve to next state    
                state = nextState   
                
                # Batch Optimization
                self.optimize_model()             

                # print(info['ale.lives'])
                if done and info['ale.lives']==0:
                    reward = -100
                    # episode_durations.append(t+1)
                    # plot_durations()
                    break
                score += reward
                if t % self.TARGET_UPDATE == 0:
                    print('Updating Target Network . . .')
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.scores.append(score)
            meanScore = np.mean(self.scores[-100:])
            
            print('Episode: ', self.iEpisode, ' score:', score, ' Avg Score:',meanScore,' epsilon: ', self.EPSILON, ' t: ', time.time()-t1 )
            # print()
            
            # print('')


            if self.iEpisode % 100 == 0:
                torch.save(self.policy_net.state_dict(),'test')

            #if self.iEpisode % self.TARGET_UPDATE == 0:
                #print('')
                #print('----- Updating Target network -----')
                #self.target_net.load_state_dict(self.policy_net.state_dict())
        
        print('======== Complete ========')
        # self.device = torch.device('cuda:0')