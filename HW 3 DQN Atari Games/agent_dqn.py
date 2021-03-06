#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

from agent import Agent
from dqn_model import DQN
import time

import os

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

GAMMA = 0.99

EPSILON = 0.02
EPS_START = EPSILON
EPS_END = 0.005
EPS_DECAY = 1000
batch_size = 32
ALPHA = 1e-5
TARGET_UPDATE = 20000

# Parameters for Replay Buffer
CAPACITY = 500 # Memory size
memory = deque(maxlen=CAPACITY) #namedtuple to be used batch_size = 32
storeEpsilon = []
StartLearning = 100
LOAD = True


class Agent_DQN():
    def __init__(self, env, args):
        # Parameters for q-learning

        super(Agent_DQN,self).__init__()

        self.env = env
        state = env.reset()
        state = state.transpose(2,0,1)

        self.policy_net = DQN(state.shape, self.env.action_space.n) # Behavior Q 
        self.target_net = DQN(state.shape, self.env.action_space.n) # Target Q 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #Initial Q

        if USE_CUDA:
            print("Using CUDA . . .     ")
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()

        print('hyperparameters and network initialized')

        if args.test_dqn or LOAD == True:
            print('loading trained model')
            checkpoint = torch.load('trainData')
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def init_game_setting(self):
        print('loading trained model')
        checkpoint = torch.load('trainData')
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])    
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        memory.append((state, action, reward, next_state, done))
    
    def replay_buffer(self):
        state, action, reward, next_state, done = zip(*random.sample(memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    def make_action(self, observation, test=True):

        observation = observation.transpose(2,0,1)
        
        if np.random.random() > EPSILON or test==True:
            observation   = Variable(torch.FloatTensor(np.float32(observation)).unsqueeze(0), volatile=True)
            q_value = self.policy_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())            
        else:
            action = random.randrange(4)
        return action

    def optimize_model(self):

        states, actions, next_states, rewards, dones  = self.replay_buffer()

        states_v = Variable(torch.FloatTensor(np.float32(states)))
        next_states_v = Variable(torch.FloatTensor(np.float32(next_states)), volatile=True)
        actions_v = Variable(torch.LongTensor(actions))
        rewards_v = Variable(torch.FloatTensor(rewards))
        done = Variable(torch.FloatTensor(dones))

        state_action_values = self.policy_net(states_v).gather(1, actions_v.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        expected_q_value = rewards_v + next_state_values * GAMMA * (1 - done) #+ rewards_v

        loss = (state_action_values - Variable(expected_q_value.data)).pow(2).mean()
        return loss
        
        
    def train(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)

        # Fill the memory with experiences
        print('Gathering experiences ...')
        meanScore = 0
        AvgRewards = []
        AllScores = []
        step = 1
        iEpisode = 0

        while meanScore < 50:
                     
            state = self.env.reset()
            done = False
            EpisodeScore = 0
            tBegin = time.time()
            done = False

            while not done:

                action = self.make_action(state)    
                nextState, reward, done, _ = self.env.step(action)
                self.push(state.transpose(2,0,1), action, nextState.transpose(2,0,1), reward, done)

                state = nextState   
                
                if len(memory) > StartLearning:
                    loss = self.optimize_model()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    iEpisode = 0
                    continue        

                # Update exploration factor
                EPSILON = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step/EPS_DECAY)
                storeEpsilon.append(EPSILON)
                step += 1
                
                EpisodeScore += reward

                if step % TARGET_UPDATE == 0:
                    print('Updating Target Network . . .')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            iEpisode += 1
            AllScores.append(EpisodeScore)
            meanScore = np.mean(AllScores[-100:])
            AvgRewards.append(meanScore)
            
            if len(memory) > StartLearning: 
                print('Episode: ', iEpisode, ' score:', EpisodeScore, ' Avg Score:',meanScore,' epsilon: ', EPSILON, ' t: ', time.time()-tBegin, ' loss:', loss.item())
            else:
                print('Gathering Data . . .')

            if iEpisode % 500 == 0:
                torch.save({
                    'epoch': iEpisode,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'AvgRewards': AvgRewards
                }, 'trainData')

                os.remove("Rewards.csv")
                with open('Rewards.csv', mode='w') as dataFile:
                    rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    rewardwriter.writerow(AvgRewards)

        print('======== Complete ========')
        torch.save({
            'epoch': iEpisode,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'AvgRewards': AvgRewards
        }, 'trainData')

        with open('Rewards.csv', mode='w') as dataFile:
            rewardwriter = csv.writer(dataFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rewardwriter.writerow(AvgRewards)
