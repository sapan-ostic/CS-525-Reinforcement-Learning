#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import argparse
from collections import namedtuple
from collections import deque
import math


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import gym
from environment import Environment


# In[120]:


# parser
parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
parser.add_argument('--env_name', default=None, help='environment name')
parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
args = parser.parse_args(args=[])


# In[121]:


env_name = 'BreakoutNoFrameskip-v4'
env = Environment(env_name, args, atari_wrapper=True)


# # Network 

# In[122]:


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """
    def __init__(self, ALPHA):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv1 = nn.Sequential(
            nn.Conv2d(4,32, kernel_size=8, stride=4),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=4, stride=2),
            nn.ReLU())
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, stride=1),
            nn.ReLU())
        
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 4) # (input, output) = (4 actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss() 

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation): # x
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # frame = frame[30:200, 5:155,:] #Image cropping to speed up
        # observation = x
        
#         if(len(observation) == 1):
#         observation = torch.Tensor(observation).unsqueeze(1)
#         print(observation.size())
        observation = torch.Tensor(observation).to(self.device) 
#         observation = observation.reshape(observation.size(0),-1) # Shrinking the image
        observation = self.conv1(observation)
        observation = self.conv2(observation)
        observation = self.conv3(observation)
        observation = observation.reshape(observation.size(0),-1) #Flattenning the sequence of image
        observation = self.fc1(observation)
        action = self.fc2(observation)
        
        ###########################
        return action #x


# # Agent

# In[155]:


Transition = namedtuple('Transition', ('state', 'action', 'next_action', 'reward'))

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
        self.GAMMA = 0.95
        self.EPSILON = 0.99
        self.EPS_START = self.EPSILON
        self.EPS_END = 0.05 
        self.EPS_DECAY = 10
        self.ALPHA = 0.003
        self.TARGET_UPDATE = 10000
        # self.REPLACE = 10000
        self.actionSpace = [0,1,2,3]

        # Parameters for Replay Buffer
        self.CAPACITY = 100000 # Memory size
        self.memory = deque(maxlen=self.CAPACITY) #namedtuple to be used
        self.position = 0
        self.memCntr = 0 # Total sample stored, len(memory) 
        self.steps = 0
        
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
        
        print('hyperparameters and network initialized')
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            policy_net.load_state_dict(torch.load('test'))
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
        rand = np.random.random()
        observation = torch.Tensor(observation).unsqueeze(0)
        actions = self.policy_net.forward(observation)
        
#         print(actions[0])
        
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions[0]).item()
        else:
            action = np.random.choice(self.actionSpace)

        # Update exploration factor
        self.EPSILON = self.EPS_END + (self.EPS_START - self.EPS_END) *            math.exp(-1 * self.steps/self.EPS_DECAY)
        
        self.storeEpsilon.append(self.EPSILON)
        self.steps += 1 

        ###########################
        return action
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) == self.CAPACITY:
            self.memory.popleft()
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
        return random.sample(self.memory, self.batch_size)
    
    def get_screen(self,observation):
        meaned = np.mean(observation, axis=2).astype(np.uint8)
        print(np.shape(meaned))
        return meaned
    
    def optimize_model(self):
        self.target_net.optimizer.zero_grad()

        transitions = np.array(self.replay_buffer())
        Qstate = self.policy_net.forward(list(transitions[:,0])).to(self.policy_net.device)
        QNextState = self.policy_net.forward(list(transitions[:,2])).to(self.policy_net.device)                
        
        maxActions = torch.argmax(QNextState, dim=1).to(self.policy_net.device)
        rewards = torch.Tensor(list(transitions[:,3])).to(self.policy_net.device)

        Qtarget = Qstate
        Qtarget[:,maxActions] = rewards + self.GAMMA*torch.max(QNextState[1])

        loss = self.policy_net.loss(Qtarget,Qstate).to(self.policy_net.device)
#         print('loss:', loss)
        loss.backward()
        self.policy_net.optimizer.step()
        
        
    def train(self):
        nEpisodes = 5000
        self.scores = []
        
        for iEpisode in range(nEpisodes):
            print('Episode: ', iEpisode)
            # Initialize environment and state
            state = env.reset()
            state = state.transpose(2,0,1)
            done = False
            score = 0
            
            while not done:
                if self.memCntr < self.batch_size:
                    action = env.get_random_action()
                else:
                    action = self.make_action(state)
                    
                nextState, reward, done, info = env.step(action)
                nextState = nextState.transpose(2,0,1)
                # Add transition to the memory
#                 state_ = self.get_screen(state) #Transformed state
#                 nextState_ = get_screen(nextState)
                self.push(state, action, nextState, reward)

                # Mpve to next state    
                state = nextState   
                
                # Batch Optimization
                if self.memCntr >= self.batch_size:
                    self.optimize_model()             

                if done and info['ale.lives']==0:
                    reward = -100
                    # episode_durations.append(t+1)
                    # plot_durations()
                    break
                score += reward

            print('score:', score)
            print('')
            self.scores.append(score)
            
            if iEpisode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('======== Complete ========')


# In[156]:


agent = Agent_DQN(env, args)
agent.train()
torch.save(agent.policy_net.state_dict(),'test')


# In[154]:



x= [i+1 for i in range(200)]
plt.plot(x,agent.scores)

