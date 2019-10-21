#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F

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