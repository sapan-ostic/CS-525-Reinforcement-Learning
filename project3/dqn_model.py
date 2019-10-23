#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

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
        self.device = torch.device('cuda:0')
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.tail = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax()
        )
        
        # self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        # self.loss = nn.MSELoss()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.head(x)
        x = self.tail(x.view(x.size(0), -1))
        ###########################
        return x

