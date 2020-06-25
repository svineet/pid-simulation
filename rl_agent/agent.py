import numpy as np
import pandas as pd

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F


def Actor(nn.Module):
    def __init__(self, h1=100, h2=100, num_actions=3):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))

        print(out.shape)

        return out

    def train(self):
        pass


def Critic(nn.Module):
    """
        Neural net that returns Value function of a state
    """
    def __init__(self):
        pass

    def forward(self, state):
        pass

    def train(self):
        pass


class Agent:
    def __init__(self, environment, actor_model, critic_model
                 lr=0.001, gamma=0.99, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.gamma = gamma
        self.optimiser = torch.optim.RMSProp(
            list(self.actor_model.parameters())+list(self.critic_model.parameters()),
            lr=lr)

    def get_action(self, state):
        """
            Forward pass for the actor and critic network

            Use gaussian exploration strategy outlined in the paper
        """
        pass

    def start_episode(self):
        """
            Clear all last episode data and start afresh for this episode
        """
        pass

    def step(self, state, action, reward, next_state):
        """
            Store data for later training 
        """
        pass

    def learn(self):
        """
            Learn from stored episode data till now

            1. TD learning for Value network
            2. Custom update rule from CACLA for del > 0
            del = advantage = (actual value of state - predicted_value)
            i.e (del > 0) =>

            Episode chain is a list of (state, action, next_state, reward) tuples
            stored by Agent itself.
        """
        pass


if __name__=='__main__':
    actor = Actor()
    actor.forward()

