import numpy as np
import pandas as pd

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, h1=50, h2=50, num_actions=3, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out

    def train(self):
        pass


class Critic(nn.Module):
    """
        Neural Networks that returns Value function of a state
    """
    def __init__(self, h1=50, h2=50, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out

    def train(self):
        pass


class Agent:
    def __init__(self, environment, actor_model, critic_model,
                 lr=0.001, gamma=0.99, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.gamma = gamma
        self.optimiser = torch.optim.RMSprop(
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
    print(actor.forward(torch.Tensor([0.5, 0.5, 3, 10, 10])))

    critic = Critic()
    print(critic.forward(torch.Tensor([0.5, 0.5, 3, 10, 10])))

