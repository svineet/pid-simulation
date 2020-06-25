import numpy as np
import pandas as pd

from collections import deque

# Imports as required


def Actor:
    """
        Neural net that returns
    """
    def __init__(self):
        pass

    def forward(self, state):
        pass

    def train(self, state, target_action):
        pass


def Critic:
    """
        Neural net that returns Value function of a state
    """
    def __init__(self):
        pass

    def forward(self, state):
        pass

    def train(self, ):
        pass


class Agent:
    def __init__(self, environment, actor_model, critic_model
                 lr=0.001, gamma=0.99, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.gamma = gamma

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

