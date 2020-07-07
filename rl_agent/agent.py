import numpy as np
import pandas as pd

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from collections import namedtuple


Transition = namedtuple("Transition", ["reward", "state", "next_state", "action", "target_action"])


class Actor(nn.Module):
    def __init__(self, h1=50, h2=50, num_actions=3, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))

        return out


class Critic(nn.Module):
    def __init__(self, h1=20, h2=20, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        return out


class Agent:
    def __init__(self, environment, actor_model, critic_model,
                 lr=0.01, gamma=0.9, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.lr = lr
        self.gamma = gamma
        self.device = device

        # Train pure SGD
        self.actor_optimizer = torch.optim.SGD(self.actor_model.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.SGD(self.critic_model.parameters(), lr=lr)

    def get_action(self, state):
        """
            Forward pass for the actor and critic network

            Use gaussian exploration strategy outlined in the paper
        """
        output = self.actor_model(torch.Tensor(list(state)))

        return output

    def start_episode(self):
        self.transitions = deque([])

    def step(self, trans):
        self.transitions.append(trans)

    def learn(self):
        """
            Learn from stored episode data till now

            1. TD learning for Value network
            2. Custom update rule from CACLA for del > 0
            del = advantage = (actual value of state - predicted_value)

            Episode chain is a list of (state, action, next_state, reward) tuples
            stored by Agent itself.
        """
        device = self.device

        batch = Transition(*zip(*self.transitions))
        states = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)

        # Value of the last state = R_t
        state_values = deque([])
        critic_loss = deque([])
        del_ts = deque([(rewards[-1]-self.critic_model(torch.Tensor(states[-1])))])
        loss = torch.Tensor([0])
        for trans in self.transitions:
            next_value = self.critic_model(
                    torch.Tensor(trans.next_state)) if trans.next_state is not None else
            cur_value = self.critic_model(torch.Tensor(trans.state))

            reward = trans.reward
            delta_t = reward + self.gamma*next_value - cur_value

            del_ts.appendleft(delta_t)
            state_values.appendleft(cur_value)

            # Critic loss
            loss += delta_t**2 # Minimise delta_t**2

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        k = 0
        total_loss = torch.Tensor([0])
        for del_t, a_t, ac_t in zip(reversed(del_ts), reversed(self.actions), reversed(self.sug_actions)):
            k += 1
            if del_t > 0:
                self.actor_optimizer.zero_grad()

                if type(a_t) != torch.Tensor:
                    import pdb; pdb.set_trace()

                # Actor loss
                loss = (a_t - ac_t)**2 # Push towards a_t
                total_loss += loss.sum()

        total_loss.backward()
        self.actor_optimizer.step()

    def load(self):
        self.actor_model.load_state_dict(
                torch.load("actor.pkl", map_location=lambda storage, loc: storage))
        self.critic_model.load_state_dict(
                torch.load("critic.pkl", map_location=lambda storage, loc: storage))

    def save(self):
        torch.save(self.actor_model.state_dict(), "actor.pkl")
        torch.save(self.critic_model.state_dict(), "critic.pkl")


if __name__=='__main__':
    actor = Actor()
    print(actor.forward(torch.Tensor([0.5, 0.5, 3, 10, 10])))

    critic = Critic()
    print(critic.forward(torch.Tensor([0.5, 0.5, 3, 10, 10])))

