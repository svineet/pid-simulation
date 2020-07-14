import numpy as np
import pandas as pd

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from collections import namedtuple


Transition = namedtuple("Transition", ["reward", "state", "next_state", "action", "target_action"])


class LunarLanderActor(nn.Module):
    def __init__(self, h1=50, h2=50, num_actions=3, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)
        self.initialise_weights()

    def initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))

        return out


class Actor(nn.Module):
    def __init__(self, h1=50, h2=50, num_actions=3, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_actions)

        self.initialise_weights()

    def initialise_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        # Move ([0, 1], [0, 1], [0, 1]) to ([0, 1], [0, 1], [2, 5])
        return out*torch.Tensor([1, 1, 3]) + torch.Tensor([0, 0, 2])

    def clamp_action(self, action):
        return torch.cat([
            torch.clamp(action[:2], min=0, max=1),
            torch.clamp(action[2:], min=2, max=5)])


class Critic(nn.Module):
    def __init__(self, h1=20, h2=20, state_size=5):
        super().__init__()

        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


class Agent:
    def __init__(self, environment, actor_model, critic_model,
                 critic_lr=0.01, actor_lr=0.01, gamma=0.9, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device

        # Train pure SGD
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_lr)

    def get_action(self, state):
        """
            Forward pass for the actor and critic network

            Use gaussian exploration strategy outlined in the paper
        """
        output = self.actor_model(torch.Tensor(list(state)))

        return output

    def start_episode(self):
        self.transitions = deque([])
        self.del_ts = deque([])

    def step(self, trans):
        self.transitions.append(trans)

        # Learn for critic from this transition
        next_value = self.critic_model(
                torch.Tensor(trans.next_state)) if trans.next_state is not None else torch.Tensor([0])
        cur_value = self.critic_model(torch.Tensor(trans.state))

        reward = trans.reward
        delta_t = reward + self.gamma*next_value - cur_value
        self.del_ts.append(delta_t)

        # Minimise delta_t, i.e push critic towards actual intended value
        self.critic_optimizer.zero_grad()
        loss = 0.5*(delta_t**2)
        loss.backward()
        self.critic_optimizer.step()

    def learn(self):
        batch = Transition(*zip(*self.transitions))

        del_ts = torch.Tensor(self.del_ts)
        a_t = torch.stack(batch.target_action)
        ac_t = torch.stack(batch.action)
        surprise = ((a_t - ac_t)**2).sum(dim=1)

        mask = (del_ts > 0)
        # Minimise (a_t - ac_t)**2 for all t where del_t > 0
        # i.e push Actor to generate a_t when it generated ac_t
        loss = surprise[mask].sum()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def get_episode_stats(self):
        return (self.transitions, self.del_ts)

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

