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
        self.fc2 = nn.Linear(h1, num_actions)
        self.sig = nn.Sigmoid()

        self.fc1.weight.data.fill_(0.01)
        self.fc1.bias.data.fill_(0.05)

        self.fc2.weight.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.05)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = self.sig(self.fc2(out))

        return out*torch.Tensor([1, 1, 3]) + torch.Tensor([0, 0, 2])


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


class Agent:
    def __init__(self, environment, actor_model, critic_model,
                 lr=0.01, gamma=0.9, device="cpu"):
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.lr = lr
        self.gamma = gamma
        self.device = device

        # Train pure SGD
        self.actor_optimizer = torch.optim.SGD(list(self.actor_model.parameters()), lr=lr, momentum=0)
        self.critic_optimizer = torch.optim.SGD(list(self.critic_model.parameters()), lr=lr, momentum=0)

    def get_action(self, state):
        """
            Forward pass for the actor and critic network

            Use gaussian exploration strategy outlined in the paper
        """
        output = self.actor_model(torch.Tensor(list(state)))

        return output

    def start_episode(self):
        self.states = deque([])
        self.rewards = deque([])
        self.actions = deque([])
        self.sug_actions = deque([])

    def step(self, state, action, suggested_action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.sug_actions.append(suggested_action)

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
        cum_rew = 0
        full_rew = 0
        rewards = np.array(self.rewards)
        discounted_rewards = deque([])
        for reward in reversed(rewards):
            full_rew += reward
            cum_rew = reward + self.gamma*cum_rew
            discounted_rewards.appendleft(cum_rew)

        device = self.device
        discounted_rewards = torch.Tensor(discounted_rewards).to(device=device)


        # Calculate real Monte Carlo value function for each state now
        states = list(self.states)

        # Value of the last state = R_t
        state_values = deque([])
        critic_loss = deque([])
        del_ts = deque([(rewards[-1]-self.critic_model(torch.Tensor(states[-1]))).detach().numpy()[0]])

        for reward, cur_state, next_state in zip(reversed(rewards[:-1]), reversed(states[:-1]), reversed(states[1:])):
            next_value = self.critic_model(torch.Tensor(next_state))
            cur_value = self.critic_model(torch.Tensor(cur_state))
            delta_t = reward + self.gamma*next_value - cur_value

            del_t_num = float(delta_t.detach().numpy())
            del_ts.appendleft(del_t_num)
            state_values.appendleft(cur_value)

            self.critic_optimizer.zero_grad()
            # Critic loss
            (-cur_value*del_t_num).backward()
            self.critic_optimizer.step()

        k = 0
        for del_t, a_t, ac_t in zip(reversed(del_ts), reversed(self.actions), reversed(self.sug_actions)):
            k+=1
            if del_t > 0:
                self.actor_optimizer.zero_grad()

                # Actor loss
                with torch.no_grad():
                    action_advantage = torch.Tensor((a_t - ac_t).detach().numpy())

                (-action_advantage*ac_t).sum().backward()
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

