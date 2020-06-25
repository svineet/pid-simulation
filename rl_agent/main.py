import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from collections import deque

from pid import PIDModel
from agent import Agent, Actor, Critic


def train(args):
    env = PIDEnvironment()

    actor = Actor()
    critic = Critic()
    agent = Agent(env, lr=args["LEARNING_RATE"], actor_model=actor, critic_model=critic,
                  device=args["DEVICE"])
    ema_reward = 0

    stats = {
        "reward_ema": deque([])
    }

    for i in range(args["NUM_EPISODES"]):
        print("Starting episode", i)
        state = env.reset()
        done = False
        total = 0

        agent.start_episode()
        while not done:
            action = agent.get_action(state)
            new_state, reward, done = env.step(action)
            total += reward
            state = new_state

        # Learn from this episode
        agent.learn()

        ema_reward = 0.9*ema_reward + 0.1*total
        if i%10==0:
            stats["reward_ema"].append(ema_reward)
            print("EMA of Reward is", ema_reward)

    return stats


if __name__ == '__main__':
    stats = train({
        "NUM_EPISODES": 2000,
        "LEARNING_RATE": 0.001,
        "DEVICE": "cpu",
        "RENDER_ENV": False
    })

    plt.plot(stats["reward_ema"])
    plt.show()


