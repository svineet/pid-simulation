import numpy as np
import pandas as pd

import torch

from matplotlib import pyplot as plt

from collections import deque

from pid import PIDModel
from agent import Agent, Actor, Critic


def train(args):
    T_SIZE = 500
    SET_POINT = 50

    t = np.linspace(0, 100, num=T_SIZE)
    SP = np.ones(T_SIZE)*SET_POINT

    env = PIDModel(ku=1.396, tu=3.28, t=t, SP=SP)

    actor = Actor()
    critic = Critic()
    agent = Agent(env, lr=args["LEARNING_RATE"], actor_model=actor, critic_model=critic,
                  device=args["DEVICE"])
    ema_reward = 0

    stats = {
        "reward_ema": deque([])
    }
    torch.autograd.set_detect_anomaly(True)

    if args["LOAD_PREVIOUS"]:
        print("Loading previously trained model")
        agent.load()

    for i in range(args["NUM_EPISODES"]):
        print("Starting episode", i)
        state = env.reset()
        done = False
        total = 0

        agent.start_episode()
        state, _, __ = env.step((0.5, 0.5, 3.5))  # Initial random state
        num_step = 0
        while not done:
            action = agent.get_action(state)

            # Exploration strategy
            gauss_noise = np.random.normal(0, args["exploration_stddev"], size=3)
            target_action = action+torch.Tensor(gauss_noise)

            if (num_step % args["PRINT_EVERY"] == 0):
                print("\tStep", num_step, "for episode", i)
                print("\t", action, target_action)
                print("\tReward accumulated:", total)

            new_state, reward, done = env.step(target_action)
            agent.step(state, target_action, action, reward)

            total += reward
            state = new_state
            num_step += 1

        # Learn from this episode
        agent.learn()

        ema_reward = 0.9*ema_reward + 0.1*total
        if i%1==0:
            agent.save()
            stats["reward_ema"].append(ema_reward)
            print("EMA of Reward is", ema_reward)

    y_caps = np.array(env.output())

    return y_caps


if __name__ == '__main__':
    stats = train({
        "NUM_EPISODES": 100,
        "LEARNING_RATE": 0.01,
        "DEVICE": "cpu",
        "exploration_stddev": 0.2,
        "LOAD_PREVIOUS": True,
        "PRINT_EVERY": 50
    })


