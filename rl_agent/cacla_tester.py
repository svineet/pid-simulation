import numpy as np
import pandas as pd

import torch

from matplotlib import pyplot as plt

from collections import deque

from pid import PIDModel
from agent import Agent, Actor, Critic, Transition

import gym


def train(args):
    env = gym.make("LunarLanderContinuous-v2")

    actor = Actor(state_size=8, num_actions=2)
    critic = Critic(state_size=8)
    agent = Agent(env,
        actor_lr=args["ACTOR_LEARNING_RATE"], critic_lr=args["CRITIC_LEARNING_RATE"],
        actor_model=actor, critic_model=critic,
        device=args["DEVICE"], gamma=args["GAMMA"])

    stats = {
        "episode_reward": deque([])
    }

    if args["LOAD_PREVIOUS"]:
        print("Loading previously trained model")
        agent.load()

    for i in range(args["NUM_EPISODES"]):
        print("Starting episode", i)
        total = 0

        agent.start_episode()
        state = env.reset()

        num_step = 0; done = False
        while not done:
            action = agent.get_action(state)

            # Exploration strategy
            gauss_noise = np.random.normal(0, args["exploration_stddev"], size=2)
            target_action = torch.clamp(action+torch.Tensor(gauss_noise), min=-1, max=1)

            new_state, reward, done, info = env.step(target_action.detach().numpy())
            transition = Transition(
                    reward=reward, state=state,
                    action=action, target_action=target_action,
                    next_state=new_state)
            agent.step(transition)

            if (num_step % args["PRINT_EVERY"] == 0):
                print("\tStep", num_step, "for episode", i)
                print("\t", action, target_action)
                print("\tReward accumulated:", total)

            assert (type(target_action) == torch.Tensor)
            assert (target_action.requires_grad)
            assert (action.requires_grad)

            total += reward
            state = new_state
            num_step += 1

        # Learn from this episode
        agent.learn()

        if i%1==0:
            agent.save()
            stats["episode_reward"].append(total/num_step) # average reward
            print("Reward for episode was", total)


    return stats


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    stats = train({
        "NUM_EPISODES": 750,
        "DEVICE": "cpu",
        "exploration_stddev": 0.1,
        "LOAD_PREVIOUS": True,
        "PRINT_EVERY": 50,
        "GAMMA": 0.95,
        "CRITIC_LEARNING_RATE": 0.01,
        "ACTOR_LEARNING_RATE": 0.01
    })

    import pdb; pdb.set_trace()
    plt.plot(stats["episode_reward"])
    plt.show()


