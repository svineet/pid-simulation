import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from collections import deque


class PIDModel:
    """
        State must be a 5 length list of
        (Kd', Kp', alpha, e_t, de_t/dt)

        Action space is (Kd', Kp', alpha), this class handles all
        the denormalisation

        Kpmin, Kpmax, etc are parameters set in __init__
    """
    def __init__(self):
        """
            Add arguments to this method as necessary
            Kp_min, Kp_max, etc should be arguments
        """
        pass

    def step(self, action):
        """
            Support this:
            new_state, reward, done = env.step(action)

            done is a boolean whether or not episode is finished
        """
        pass

    def reset(self):
        """
            Restart the episode, clear all data
        """
        pass

