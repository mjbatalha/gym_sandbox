__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import numpy as np


def identity_reward(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):
    return reward
