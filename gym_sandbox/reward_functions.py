__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import numpy as np

from numpy import sin, cos


def identity_reward(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: identity  """

    return reward


def energy_scale(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: energy_scale  """

    # state
    vel = state[1]  # cart linear velocity
    avl = state[3]  # pole angular velocity

    # Neural Adaptive Elements That Can Solve Difficult Learning Control Problems
    # http://incompleteideas.net/papers/barto-sutton-anderson-83.pdf
    l = 0.5  # pole length
    mc = 1  # cart mass
    mp = 0.1  # pole mass

    i = 1 / 3 * mp * l ** 2  # moment of inertia of a rod about the end
    rke = 1 / 2 * i * avl ** 2  # rotational kinetic energy
    lke = 1 / 2 * mc * vel ** 2  # linear kinetic energy

    return reward / (rke + lke)


def energy_adjust(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: energy_adjust  """

    # state
    vel = state[1]  # cart linear velocity
    avl = state[3]  # pole angular velocity

    # Neural Adaptive Elements That Can Solve Difficult Learning Control Problems
    # http://incompleteideas.net/papers/barto-sutton-anderson-83.pdf
    l = 0.5  # pole length
    mc = 1  # cart mass
    mp = 0.1  # pole mass

    i = 1 / 3 * mp * l ** 2  # moment of inertia of a rod about the end
    rke = 1 / 2 * i * avl ** 2  # rotational kinetic energy
    lke = 1 / 2 * mc * vel ** 2  # linear kinetic energy

    return reward / (1 + rke + lke)


def distance_penalty(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: dist_penalty  """

    # state
    pos = state[0]  # cart position

    # Gymnasium Documentation
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    pos_min, pos_max = -4.8, 4.8

    rel_dist = abs(pos) / pos_max  # relative distance to center

    return reward - rel_dist


def distance_adjust(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: dist_adjust  """

    # state
    pos = state[0]  # cart position

    # Gymnasium Documentation
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    pos_min, pos_max = -4.8, 4.8

    rel_dist = abs(pos) / pos_max  # relative distance to center

    return reward / (1 + rel_dist)


def acceleration_scale(action: int, state: np.ndarray, reward: float, terminated:bool, truncated:bool, info:dict):

    """ key word: acc_scale  """

    """
    JUST A DRAFT, DO NOT USE!!!
    Requires previous state to compute 2nd order information. 
    """

    # state
    # pos = state[0]
    # vel = state[1]
    ang = state[2]
    avl = state[3]

    # Neural Adaptive Elements That Can Solve Difficult Learning Control Problems
    # http://incompleteideas.net/papers/barto-sutton-anderson-83.pdf
    g = 9.8
    f = 10 if int(action) == 1 else -10
    l = 0.5
    mc = 1
    mp = 0.1

    # Correct equations for the dynamics of the cart-pole system
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    aac = (g * sin(ang) + cos(ang) * ((-f - mp * l * avl ** 2 * sin(ang)) / (mc + mp))) / \
          (l * ((4 / 3) - ((mp * cos(ang) ** 2) / (mc + mp))))

    acc = (f + mp * l * (avl ** 2 * sin(ang) - aac * cos(ang))) / (mc + mp)

    return reward / abs(acc)
