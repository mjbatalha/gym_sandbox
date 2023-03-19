__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import numpy as np
import torch

from collections import deque
from random import sample
from torch.distributions.bernoulli import Bernoulli
from torch.optim import AdamW

from .reward_functions import identity_reward, energy_scale, energy_adjust, distance_penalty, distance_adjust

OPTIMIZERS = {
    "adamw": AdamW,
}

DISTRIBUTIONS = {
    "bernoulli": Bernoulli,
}

REWARDS = {
    "identity": identity_reward,
    "energy_scale": energy_scale,
    "energy_adjust": energy_adjust,
    "dist_penalty": distance_penalty,
    "dist_adjust": distance_adjust,
}


class REINFORCE:

    # todo: summary

    """
    Implementation inspired by:
    https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py
    Literature:
    - Policy Gradient Methods for Reinforcement Learning with Function Approximation
    https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
    """

    def __init__(self, env, model, params: dict):

        # todo: argument description

        # params
        self.lr = params["lr"]
        self.gamma = params["gamma"]

        # objects
        self.env = env
        self.model = model
        self.optimizer = OPTIMIZERS[params["optimizer"]](self.model.parameters(), lr=self.lr)
        self.reward_function = REWARDS[params["reward"]]
        self.distribution = DISTRIBUTIONS[params["distribution"]]
        self.lps = []
        self.rs = []
        self.info = {}

    def sample_action(self, state: np.ndarray):
        # convert to torch tensor
        state = torch.tensor(np.array([state]))
        # predict distribution parameters
        dist_params = self.model(state)
        # instantiate distribution
        dist = self.distribution(dist_params)
        # sample action
        action = dist.sample()
        # corresponding log probability
        lp = dist.log_prob(action)
        self.lps.append(lp)

        return action

    def update(self):
        # compute step values
        v, vs = 0, []
        for r in self.rs[::-1]:
            v = r + self.gamma * v
            vs.insert(0, v)
        # convert to torch tensor
        vs = torch.tensor(vs)
        # compute loss
        loss = 0
        for lp, v in zip(self.lps, vs):
            loss += -1 * lp.mean() * v
        # weight update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # reset lists
        self.lps = []
        self.rs = []

    def train(self):
        # reset environment
        state, self.info = self.env.reset()
        # iterate over steps until episode ends
        done = False
        while not done:
            # sample action conditioning on state
            action = int(self.sample_action(state)[0][0])  # todo: solve ad hoc indexing & typing for cartpole
            # environment step
            state, reward, terminated, truncated, self.info = self.env.step(action)
            # transform reward
            reward = self.reward_function(action, state, reward, terminated, truncated, self.info)
            self.rs.append(reward)
            # true when episode ends
            done = terminated or truncated
        # update model weights
        self.update()


class DQL:

    # todo: summary

    """
    Implementation inspired by:
    https://github.com/simoninithomas/deep_q_learning/blob/master/DQL%20Cartpole.ipynb
    Literature:
    - Human-level control through deep reinforcement learning
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    """

    def __init__(self, env, model, params: dict):

        # todo: argument description

        # params
        self.lr = params["lr"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.batch_size = params["batch_size"]
        self.mem_size = params["memory_size"]

        # objects
        self.env = env
        self.model = model
        self.optimizer = OPTIMIZERS[params["optimizer"]](self.model.parameters(), lr=self.lr)
        self.reward_function = REWARDS[params["reward"]]
        self.memory = deque(maxlen=self.mem_size)
        self.info = {}

    def sample_action(self, state: np.ndarray):
        # random action sampling
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        # action maximizing estimated q value
        else:
            return torch.argmax(self.model(torch.tensor(state)))

    def update(self):
        # sample batch from memory
        batch = sample(self.memory, self.batch_size)
        # iterate over batch
        qs, states, actions = [], [], []
        for state, action, reward, next_state, terminated, truncated in batch:
            # convert states to torch arrays
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            # q value for last episode step
            if terminated or truncated:
                q = reward
            # otherwise estimate q value
            else:
                q = reward + self.gamma * torch.amax(self.model(next_state))
            # append to batch
            qs.append(q)
            states.append(state)
            actions.append(action)
        # compute loss
        loss = 0
        for q, state, action in zip(qs, states, actions):
            loss += (q - self.model(state)[int(action)]) ** 2
        # weight update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        # reset environment
        state, self.info = self.env.reset()
        # iterate over steps until episode ends
        done = False
        while not done:
            # sample action conditioning on state
            action = self.sample_action(state)
            # environment step
            next_state, reward, terminated, truncated, self.info = self.env.step(int(action))
            # transform reward
            reward = self.reward_function(action, state, reward, terminated, truncated, self.info)
            # append to memory
            self.memory.append((state, action, reward, next_state, terminated, truncated))
            # update state
            state = next_state
            # true when episode ends
            done = terminated or truncated
            # skip updates if not enough memory
            if len(self.memory) < self.batch_size:
                continue
            # update model weights & epsilon
            self.update()
            self.update_epsilon()
