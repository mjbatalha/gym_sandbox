__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import numpy as np
import torch

from torch.distributions.bernoulli import Bernoulli
from torch.optim import AdamW

from .reward_functions import identity_reward

OPTIMIZERS = {
    "adamw": AdamW,
}

DISTRIBUTIONS = {
    "bernoulli": Bernoulli,
}

REWARDS = {
    "identity": identity_reward,
}


class REINFORCE:

    # todo: summary & references

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
