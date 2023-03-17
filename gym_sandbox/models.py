__data__ = "17/03/2023"
__version__ = "0.0.1"
__author__ = "Manuel Batalha"

import torch

from collections import OrderedDict
from torch.nn import Identity, Linear, Module, Sequential, Sigmoid, Tanh

ACTIVATIONS = {
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "identity": Identity
}


class SeqLinear(Module):

    def __init__(self, params: dict):
        super().__init__()

        # params
        in_dims = params["in_dims"]
        out_dims = params["out_dims"]
        l_dims = [in_dims] + params["hidden_dims"] + [out_dims]
        activations = params["activations"]

        # model setup
        model_list = []
        for i, act_name in enumerate(activations):
            layer = ("linear" + str(i), Linear(l_dims[i], l_dims[i + 1]))
            model_list.append(layer)
            act = (act_name + str(i), ACTIVATIONS[act_name]())
            model_list.append(act)

        # sequential container
        self.model = Sequential(OrderedDict(model_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
