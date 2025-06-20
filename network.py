import torch
from torch import nn

class Network(nn.Module):
    """
    Build a physics informed neural network (PINN) model for the heat equation.
    """

    def __init__(self, num_inputs=2, layers=None, activation=nn.Tanh, num_outputs=1):
        super().__init__()
        if layers is None:
            layers = [32, 32, 32, 32, 32, 32]

        modules = []
        in_features = num_inputs
        for units in layers:
            modules.append(nn.Linear(in_features, units))
            modules.append(activation())
            in_features = units
        modules.append(nn.Linear(in_features, num_outputs))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def build(cls, *args, **kwargs):
        return cls(*args, **kwargs)
