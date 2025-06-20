import torch
from torch import nn
from layer import GradientLayer

class PINN(nn.Module):
    """
    Build a physics informed neural network (PINN) model for the heat equation.
    Attributes:
        network: pytorch network model with input (t, x) and output u(t, x).
        c: 2
        grads: gradient layer.
    """

    def __init__(self, network, c=2):
        """
        Args:
            network: pytorch network model with input (t, x) and output u(t, x).
            c: Default is 2.
        """

        super().__init__()
        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)

    def forward(self, tx_eqn, tx_ini, tx_bnd_up, tx_bnd_down):
        _, du_dt, _, _, d2u_dx2 = self.grads(tx_eqn)
        u_eqn = du_dt - self.c * self.c * d2u_dx2

        u_ini, _, _, _, _ = self.grads(tx_ini)

        u_bnd_up, _, _, _, _ = self.grads(tx_bnd_up)
        u_bnd_down, _, _, _, _ = self.grads(tx_bnd_down)

        return u_eqn, u_ini, u_bnd_up, u_bnd_down

    def build(self):
        return self
