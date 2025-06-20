import torch
from torch import nn

class GradientLayer(nn.Module):
    """
    Custom layer to compute 1st and 2nd derivatives for the heat equation.
    Attributes:
        model: pytorch network model.
    """

    def __init__(self, model):
        self.model = model
        super().__init__()

    def forward(self, tx):
        """
        Computing 1st and 2nd derivatives for the heat equation.
        Args:
            tx: input variables (t, x).
        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
        """

        tx.requires_grad_(True)
        u = self.model(tx)
        du_dtx = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dt = du_dtx[..., 0:1]
        du_dx = du_dtx[..., 1:2]
        d2u_dt = torch.autograd.grad(du_dt, tx, grad_outputs=torch.ones_like(du_dt), create_graph=True)[0]
        d2u_dx = torch.autograd.grad(du_dx, tx, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        d2u_dt2 = d2u_dt[..., 0:1]
        d2u_dx2 = d2u_dx[..., 1:2]

        return u, du_dt, du_dx, d2u_dt2, d2u_dx2
