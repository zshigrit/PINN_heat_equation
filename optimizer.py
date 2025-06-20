import torch

class L_BFGS_B:
    """Optimizer using PyTorch's L-BFGS algorithm."""
    def __init__(self, model, x_train, y_train, maxiter=30000):
        self.model = model
        self.x_train = [torch.tensor(x, dtype=torch.float32) for x in x_train]
        self.y_train = [torch.tensor(y, dtype=torch.float32) for y in y_train]
        self.maxiter = maxiter

    def fit(self):
        optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=self.maxiter,
                                       line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            outputs = self.model(*self.x_train)
            loss = sum(torch.mean((pred - true) ** 2) for pred, true in zip(outputs, self.y_train))
            loss.backward()
            return loss

        optimizer.step(closure)
