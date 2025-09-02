import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from d2l import torch as d2l

# utils
def MSE(y_hat, y):
    n = y.numel()
    return torch.sum((y - y_hat) ** 2) / n

class SGDFromScratch(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.lr = lr 

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.data.add_(-self.lr, param.grad.data)
        
        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()

def softmax(o):
    o_max = o.max(dim=1, keepdim=True).values
    transformed_logits = o - o_max
    return torch.exp(transformed_logits) / torch.exp(transformed_logits).sum(dim=1, keepdim=True)

def CrossEntropyError(y_hat, y):
    row_indices = torch.arange(y.shape[0])
    return -torch.log(y_hat[row_indices, y]).mean()

# model from 1.0-building-blocks.ipynb
class LinearRegression(d2l.Module):
    def __init__(self, in_features, out_features, lr=0.01, bias=True):
        super().__init__()
        # weight, add requires_grad=True to perform manual weight changes
        self.w = nn.Parameter(torch.normal(0, 0.01, (in_features, out_features)))
        # bias
        self.b = nn.Parameter(torch.zeros(out_features) if bias else None)
        # will be covered later
        self.lr = lr
        self.bias = bias

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

class SoftmaxRegression(d2l.Classifier):
    def __init__(self, in_features, out_features, lr=0.01, bias=True):
        super().__init__()
        self.lr = lr
        self.bias = bias
        self.net = LinearRegression(in_features, out_features, lr, bias)
        
    def forward(self, X):
        X = X.reshape((-1, self.net.w.shape[0]))
        return softmax(torch.matmul(X, self.net.w) + self.net.b)
    
    def loss(self, y_hat, y):
        return CrossEntropyError(y_hat, y)

    def configure_optimizers(self):
        return SGDFromScratch(self.parameters(), self.lr)



