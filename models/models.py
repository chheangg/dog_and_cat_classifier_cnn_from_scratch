import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from d2l import torch as d2l
from torch.nn import functional as F

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

# models from 2.0-cnn-layer.ipynb
class Conv2D(d2l.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, lr=0.01, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.w = nn.Parameter(torch.normal(0, 0.01, (out_channels, in_channels, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(out_channels)) if bias else None
    
    def forward(self, X):
        # create dimension
        p = self.padding
        s = self.stride
        k = self.kernel_size
        
        # if padding gt 1
        if p > 0:
            batch_size, channels, height, width = X.shape
            padded_tensor = torch.zeros((batch_size, channels, height + 2 * p, width + 2 * p),
                                        dtype=X.dtype, device=X.device)
            padded_tensor[:, :, p:height + p, p:width + p] = X
            X = padded_tensor
            
        ## NAIVE IMPLEMENTATION, SUPER SLOW!!
        # # Output size calculation
        # batch_size, channels, input_height, input_width = X.shape
        # output_height = (input_height - k) // s + 1
        # output_width = (input_width - k) // s + 1

        # # Create the result tensor
        # result = torch.zeros((batch_size, self.out_channels, output_height, output_width),
        #                      dtype=X.dtype, device=X.device)
        
        # # Manual loop-based convolution
        # for b in range(batch_size):
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             for out_c in range(self.out_channels):
        #                 # Extract the slice of the input tensor
        #                 input_slice = X[b, :, i * s : i * s + k, j * s : j * s + k]
        #                 # Grab the corresponding kernel
        #                 kernel = self.w[out_c]
        #                 # Perform element-wise multiplication and sum
        #                 result[b, out_c, i, j] = (input_slice * kernel).sum()
        #                 if self.b is not None:
        #                     result[b, out_c, i, j] += self.b[out_c]

        # return result
            
        # unfold X into flattened_kernel_size * patches
        unfolded_X = F.unfold(X, kernel_size=(k, k), padding=0, stride=s)
        
        # unfold weight into out_channel * flattened_kernel_size
        unfolded_weight = self.w.view(self.out_channels, -1)
        
        # Perform matrix multiplication
        output_matrix = unfolded_weight @ unfolded_X
        
        # Calculate output dimensions
        batch_size, _, input_height, input_width = X.shape
        output_height = (input_height - k) // s + 1
        output_width = (input_width - k) // s + 1
        
        # Reshape the output matrix to the correct tensor shape
        output_tensor = output_matrix.view(batch_size, self.out_channels, output_height, output_width)

        return output_tensor + self.b[None, :, None, None] if self.bias else 0  
    
class MaxPool2d(d2l.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, X):
        # Unfold the tensor into patches
        unfolded_X = F.unfold[:](X, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        batch_size, _, L = unfolded_X.shape
        channels = X.shape[1]
    
        # unfold x furthers
        unfolded_X = unfolded_X.view(
            batch_size, channels, self.kernel_size * self.kernel_size, L
        )
        
        # Take max over kernel dimension (not channels)
        pooled_X = unfolded_X.max(dim=2)[0]  # shape: (batch_size, channels, L)
            
        # Get the original dimensions
        height, width = X.shape[2], X.shape[3]
        
        # Correctly calculate output height and width using original dimensions.
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Reshape the pooled output back to the correct 4D tensor shape.
        output = pooled_X.view(batch_size, channels, output_height, output_width)
        
        return output

class Dropout(d2l.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, X):
        # if not training
        if not self.training:
            # return x
            return X
        
        # get dimensions
        batch_size, channels, height, width = X.shape
        
        # create random tensor
        indices = torch.rand((batch_size, channels, height, width))

        # select all of those where the values are less than the set probability
        X = X * (indices > self.p).float()
        
        # rescale
        return X / (1 - self.p)