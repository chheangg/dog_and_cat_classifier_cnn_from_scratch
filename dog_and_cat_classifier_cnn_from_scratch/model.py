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
        
        # # if padding gt 1
        # if p > 0:
        #     batch_size, channels, height, width = X.shape
        #     padded_tensor = torch.zeros((batch_size, channels, height + 2 * p, width + 2 * p),
        #                                 dtype=X.dtype, device=X.device)
        #     padded_tensor[:, :, p:height + p, p:width + p] = X
        #     X = padded_tensor
            
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
        unfolded_X = F.unfold(X, kernel_size=k, stride=s, padding=p)
        
        # unfold weight into out_channel * flattened_kernel_size
        unfolded_weight = self.w.view(self.out_channels, -1)
        
        # Perform matrix multiplication
        output_matrix = unfolded_weight @ unfolded_X
        
        # Calculate output dimensions
        batch_size, _, input_height, input_width = X.shape
        output_height = (input_height - k + 2 * p) // s + 1
        output_width = (input_width - k + 2 * p) // s + 1
        
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
        unfolded_X = F.unfold(X, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
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

class GlobalAvgPool2d(d2l.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X.mean(dim=[2, 3], keepdim=True)
    
class ReLU(d2l.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.max(torch.tensor(0.0), X)

class Dropout(d2l.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, X):
        # During inference, just return X unchanged
        if not self.training:
            return X
        
        # Create a mask with the same shape as X
        mask = (torch.rand(X.shape, device=X.device) > self.p).float()
        
        # Apply the mask and scale to maintain the expected value
        return X * mask / (1 - self.p)
    
def L2Regularization(w, lambd):
    return (lambd / 2) * torch.norm(w, 2) ** 2
    
# models from 3.0-resnet-architecture.ipynb

# The following definitions are a direct copy from the d2l, I'm too lazy to work this one out
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

class BatchNorm2d(d2l.Module):    
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
    
class ResNetLayer(d2l.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.ReLU = ReLU()
        
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=1, stride=strides)
        self.bn1 = BatchNorm2d(out_channels, 4)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(out_channels, 4)
        self.conv3 = Conv2D(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = BatchNorm2d(out_channels * 4, 4)
        
        # Skip connection to match input/output dimensions if needed
        if use_1x1conv or in_channels != out_channels * 4:
            self.conv4 = Conv2D(in_channels, out_channels * 4, kernel_size=1, stride=strides)
            self.bn4 = BatchNorm2d(out_channels * 4, 4)
        else:
            self.conv4 = None
        
        self.ReLU = ReLU()
    
    def forward(self, X):
        Y = self.ReLU(self.bn1(self.conv1(X)))
        Y = self.ReLU(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        
        # add skip connection
        if self.conv4:
            X = self.bn4(self.conv4(X))
            
        Y += X
        
        return self.ReLU(Y)

class ResNet50(d2l.Classifier):
    def __init__(self, num_classes, lr,  in_channels=1, dropout_rate=0.5):
        super().__init__()
        self.lr = lr
        self.bias = True
        self.dropout_rate = dropout_rate
        self.conv1 = Conv2D(kernel_size=7, in_channels=in_channels, out_channels=64, stride=2)
        self.pool1 = MaxPool2d(kernel_size=3, stride=2)
        self.dropout1 = Dropout(p=dropout_rate)
        self.conv2 = nn.Sequential(
            ResNetLayer(in_channels=64, out_channels=64, use_1x1conv=True),
            ResNetLayer(in_channels=256, out_channels=64, use_1x1conv=True),
            ResNetLayer(in_channels=256, out_channels=64, use_1x1conv=True)
        )
        self.conv3 = nn.Sequential(
            ResNetLayer(in_channels=256, out_channels=128, use_1x1conv=True),
            ResNetLayer(in_channels=512, out_channels=128, use_1x1conv=True),
            ResNetLayer(in_channels=512, out_channels=128, use_1x1conv=True),
            ResNetLayer(in_channels=512, out_channels=128, use_1x1conv=True)
        )
        self.conv4 = nn.Sequential(
            ResNetLayer(in_channels=512, out_channels=256, use_1x1conv=True),
            ResNetLayer(in_channels=1024, out_channels=256, use_1x1conv=True),
            ResNetLayer(in_channels=1024, out_channels=256, use_1x1conv=True),
            ResNetLayer(in_channels=1024, out_channels=256, use_1x1conv=True),
            ResNetLayer(in_channels=1024, out_channels=256, use_1x1conv=True),
            ResNetLayer(in_channels=1024, out_channels=256, use_1x1conv=True),
        )
        self.conv5 = nn.Sequential(
            ResNetLayer(in_channels=1024, out_channels=512, use_1x1conv=True),
            ResNetLayer(in_channels=2048, out_channels=512, use_1x1conv=True),
            ResNetLayer(in_channels=2048, out_channels=512, use_1x1conv=True),
        )
        self.pool2 = GlobalAvgPool2d()        # Add dropout before the final fully connected layers
        self.dropout2 = Dropout(p=self.dropout_rate)
        self.fc = LinearRegression(in_features=2048, out_features=1000, lr=self.lr, bias=self.bias)
        self.dropout3 = Dropout(p=self.dropout_rate * 0.5)
        self.softmax = SoftmaxRegression(1000, num_classes, lr=self.lr, bias=self.bias)
    
    def forward(self, X):
        Y = self.pool1(self.conv1(X))
        Y = self.dropout1(Y)  # Dropout after initial feature extraction
        
        Y = self.conv2(Y)
        Y = self.conv3(Y)
        Y = self.conv4(Y)
        Y = self.conv5(Y)
        
        Y = self.pool2(Y)
        Y = Y.reshape(Y.shape[0], -1)
        
        Y = self.dropout2(Y)
        
        Y = self.fc(Y)
        Y = self.dropout3(Y)
        
        Y = self.softmax(Y)
        return Y
        
        
    def loss(self, y_hat, y):
        return CrossEntropyError(y_hat, y)
    
    def configure_optimizers(self):
        return SGDFromScratch(self.parameters(), self.lr)