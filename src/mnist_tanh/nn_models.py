"""
Neural-network definitions used by the MNIST experiments.

`Net` is the model behind Figures 13-15: a two-layer perceptron with a tanh
hidden layer and a *linear* output layer feeding the cross-entropy.  In the
paper's notation the per-sample loss is

    f_i(W_1, W_2) = H( W_2^T tanh(W_1^T a_i + c_1) + c_2, b_i ),

    W_1 in R^{784 x 500},  W_2 in R^{500 x 10},  c_1 in R^500,  c_2 in R^10,
    a_i the input image in R^784, b_i the one-hot label in R^10,
    H the multi-class cross-entropy.

Both linear layers include trainable biases c_1 and c_2.
The CIFAR-10 network of Figure 16 is defined separately in ../cifar/vgg.py.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Two-layer MLP 784 -> 500 (tanh) -> 10 used for MNIST (Figures 13-15)."""

    def __init__(self, input_dim: int = 784, output_dim: int = 10,
                 hidden_dim: int = 500):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)
