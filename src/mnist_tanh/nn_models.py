"""
Neural-network definitions used by the MNIST and CIFAR experiments.

`Net` is the model behind Figures 13-15 of the paper.  The paper writes the
per-sample loss as

    f_i(X) = H( tanh(X^T a_{1i}) a_{2i}, b_i ),   H = multi-class cross-entropy

which is a **two-layer perceptron with a tanh hidden layer** and a *linear*
output layer (no activation before the cross-entropy).  The symbols map onto
the code as follows.

    a_{1i}   the input image a_i in R^784            (per-sample data)
    X        the first-layer weight  W1 in R^{784x500}
    a_{2i}   the second-layer weight W2 in R^{500x10}
    b_i      the one-hot label in R^10
    H        the multi-class cross-entropy

Two remarks on the paper's notation, since it is easy to misread:

* ``a_{2i}`` carries the sample index ``i`` but is **not** per-sample data --
  it is the output-layer *weight matrix*.  Only ``a_{1i}`` (the image) and
  ``b_i`` (the label) depend on the sample.
* the dimension of ``X`` given in the paper, ``R^{784x10}``, is inconsistent
  with the formula; the hidden width is 500, so ``X`` (that is, ``W1``) lives in
  ``R^{784x500}``.  Both layers also carry biases, which the formula omits.

The hidden width (500) and the use of ``tanh`` are taken from the authors'
original training code (``nn_models.py`` in the experiment sources); the class
below is a faithful re-implementation.
"""

from __future__ import annotations

import torch
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


class LogisticRegression(nn.Module):
    """Single linear layer (multinomial logistic regression, Figure EC.4)."""

    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
