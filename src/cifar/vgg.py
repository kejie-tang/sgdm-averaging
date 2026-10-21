"""
VGG-11 for CIFAR-10 (Figure 16).

The paper cites VGG-11 as in Simonyan & Zisserman (2014), i.e. the *original*
network: eight convolutional layers, five max-pools, the three fully connected
layers of the original classifier (4096 -> 4096 -> 10 with dropout) and **no
batch normalisation**.

Two details of the specification:

* **the classifier head.**  VGG-11's original head is
  ``512*7*7 -> 4096 -> 4096 -> 10``, sized for 224x224 inputs.  CIFAR-10 images
  are 32x32, and five 2x2 pools reduce that to 1x1, so the head's input is
  ``512`` here.
* **the step size.**  ``torch.optim.SGD`` scales its momentum buffer by ``lr``,
  so ``lr = alpha, momentum = gamma`` realises ``x -= alpha * m`` with
  ``m = gamma*m + g``.  The paper's recursion is
  ``m = gamma*m + (1-gamma)*g, x -= alpha*m``, which is
  ``SGD(lr = alpha*(1-gamma), momentum = gamma)`` -- the convention every other
  experiment in this release uses (see README section 6.2, and
  ``docs/momentum_convention.md`` for the derivation).  With the unscaled
  ``lr = alpha`` the effective step is ``alpha/(1-gamma)``, i.e. ten times too
  large at ``gamma = 0.9``.
"""

from __future__ import annotations

import torch.nn as nn

# (number of convolutions, output channels) per block, one max-pool each
CONV_ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


class Vgg11(nn.Module):
    """VGG-11 (no batch norm) with the original three-layer classifier head."""

    def __init__(self, num_classes: int = 10, hidden: int = 4096,
                 dropout: float = 0.5):
        super().__init__()
        blocks = []
        in_channels = 3
        for num_convs, out_channels in CONV_ARCH:
            blocks.append(_vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(out_channels, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.fc(self.flatten(self.conv(x)))


def _vgg_block(num_convs: int, in_channels: int, out_channels: int) -> nn.Sequential:
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=1),
                   nn.ReLU(inplace=True)]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def VGG11(num_classes: int = 10, hidden: int = 4096,
          dropout: float = 0.5) -> Vgg11:
    """VGG-11 with the paper's classifier head.

    ``hidden=4096`` is the original VGG-11 width and what reproduces Figure 16;
    ``hidden`` exists so a smaller head can be used for quick tests.
    """
    return Vgg11(num_classes, hidden=hidden, dropout=dropout)
