"""
VGG-11 for CIFAR-10 (Figure 16).

The paper cites VGG-11 as in Simonyan & Zisserman (2014), i.e. the *original*
network: eight convolutional layers, five max-pools and **no batch
normalisation**.  The usual CIFAR-10 realisation shrinks the 7x7x512 feature map
to 1x1x512 with the five 2x2 max-pools and feeds it to a single 512 -> 10 linear
classifier.
"""

from __future__ import annotations

import torch.nn as nn

CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name: str = "VGG11", num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers(CFG[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(num_classes: int = 10) -> VGG:
    return VGG("VGG11", num_classes)
