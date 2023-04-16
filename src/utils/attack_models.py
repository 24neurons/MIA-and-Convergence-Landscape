"""
This module contains all the architectures that should be used for attacking,
mainly neural nets with 1-3 hidden layers shall be used in most cases.
"""
import torch
from torch import nn

class NNTwoLayers(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits