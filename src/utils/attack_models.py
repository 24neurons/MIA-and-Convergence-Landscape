import torch 
from torch import nn

__all__ = ['NN_attacker']

class LogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = nn.Linear(in_features=10,
                                out_features=1)
        self.relu = nn.ReLU()
    def forward(self, input):
        
        output = self.block1(input)
        return output