import torch
from torch import nn


## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_channel),
        )
    
    def forward(self, x):
        return self.mlp(x)