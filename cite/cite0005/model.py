import torch
from torch import nn


## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channel, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, output_channel),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.mlp(x)