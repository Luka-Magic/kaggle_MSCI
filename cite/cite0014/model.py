import torch
from torch import nn

## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        layers = [8, 8, 8]

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_channel, eps=1e-7),
            nn.Linear(input_channel, int(2**layers[0])),
            nn.ReLU(),
            nn.BatchNorm1d(int(2**layers[0]), eps=1e-7),
            nn.Linear(int(2**layers[0]), int(2**layers[1])),
            nn.ReLU(),            
            nn.BatchNorm1d(int(2**layers[1]), eps=1e-7),
            nn.Linear(int(2**layers[1]), int(2**layers[2])),
            nn.ReLU(),
            nn.BatchNorm1d(int(2**layers[2]), eps=1e-7),
            nn.Linear(int(2**layers[2]), output_channel),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.mlp(x)