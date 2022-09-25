import torch
from torch import nn


## Model
class MsciModel(nn.Module):
    def __init__(self, cfg, input_channel, output_channel):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_channel, int(2**cfg.hidden1)),
            nn.ReLU(),
            nn.Linear(int(2**cfg.hidden1), int(2**cfg.hidden2)),
            nn.ReLU(),            
            nn.Linear(int(2**cfg.hidden2), int(2**cfg.hidden3)),
            nn.ReLU(),
            nn.Linear(int(2**cfg.hidden3), output_channel),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.mlp(x)