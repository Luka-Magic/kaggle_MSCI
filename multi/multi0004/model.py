import torch
from torch import nn


## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channel, 320),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(320, 360),
            nn.ReLU(),
            nn.Linear(360, 620),
            nn.ReLU(),            
            nn.Linear(620, 440),
            nn.ReLU(),
            nn.Linear(440, output_channel),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.mlp(x)