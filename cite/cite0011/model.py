import torch
from torch import nn

## Model
class MsciModel(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        layers = [6.727112318181705, 9.375593325632668, 8.538792791615048, 8.082130034481784]

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_channel, eps=1e-7),
            nn.Linear(input_channel, int(2**layers[0])),
            nn.ReLU(),
            nn.BatchNorm1d(int(2**layers[0]), eps=1e-7),
            nn.Linear(int(2**layers[0]), int(2**layers[1])),
            nn.ReLU(),            
            nn.BatchNorm1d(int(2**layers[1]), eps=1e-7),
            nn.Linear(int(2**layers[1]), int(layers[2])),
            nn.ReLU(),
            nn.BatchNorm1d(int(2**layers[2]), eps=1e-7),
            nn.Linear(int(2**layers[2]), int(2**layers[3])),
            nn.ReLU(),
            nn.BatchNorm1d(int(2**layers[3]), eps=1e-7),
            nn.Linear(int(2**layers[3]), output_channel),
            nn.Softplus()
        )

    
    def forward(self, x):
        return self.mlp(x)