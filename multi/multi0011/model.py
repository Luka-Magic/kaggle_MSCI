import torch
from torch import nn


## Model
class MsciModel(nn.Module):
    def __init__(self, cfg, input_channel, output_channel):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_channel, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout1),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout2),
            nn.Linear(cfg.hidden2, cfg.hidden3),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout3),
            nn.Linear(cfg.hidden3, cfg.hidden4),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout4),
            nn.Linear(cfg.hidden4, cfg.hidden5),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout5),
            nn.Linear(cfg.hidden5, output_channel),
            nn.Softplus()
        )
        
    
    def forward(self, x):
        return self.mlp(x)