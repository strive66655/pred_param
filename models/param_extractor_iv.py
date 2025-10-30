# models/param_extractor_iv.py
import torch
import torch.nn as nn

class ParamExtractorIVNet(nn.Module):
    def __init__(self, input_dim=63, hidden_size=1024, num_hidden=3, output_dim=3, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_act = nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.output_act(self.net(x))
