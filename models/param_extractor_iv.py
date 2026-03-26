# models/param_extractor_iv.py
import torch.nn as nn


class ParamExtractorIVNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_act = nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.output_act(self.net(x))

model = ParamExtractorIVNet()
print(model)