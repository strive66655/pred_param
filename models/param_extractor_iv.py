import torch
import torch.nn as nn

class ParamExtractorIVNet(nn.Module):
    def __init__(self, input_dim=21, hidden_size=512, num_hidden=3, output_dim=3, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_act = nn.Sigmoid()  # 因为数据归一化到了 [0,1]

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.output_act(self.net(x))
