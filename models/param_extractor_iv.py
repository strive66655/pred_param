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

import torch
import torch.nn as nn

class Conv1DExtractor(nn.Module):
    """
    一维卷积 I-V 特征提取网络
    输入：flatten 的 IV 数据 (batch, D)
    自动 reshape 为 (batch, 1, D) 输入 CNN
    """
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 输入 shape: (B, D)
        x = x.view(x.size(0), 1, self.input_dim)  # → (B,1,D)
        x = self.conv(x)  # → (B,128,1)
        out = self.fc(x)  # → (B, output_dim)
        return out
