import torch
import torch.nn as nn

class IV1DCNN_10x21(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            # 输入: (B, 10, 21)
            nn.Conv1d(10, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # 聚合整个 21 长度的序列
            nn.AdaptiveAvgPool1d(1)   # (B, 64, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # x: (B, 10, 21)
        x = self.cnn(x)       # (B, 64, 1)
        x = x.squeeze(-1)    # (B, 64)
        return self.mlp(x)