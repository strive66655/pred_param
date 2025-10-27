import torch
from torch import nn


class ParamExtractorNet(nn.Module):
    def __init__(self, in_lower=(7, 6, 17), lg_len=7, out_dim=28, hidden_size=1024, num_hidden=3, dropout=0.0):
        super().__init__()
        lower_flat = in_lower[0]*in_lower*in_lower[2]
        self.lower_fc_in = nn.Linear(lower_flat, hidden_size)
        self.lower_hidden = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(max(0, num_hidden - 1))
        ])
        self.lg_fc = nn.Sequential(
            nn.Linear(lg_len, max(4, hidden_size//8)),
            nn.ReLU(),
            nn.Linear(max(4, hidden_size//8), hidden_size)
        )
        self.comb_fn = nn.ModuleList([nn.Linear(hidden_size*2, hidden_size*2)])
        self.final = nn.Linear(hidden_size*2, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.Sigmoid()

    def forward(self, ivcv, lg):
        b = ivcv.shape[0]
        x = ivcv.view(b, -1)
        x = self.act(self.lower_fc_in(x))
        for layer in self.lower_hidden:
            x = self.act(layer(x))
            x = self.dropout(x)
        y = self.lg_fc(lg)
        comb = torch.cat([x, y], dim=1)  # 特征维度dim=1
        for layer in self.comb_fn:
            comb = self.act(layer(comb))
            comb = self.dropout(comb)
        out = self.final(comb)
        out = self.out_act(out)
        return out
