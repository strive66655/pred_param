# models/residual_param_extractor.py
import torch.nn as nn


class ResidualMLPParamExtractor(nn.Module):
    """
    PCA + Residual MLP
    params = Linear(x) + ResidualMLP(x)
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=3, dropout=0.1):
        super().__init__()

        self.linear_head = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )
            )

        # 输出投影
        self.out_proj = nn.Linear(input_dim, output_dim)

        # 初始化建议（稳定训练）
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        x: (B, PCA_dim)
        """
        base = self.linear_head(x)
        h = x
        for block in self.blocks:
            h = h + block(h)
        delta = self.out_proj(h)

        return base + delta
