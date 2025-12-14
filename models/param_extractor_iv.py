# param_extractor_iv.py (优化版：添加 1x1 卷积进行通道融合)

import torch
import torch.nn as nn
import numpy as np


# 保持顶部不导入 config，依赖 __init__ 传入的 config

class ParamExtractorIVNet(nn.Module):
    """
    纯 1D 卷积神经网络参数提取器。添加 1x1 卷积以增强通道间特征融合。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = len(config.output_params)

        # 🎯 只构建 CNN 网络
        print(f"🎯 模式: 纯 1D CNN (含 1x1 融合)。输入形状: ({config.cnn_input_channels}, {config.cnn_sequence_length})")
        self.net = self._build_cnn_net(config)

        self.output_act = nn.Identity()

    def _build_cnn_net(self, config):
        """
        构建 10 通道 1D CNN 模型 (只使用两层 Max Pooling)。
        """

        # 1. 卷积特征提取层
        conv_layers = []
        prev_channels = config.cnn_input_channels  # 10

        # 🎯 关键优化：只使用配置中的前两层卷积块
        num_layers_to_use = 2

        for i, (channels, kernel_size) in enumerate(zip(
                config.cnn_channels[:num_layers_to_use], config.cnn_kernel_sizes[:num_layers_to_use]
        )):
            # --- 标准卷积块 ---
            conv_layers += [
                # 使用 padding=kernel_size//2 以保持序列长度在 MaxPool 前不变
                nn.Conv1d(prev_channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                # Max Pooling 进行 2x 降采样
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            prev_channels = channels

            # --- 🎯 1x1 卷积融合块 (添加的新内容) ---
            # 在 MaxPool 之后，使用 1x1 卷积在不改变序列长度 L 的前提下，混合通道 C 的信息
            conv_layers += [
                nn.Conv1d(prev_channels, channels, kernel_size=1, padding=0),  # 1x1 卷积
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            ]
            # 此时 prev_channels 保持不变

        # 2. 计算展平后的维度 (使用一个 dummy tensor)
        dummy_input = torch.zeros(1, config.cnn_input_channels, config.cnn_sequence_length)
        conv_output = nn.Sequential(*conv_layers)(dummy_input)
        flattened_size = conv_output.view(1, -1).size(1)
        print(f"   -> 展平特征维度 (Flattened size): {flattened_size}")

        # 3. 全连接回归层 (MLP)
        mlp_layers = []
        prev_dim = flattened_size

        for hidden_dim in config.cnn_final_mlp_layers:
            mlp_layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ]
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, self.output_dim))

        # 4. 组合网络: Conv Layers -> Flatten -> MLP
        return nn.Sequential(*(conv_layers + [nn.Flatten()] + mlp_layers))

    def forward(self, x):
        """
        输入 x 的形状应为 (Batch, 10, 21)。
        如果输入是展平的 (Batch, 210)，则进行重塑。
        """
        # CNN 模式：确保输入是 3D 的 (Batch, C, L)
        if x.dim() == 2:
            if x.size(1) == self.config.input_dim:
                B = x.size(0)
                C = self.config.cnn_input_channels
                L = self.config.cnn_sequence_length
                x = x.view(B, C, L)

        return self.output_act(self.net(x))