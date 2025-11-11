# train/train_iv_extractor.py
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"

        # 确保数据至少是二维的 (N_samples, N_features)
        if iv_data.ndim == 1:
            iv_data = iv_data[:, np.newaxis]

        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        # 生成或使用已有归一化元信息
        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

        # 保存归一化信息
        if self.save_meta_path:
            self._save_norm_meta(self.save_meta_path)

    def _compute_norm_meta(self):
        """
        计算归一化元信息。
        IV数据：改为逐特征（列）Min-Max。
        参数：保持逐维度 Min-Max。
        """
        return {
            # **核心修改：沿 axis=0 (样本轴) 计算每列/每个特征的 Min/Max**
            "iv_min": self.iv_data.min(axis=0).tolist(),
            "iv_max": self.iv_data.max(axis=0).tolist(),

            "params_min": [float(self.params[:, i].min()) for i in range(self.params.shape[1])],
            "params_max": [float(self.params[:, i].max()) for i in range(self.params.shape[1])]
        }

    def _apply_norm(self):
        """对 IV 和参数进行 Min-Max 归一化"""

        # 1. IV 数据逐特征归一化
        iv_min = np.array(self.norm_meta["iv_min"], dtype=np.float32)
        iv_max = np.array(self.norm_meta["iv_max"], dtype=np.float32)

        # 注意：这里需要确保 iv_min 和 iv_max 的 shape 可以广播（即与 self.iv_data.shape[1] 匹配）
        # (N_features,) 可以广播到 (N_samples, N_features)
        denominator = iv_max - iv_min
        # 避免除以 0，只在分母非零处归一化
        denominator[denominator == 0] = 1e-12

        self.iv_data = (self.iv_data - iv_min) / denominator

        # 2. 参数逐维度归一化 (保持不变)
        pmin = np.array(self.norm_meta["params_min"], dtype=np.float32)
        pmax = np.array(self.norm_meta["params_max"], dtype=np.float32)

        denominator_p = pmax - pmin
        denominator_p[denominator_p == 0] = 1e-12

        self.params = (self.params - pmin) / denominator_p

    def _save_norm_meta(self, path):
        """保存归一化信息到 JSON 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.norm_meta, f, indent=2)
        print(f"Normalization meta saved to {path}")

    def __len__(self):
        return self.iv_data.shape[0]

    def __getitem__(self, idx):
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx])
        }
