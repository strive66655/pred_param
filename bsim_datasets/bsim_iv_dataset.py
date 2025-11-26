import sys

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

sys.path.append(os.path.dirname(__file__))
from config import config

class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"

        # 确保数据至少是二维的 (N_samples, N_features)
        if iv_data.ndim == 1:
            iv_data = iv_data[:, np.newaxis]

        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        if config.log_transform:
            self._apply_log_transform()

        # 生成或使用已有归一化元信息
        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

        # 保存归一化信息
        if self.save_meta_path:
            self._save_norm_meta(self.save_meta_path)

    def _apply_log_transform(self):

        split_idx = config.vg_points
        if split_idx >= self.iv_data.shape[1]:
            print("⚠️ 警告: vg_points 大于特征维度，未执行 Log 变换")
            return
            V_part = self.iv_data[:, :split_idx]
            I_part = self.iv_data[:, split_idx:]
            I_part = np.clip(I_part, a_min=config.clip_min_current, a_max=None)
            I_part = np.log10(I_part)
            self.iv_data = np.hstack([V_part, I_part])
            print(f"✅ Applied Log10 transform. Cols {split_idx}:end (Currents) clipped to {config.clip_min_current} and logged.")
    def _compute_norm_meta(self):
        """
        计算归一化元信息。
        IV数据：保持逐特征（列）Min-Max 归一化。
        参数：改为逐维度 Z-score 归一化。
        """
        return {
            # IV 数据保持 Min-Max
            "iv_min": self.iv_data.min(axis=0).tolist(),
            "iv_max": self.iv_data.max(axis=0).tolist(),

            "params_mu": [float(self.params[:, i].mean()) for i in range(self.params.shape[1])],
            "params_sigma": [float(self.params[:, i].std()) for i in range(self.params.shape[1])]
        }

    def _apply_norm(self):

        iv_min = np.array(self.norm_meta["iv_min"], dtype=np.float32)
        iv_max = np.array(self.norm_meta["iv_max"], dtype=np.float32)

        denominator = iv_max - iv_min
        # 避免除以 0，只在分母非零处归一化
        denominator[denominator == 0] = 1e-12

        self.iv_data = (self.iv_data - iv_min) / denominator

        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)
        p_sigma_safe = np.where(p_sigma == 0, 1e-12, p_sigma)

        # Z-score 公式: (X - mu) / sigma
        self.params = (self.params - p_mu) / p_sigma_safe

    def inverse_transform_params(self, normalized_params):
        """
        逆向转换归一化后的参数 (Z-score 反归一化)。
        Formula: X = Z * sigma + mu
        """
        normalized_params = np.asarray(normalized_params, dtype=np.float32)
        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)
        denormalized_params = normalized_params * p_sigma + p_mu
        return denormalized_params


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