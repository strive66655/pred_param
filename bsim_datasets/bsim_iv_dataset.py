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

        self.num_v_features = config.vg_points
        self.iv_data = iv_data.astype(np.float32)

        self.V_raw = self.iv_data[:, :self.num_v_features]
        self.I_raw = self.iv_data[:, self.num_v_features:]

        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        if config.log_transform:
            self._apply_log_transform()

        # self.iv_data = self._add_gradients(self.iv_data)
        # 生成或使用已有归一化元信息
        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

        self.iv_data = np.hstack([self.V_norm, self.I_norm])

        # 保存归一化信息
        if self.save_meta_path:
            self._save_norm_meta(self.save_meta_path)

    def _apply_log_transform(self):
        # 仅对电流数据应用 Log 变换
        self.I_raw = np.clip(self.I_raw, a_min=config.clip_min_current, a_max=None)
        self.I_raw = np.log10(self.I_raw)
        print(f"已对电流特征应用 Log10 变换 (Shape: {self.I_raw.shape})")

    def _add_gradients(self, data):
        """
        计算 I-V 曲线的斜率 (Gradient)，并将其作为新特征拼接到原始数据后面。
        这对提取 AGS (Subthreshold Slope) 至关重要。
        """
        N, D = data.shape
        num_curves = config.num_curves
        pts = config.vg_points

        expected_dim = num_curves * pts * config.num_lg
        if D != expected_dim:
            print(f"警告: 数据维度 ({D}) 与 Config 预期 ({expected_dim}) 不符，跳过梯度特征生成。")
            return data

        reshaped = data.reshape(N, -1, pts)
        gradients = np.gradient(reshaped, axis=2)

        gradients_flat = gradients.reshape(N, -1)

        gradients_flat = gradients_flat * 10.0

        new_data = np.hstack([data, gradients_flat])
        print(f"已添加梯度特征 (Shape: {data.shape} -> {new_data.shape})")
        return new_data

    def _compute_norm_meta(self):
        """
        计算归一化元信息。
        V 数据：Z-score 归一化
        I 数据：Z-score 归一化
        参数：Z-score 归一化
        """
        return {
            # 🚨 修改：V 数据 (前 config.vg_points 列) Z-score
            "V_mu": [float(self.V_raw[:, i].mean()) for i in range(self.V_raw.shape[1])],
            "V_sigma": [float(self.V_raw[:, i].std()) for i in range(self.V_raw.shape[1])],

            # 🚨 修改：I 数据 (其余列) Z-score
            "I_mu": [float(self.I_raw[:, i].mean()) for i in range(self.I_raw.shape[1])],
            "I_sigma": [float(self.I_raw[:, i].std()) for i in range(self.I_raw.shape[1])],

            # 参数数据保持 Z-score
            "params_mu": [float(self.params[:, i].mean()) for i in range(self.params.shape[1])],
            "params_sigma": [float(self.params[:, i].std()) for i in range(self.params.shape[1])]
        }

    def _apply_norm(self):

        # --- 1. V 数据归一化 (Z-score) ---
        v_mu = np.array(self.norm_meta["V_mu"], dtype=np.float32)
        v_sigma = np.array(self.norm_meta["V_sigma"], dtype=np.float32)
        v_sigma_safe = np.where(v_sigma == 0, 1e-12, v_sigma)

        self.V_norm = (self.V_raw - v_mu) / v_sigma_safe

        # --- 2. I 数据归一化 (Z-score) ---
        i_mu = np.array(self.norm_meta["I_mu"], dtype=np.float32)
        i_sigma = np.array(self.norm_meta["I_sigma"], dtype=np.float32)
        i_sigma_safe = np.where(i_sigma == 0, 1e-12, i_sigma)

        self.I_norm = (self.I_raw - i_mu) / i_sigma_safe

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