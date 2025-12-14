import sys

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

sys.path.append(os.path.dirname(__file__))

from config import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"

        expected_dim = config.num_curves * config.vg_points
        assert iv_data.shape[1] == expected_dim, f"输入数据维度错误: 预期 {expected_dim}, 实际 {iv_data.shape[1]}"
        # 确保数据至少是二维的 (N_samples, N_features)
        if iv_data.ndim == 1:
            iv_data = iv_data[:, np.newaxis]

        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        if config.log_transform:
            self._apply_log_transform()

        self.I_scaler = StandardScaler()
        self.params_scaler = StandardScaler()

        # 生成或使用已有归一化元信息
        if norm_meta is None:
            self._fit_scalers()
            self.norm_meta = self._compute_norm_meta_from_scalers()
        else:
            self.norm_meta = norm_meta
            self._load_meta_to_scalers()

        self._apply_norm()
        # 保存归一化信息
        if self.save_meta_path and norm_meta is None:  # 仅在训练集 (第一次) 运行时保存
            self._save_norm_meta(self.save_meta_path)

    def _apply_log_transform(self):
        # 仅对电流数据应用 Log 变换
        self.iv_data = np.clip(self.iv_data, a_min=config.clip_min_current, a_max=None)
        self.iv_data = np.log10(self.iv_data)
        print(f"已对电流特征应用 Log10 变换 (Shape: {self.iv_data.shape})")


    def _fit_scalers(self):
        """用原始数据拟合 StandardScaler"""
        # Fit I data
        self.I_scaler.fit(self.iv_data)
        # Fit params data
        self.params_scaler.fit(self.params)


    def _compute_norm_meta_from_scalers(self):
        """从拟合好的 Scaler 实例中提取均值和方差"""
        return {
            "I_mu": self.I_scaler.mean_.tolist(),
            "I_scale": self.I_scaler.scale_.tolist(),

            "params_mu": self.params_scaler.mean_.tolist(),
            "params_scale": self.params_scaler.scale_.tolist(),
        }

    def _load_meta_to_scalers(self):
        """将 JSON 中的均值/方差加载回 Scaler 实例"""

        # 加载 I Scaler 参数
        self.I_scaler.mean_ = np.array(self.norm_meta["I_mu"], dtype=np.float32)
        # 避免除以 0 导致 nan 或 inf
        self.I_scaler.scale_ = np.where(np.array(self.norm_meta["I_scale"], dtype=np.float32) == 0,
                                        1e-12,
                                        np.array(self.norm_meta["I_scale"], dtype=np.float32))
        self.I_scaler.n_features_in_ = len(self.I_scaler.mean_)

        # 加载 Params Scaler 参数
        self.params_scaler.mean_ = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        self.params_scaler.scale_ = np.where(np.array(self.norm_meta["params_scale"], dtype=np.float32) == 0,
                                            1e-12,
                                            np.array(self.norm_meta["params_scale"], dtype=np.float32))
        self.params_scaler.n_features_in_ = len(self.params_scaler.mean_)

    def _apply_norm(self):
        """使用 Scaler 实例进行转换"""

        self.iv_data = self.I_scaler.transform(self.iv_data)
        self.params = self.params_scaler.transform(self.params)


    def inverse_transform_params(self, normalized_params):
        """
        逆向转换归一化后的参数 (使用 StandardScaler 的 inverse_transform)。
        """
        normalized_params = np.asarray(normalized_params, dtype=np.float32)
        denormalized_params = self.params_scaler.inverse_transform(normalized_params)
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
        C = config.cnn_input_channels  # 10
        L = config.cnn_sequence_length  # 21
        feature_item = self.iv_data[idx].reshape(C, L)

        return {
            "iv": torch.from_numpy(feature_item),
            "params": torch.from_numpy(self.params[idx])
        }