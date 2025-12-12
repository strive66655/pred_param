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

        self.V_scaler = StandardScaler()
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
        self.iv_data = np.hstack([self.V_norm, self.I_norm])

        self._apply_pca()
        # 保存归一化信息
        if self.save_meta_path and norm_meta is None:  # 仅在训练集 (第一次) 运行时保存
            self._save_norm_meta(self.save_meta_path)

    def _apply_log_transform(self):
        # 仅对电流数据应用 Log 变换
        self.I_raw = np.clip(self.I_raw, a_min=config.clip_min_current, a_max=None)
        self.I_raw = np.log10(self.I_raw)
        print(f"已对电流特征应用 Log10 变换 (Shape: {self.I_raw.shape})")


    def _fit_scalers(self):
        """用原始数据拟合 StandardScaler"""
        # Fit V data
        self.V_scaler.fit(self.V_raw)
        # Fit I data
        self.I_scaler.fit(self.I_raw)
        # Fit params data
        self.params_scaler.fit(self.params)


    def _compute_norm_meta_from_scalers(self):
        """从拟合好的 Scaler 实例中提取均值和方差"""
        return {
            "V_mu": self.V_scaler.mean_.tolist(),
            "V_scale": self.V_scaler.scale_.tolist(),

            "I_mu": self.I_scaler.mean_.tolist(),
            "I_scale": self.I_scaler.scale_.tolist(),

            "params_mu": self.params_scaler.mean_.tolist(),
            "params_scale": self.params_scaler.scale_.tolist(),
        }

    def _load_meta_to_scalers(self):
        """将 JSON 中的均值/方差加载回 Scaler 实例"""
        # 加载 V Scaler 参数
        self.V_scaler.mean_ = np.array(self.norm_meta["V_mu"], dtype=np.float32)
        self.V_scaler.scale_ = np.array(self.norm_meta["V_scale"], dtype=np.float32)
        self.V_scaler.n_features_in_ = len(self.V_scaler.mean_)

        # 加载 I Scaler 参数
        self.I_scaler.mean_ = np.array(self.norm_meta["I_mu"], dtype=np.float32)
        self.I_scaler.scale_ = np.array(self.norm_meta["I_scale"], dtype=np.float32)
        self.I_scaler.n_features_in_ = len(self.I_scaler.mean_)

        # 加载 Params Scaler 参数
        self.params_scaler.mean_ = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        self.params_scaler.scale_ = np.array(self.norm_meta["params_scale"], dtype=np.float32)
        self.params_scaler.n_features_in_ = len(self.params_scaler.mean_)

    def _apply_norm(self):
        """使用 Scaler 实例进行转换"""

        # 1. V 数据归一化
        self.V_norm = self.V_scaler.transform(self.V_raw)

        # 2. I 数据归一化
        self.I_norm = self.I_scaler.transform(self.I_raw)

        # 3. 参数归一化
        self.params = self.params_scaler.transform(self.params)

    def _apply_pca(self):
        """
        根据 norm_meta 决定是拟合 PCA 还是应用已有的 PCA 转换。
        """
        # --- 从 config.py 读取目标维度 ---
        # 如果 config 中没有定义 pca_output_dim，我们使用默认值
        try:
            TARGET_DIM = config.pca_output_dim
        except AttributeError:
            TARGET_DIM = 20  # 默认使用 20 维
            print(f"⚠️ config.pca_output_dim 未定义，默认使用 {TARGET_DIM} 维进行降维。")

        # 确保 TARGET_DIM 是整数
        if not isinstance(TARGET_DIM, int):
            TARGET_DIM = int(TARGET_DIM)

        if "pca_components" not in self.norm_meta:
            # --- 训练集: 拟合 PCA 转换器 ---

            # 将 n_components 设置为固定的整数
            pca = PCA(n_components=TARGET_DIM, svd_solver='full')

            # fit_transform 会自动进行中心化
            self.iv_data = pca.fit_transform(self.iv_data)

            # 保存关键信息到 norm_meta
            # 修正 TypeError: 确保将 numpy.int64 转换为标准 Python int
            self.norm_meta["pca_n_components"] = int(pca.n_components_)
            self.norm_meta["pca_components"] = pca.components_.tolist()
            self.norm_meta["pca_mean"] = pca.mean_.tolist()

            # 可以保存解释方差比，方便分析
            self.norm_meta["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

            # 计算总共保留的方差百分比
            total_variance = np.sum(pca.explained_variance_ratio_)

            print("=" * 60)
            print(f"🔥 PCA 拟合完成：特征从 {config.input_dim} 维降至 {pca.n_components_} 维。")
            print(f"   共保留 {total_variance * 100:.2f}% 的数据方差。")
            print("=" * 60)

        else:
            # --- 验证/测试集: 应用已有的 PCA 转换 ---

            pca_mean = np.array(self.norm_meta["pca_mean"], dtype=np.float32)
            pca_components = np.array(self.norm_meta["pca_components"], dtype=np.float32)

            # 手动应用 PCA: (数据中心化 - 投影)
            iv_data_centered = self.iv_data - pca_mean
            self.iv_data = np.dot(iv_data_centered, pca_components.T)

            print(f"✅ PCA 应用完成：特征已降维至 {self.iv_data.shape[1]} 维。")


    def inverse_transform_params(self, normalized_params):
        """
        逆向转换归一化后的参数 (使用 StandardScaler 的 inverse_transform)。
        """
        normalized_params = np.asarray(normalized_params, dtype=np.float32)

        # 使用 StandardScaler 实例进行反向转换
        # 注意: 传入的数据必须是 shape (N, D)
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
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx])
        }