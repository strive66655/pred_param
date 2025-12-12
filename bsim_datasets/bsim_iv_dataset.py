import sys

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

sys.path.append(os.path.dirname(__file__))
from config import config
from sklearn.decomposition import PCA

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
        # self.iv_data = np.hstack([self.V_norm, self.I_norm, self.G_norm])
        self.iv_data = np.hstack([self.V_norm, self.I_norm])

        self._apply_pca()
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
        data: shape (N, 231) = [V(21), I(210)]
        我们需要将 I 拆成 10 段，每段 21 点，对每段分别求 d(logI)/dV
        """

        V = data[:, :self.num_v_features]  # (N,21)
        I = data[:, self.num_v_features:]  # (N,210)

        num_curves = I.shape[1] // self.num_v_features  # 210 / 21 = 10
        assert I.shape[1] % self.num_v_features == 0, "I 列数不是 V 点数的整数倍"

        gradients = np.zeros_like(I, dtype=np.float32)  # (N,210)

        # 对每条独立曲线求梯度
        for c in range(num_curves):
            start = c * self.num_v_features
            end = (c + 1) * self.num_v_features

            Ic = I[:, start:end]  # (N,21)

            # 对每个样本分别求梯度
            for i in range(I.shape[0]):
                gradients[i, start:end] = np.gradient(Ic[i], V[i])

        print(
            f"添加梯度特征 (拆成 {num_curves} 条曲线): {data.shape} → {(data.shape[0], data.shape[1] + gradients.shape[1])}")
        self.grad_raw = gradients
        new_data = np.hstack([data, gradients])
        return new_data

    def _compute_norm_meta(self):
        V = self.V_raw
        I = self.I_raw
        # G = self.grad_raw  # 新增

        return {
            "V_mu": V.mean(axis=0).tolist(),
            "V_sigma": V.std(axis=0).tolist(),

            "I_mu": I.mean(axis=0).tolist(),
            "I_sigma": I.std(axis=0).tolist(),

            # "G_mu": G.mean(axis=0).tolist(),
            # "G_sigma": G.std(axis=0).tolist(),

            "params_mu": self.params.mean(axis=0).tolist(),
            "params_sigma": self.params.std(axis=0).tolist(),
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

        # g_mu = np.array(self.norm_meta["G_mu"], dtype=np.float32)
        # g_sigma = np.array(self.norm_meta["G_sigma"], dtype=np.float32)
        # self.G_norm = (self.grad_raw - g_mu) / np.where(g_sigma == 0, 1e-12, g_sigma)

        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)
        p_sigma_safe = np.where(p_sigma == 0, 1e-12, p_sigma)

        # Z-score 公式: (X - mu) / sigma
        self.params = (self.params - p_mu) / p_sigma_safe

        # bsim_iv_dataset.py (仅修改 _apply_pca 部分)

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