import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 新增导入

sys.path.append(os.path.dirname(__file__))
from config import config


class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        """
        BSIM 参数提取数据集类 (集成 PCA 版)
        适配输入结构: [N, (num_curves * 3 * vg_points)]
        """
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"

        # 1. 基础数据加载
        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        # 2. 对电流特征进行 Log 变换
        if config.log_transform:
            self._apply_log_transform()

        # 3. 归一化处理 (Min-Max)
        if norm_meta is None:
            # 训练集模式：计算统计量（含 PCA）
            self.norm_meta = self._compute_norm_meta()
            if self.save_meta_path:
                self._save_norm_meta(self.save_meta_path)
        else:
            # 验证/测试集模式：复用统计量
            self.norm_meta = norm_meta

        # 执行基础归一化 (Min-Max)
        self._apply_norm()

        # 4. 执行 PCA 降维
        # 注意：必须在 Min-Max 归一化之后执行
        if getattr(config, 'pca_enabled', False):
            self._apply_pca()

    # def _apply_log_transform(self):
    #     """对电流应用 Log10 变换"""
    #     N, D = self.iv_data.shape
    #     num_c = config.num_curves
    #     pts = config.vg_points
    #     expected_dim = num_c * 3 * pts
    #     assert D == expected_dim, f"特征维度错误: 实际 {D} vs 预期 {expected_dim}"
    #
    #     mask_id = np.zeros(D, dtype=bool)
    #     block_size = 3 * pts
    #     for i in range(num_c):
    #         start = i * block_size
    #         mask_id[start + 2 * pts: start + 3 * pts] = True
    #
    #     currents = self.iv_data[:, mask_id]
    #     currents = np.clip(currents, a_min=config.clip_min_current, a_max=None)
    #     self.iv_data[:, mask_id] = np.log10(currents)
    #     print(f"✅ Log10 变换完成。")
    def _apply_log_transform(self):
        """由于输入仅包含电流，直接对整个 iv_data 应用 Log10 变换"""
        # 验证维度是否匹配 config 里的设置
        expected_dim = config.num_curves * config.vg_points
        current_dim = self.iv_data.shape[1]

        if current_dim != expected_dim:
            print(f"⚠️ 警告: 特征维度 ({current_dim}) 与预期 ({expected_dim}) 不符。")

        # 限制最小电流防止 log(0)
        self.iv_data = np.clip(self.iv_data, a_min=config.clip_min_current, a_max=None)
        self.iv_data = np.log10(self.iv_data)
        print(f"✅ Log10 变换完成 (作用于全量电流特征)。")

    def _compute_norm_meta(self):
        """
        计算归一化和 PCA 统计量。
        """
        # 基础统计量
        meta = {
            "iv_min": self.iv_data.min(axis=0).tolist(),
            "iv_max": self.iv_data.max(axis=0).tolist(),
            "params_mu": self.params.mean(axis=0).tolist(),
            "params_sigma": self.params.std(axis=0).tolist()
        }

        # 如果启用 PCA，在此时计算主成分
        if getattr(config, 'pca_enabled', False):
            print(f"拟合 PCA (目标维度: {config.pca_n_components})...")
            # 1. 临时进行 Min-Max 归一化以便 PCA 拟合
            iv_min = np.array(meta["iv_min"], dtype=np.float32)
            iv_max = np.array(meta["iv_max"], dtype=np.float32)
            denom = iv_max - iv_min
            denom[denom == 0] = 1.0
            temp_iv = (self.iv_data - iv_min) / denom

            # 2. 拟合 PCA
            pca = PCA(n_components=config.pca_n_components)
            pca.fit(temp_iv)

            # 3. 存入元数据
            meta["pca_components"] = pca.components_.tolist()
            meta["pca_mean"] = pca.mean_.tolist()
            meta["pca_explained_variance"] = pca.explained_variance_ratio_.sum().item()
            print(f"PCA 拟合完成，解释方差占比: {meta['pca_explained_variance']:.4f}")

        return meta

    def _apply_norm(self):
        """应用基础归一化"""
        # IV 数据 Min-Max
        iv_min = np.array(self.norm_meta["iv_min"], dtype=np.float32)
        iv_max = np.array(self.norm_meta["iv_max"], dtype=np.float32)
        denominator = iv_max - iv_min
        denominator[denominator == 0] = 1.0
        self.iv_data = (self.iv_data - iv_min) / denominator

        # Params 数据 Z-score
        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)
        p_sigma[p_sigma == 0] = 1.0
        self.params = (self.params - p_mu) / p_sigma

    def _apply_pca(self):
        """应用 PCA 线性变换"""
        if "pca_components" in self.norm_meta:
            components = np.array(self.norm_meta["pca_components"], dtype=np.float32)
            mean = np.array(self.norm_meta["pca_mean"], dtype=np.float32)

            # 变换公式: (X - mean) @ V.T
            self.iv_data = (self.iv_data - mean) @ components.T
            print(f"✅ PCA 变换应用完成。特征维度从 {components.shape[1]} 降至 {components.shape[0]}")

    def inverse_transform_params(self, normalized_params):
        """反归一化预测出的参数"""
        normalized_params = np.asarray(normalized_params, dtype=np.float32)
        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)
        return normalized_params * p_sigma + p_mu

    def _save_norm_meta(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.norm_meta, f, indent=2)
        print(f"Normalization & PCA meta saved to {path}")

    def __len__(self):
        return self.iv_data.shape[0]

    def __getitem__(self, idx):
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx])
        }