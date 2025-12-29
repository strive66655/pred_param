import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from config import config


class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        """
        BSIM 参数提取数据集类 (修复版)
        适配输入结构: [N, (num_curves * 3 * vg_points)]
        """
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"

        # 1. 基础数据加载
        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        # 2. 对电流特征进行 Log 变换 (仅影响 Id)
        if config.log_transform:
            self._apply_log_transform()

        # 3. 归一化处理
        # 即使是验证集/测试集，也必须使用 norm_meta (从训练集计算得到的统计量)
        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
            # 仅在是训练集(norm_meta为None)时保存元数据
            if self.save_meta_path:
                self._save_norm_meta(self.save_meta_path)
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

        # 4. 可选: PCA (目前建议关闭，保持物理特征直观性)
        if config.pca_enabled:
            # 如果确实需要 PCA，可以在这里加，但必须小心只对电流做或者整体做
            # 鉴于我们现在的特征具有很强的物理结构 (Vg, Vd, Id)，直接送入 MLP 效果通常更好
            pass

    def _apply_log_transform(self):
        """
        仅对电流通道 (Id) 应用 Log10 变换。
        数据结构: [N, Flattened_Features] -> Reshape -> [N, Num_Curves, 3_Channels, Pts]
        """
        N, D = self.iv_data.shape
        num_c = config.num_curves
        pts = config.vg_points

        try:
            # Reshape 恢复物理结构
            # Channel 0: Vg, Channel 1: Vd, Channel 2: Id
            data_reshaped = self.iv_data.reshape(N, num_c * 3 * pts)

            # 由于我们的特征排列是 [Vg_vec, Vd_vec, Id_vec] for Curve 1...
            # 这里直接用切片操作比较复杂，不如用 mask

            # 更稳健的方法：构建一个 mask
            # 每个 block 长度 = 3 * pts
            # Id 位于 block 的最后 pts 个位置

            mask_id = np.zeros(D, dtype=bool)
            block_size = 3 * pts

            for i in range(num_c):
                start = i * block_size
                # Vg: start ~ start+pts
                # Vd: start+pts ~ start+2*pts
                # Id: start+2*pts ~ start+3*pts
                mask_id[start + 2 * pts: start + 3 * pts] = True

            # 提取电流
            currents = self.iv_data[:, mask_id]

            # Clip & Log
            currents = np.clip(currents, a_min=config.clip_min_current, a_max=None)
            currents_log = np.log10(currents)

            # 放回原数组
            self.iv_data[:, mask_id] = currents_log

            print(f"已对电流特征 (Id) 应用 Log10 变换。")

        except Exception as e:
            print(f"❌ Log变换失败: {e}")
            sys.exit(1)

    def _compute_norm_meta(self):
        """
        计算归一化统计量。
        - IV数据 (Min-Max):
          因为 Vg, Vd 是已知范围，Log(Id) 也是有界的。Min-Max 能很好地保留形状。
        - Params (StandardScaler/Z-score):
          参数通常服从正态分布，用均值方差归一化更好。
        """
        return {
            # IV: 沿 feature 维度计算 min/max
            "iv_min": self.iv_data.min(axis=0).tolist(),
            "iv_max": self.iv_data.max(axis=0).tolist(),

            # Params: 计算均值和标准差
            "params_mu": self.params.mean(axis=0).tolist(),
            "params_sigma": self.params.std(axis=0).tolist()
        }

    def _apply_norm(self):
        """应用归一化"""
        # --- IV 数据 (Min-Max) ---
        iv_min = np.array(self.norm_meta["iv_min"], dtype=np.float32)
        iv_max = np.array(self.norm_meta["iv_max"], dtype=np.float32)

        denominator = iv_max - iv_min
        # 防止除以 0 (例如 Vd=0.1 这一列全是 0.1，max-min=0)
        # 对于常数列，归一化后通常设为 0
        denominator[denominator == 0] = 1.0

        self.iv_data = (self.iv_data - iv_min) / denominator

        # --- Params 数据 (Z-score) ---
        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)

        # 防止除以 0
        p_sigma[p_sigma == 0] = 1.0

        self.params = (self.params - p_mu) / p_sigma

    def inverse_transform_params(self, normalized_params):
        """反归一化预测出的参数"""
        normalized_params = np.asarray(normalized_params, dtype=np.float32)
        p_mu = np.array(self.norm_meta["params_mu"], dtype=np.float32)
        p_sigma = np.array(self.norm_meta["params_sigma"], dtype=np.float32)

        return normalized_params * p_sigma + p_mu

    def _save_norm_meta(self, path):
        """保存元数据"""
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