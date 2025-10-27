import numpy as np
import torch
from torch.utils.data import Dataset


class BSIMDataset(Dataset):
    def __init__(self, ivcv: np.ndarray, lg: np.ndarray, params: np.ndarray, norm_meta: dict = None):
        """
        ivcv: (N,7,6,17)
        lg:   (N,7)
        params:(N,28)
        """
        assert ivcv.shape[0] == lg.shape[0] == params.shape[0], "样本数量不一致"
        self.ivcv = ivcv.astype(np.float32)
        self.lg = lg.astype(np.float32)
        self.params = params.astype(np.float32)
        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
        else:
            self.norm_meta = norm_meta
        self._apply_norm()

    def _compute_norm_meta(self):
        meta = {
            'ivcv_min': float(self.ivcv.min()),
            'ivcv_max': float(self.ivcv.max()),
            'lg_min': float(self.lg.min()),
            'lg_max': float(self.lg.max()),
            'params_min': self.params.min(axis=0).astype(float),
            'params_max': self.params.max(axis=0).astype(float)}
        return meta

    def _apply_norm(self):
        a, b = self.norm_meta['ivcv_min'], self.norm_meta['ivcv_max']
        self.ivcv = (self.ivcv - a) / (b - a + 1e-12)
        a, b = self.norm_meta['lg_min'], self.norm_meta['lg_max']
        self.lg = (self.lg - a) / (b - a + 1e-12)
        pmin = np.array(self.norm_meta['params_min'], dtype=np.float32)
        pmax = np.array(self.norm_meta['params_max'], dtype=np.float32)
        self.params = (self.params - pmin) / (pmax - pmin + 1e-12)

    # 数据数量
    def __len__(self):
        return self.ivcv.shape[0]

    def __getitem__(self, idx):
        return {
            'ivcv': torch.from_numpy(self.ivcv[idx]),
            'lg': torch.from_numpy((self.lg[idx])),
            'params': torch.from_numpy(self.params[idx])
        }
