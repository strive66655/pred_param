import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None):
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"
        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)

        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

    def _compute_norm_meta(self):
        return {
            "iv_min": float(self.iv_data.min()),
            "iv_max": float(self.iv_data.max()),
            "params_min": [float(self.params[:, i].min()) for i in range(self.params.shape[1])],
            "params_max": [float(self.params[:, i].max()) for i in range(self.params.shape[1])]
        }

    def _apply_norm(self):
        a, b = self.norm_meta["iv_min"], self.norm_meta["iv_max"]
        self.iv_data = (self.iv_data - a) / (b - a + 1e-12)

        pmin = np.array(self.norm_meta["params_min"], dtype=np.float32)
        pmax = np.array(self.norm_meta["params_max"], dtype=np.float32)
        self.params = (self.params - pmin) / (pmax - pmin + 1e-12)

    def __len__(self):
        return self.iv_data.shape[0]

    def __getitem__(self, idx):
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx])
        }
