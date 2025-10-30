import torch
from torch.utils.data import Dataset
import numpy as np


class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params):
        assert iv_data.shape[0] == params.shape[0], "样本数量不一致"
        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)

    def __len__(self):
        return self.iv_data.shape[0]

    def __getitem__(self, idx):
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx])
        }
