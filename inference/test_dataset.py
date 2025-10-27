import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bsim_datasets.bsmi_dataset import BSIMDataset


def test_dataset():
    ivcv = np.random.rand(10, 7, 6, 17).astype('float32')
    lg = np.random.rand(10, 7).astype('float32')
    params = np.random.rand(10, 28).astype('float32')
    ds = BSIMDataset(ivcv, lg, params)
    sample = ds[0]
    assert 'ivcv' in sample and 'lg' in sample and 'params' in sample
    print('dataset ok')

if __name__ == '__main__':
    test_dataset()
