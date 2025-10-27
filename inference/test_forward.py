import os
import sys

import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.param_extractor import ParamExtractorNet


def test_forward():
    model = ParamExtractorNet()
    ivcv = np.random.rand(2, 7, 6, 17).astype('float32')
    lg = np.random.rand(2, 7).astype('float32')
    ivcv_t = torch.from_numpy(ivcv)
    lg_t = torch.from_numpy(lg)
    out = model(ivcv_t, lg_t)
    assert out.shape == (2, 28)
    print('forward ok')

if __name__ == '__main__':
    test_forward()