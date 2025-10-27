import os
from typing import Tuple

import numpy as np


def gen_syn_dataset(N=2000, seed=1234) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    params = rng.uniform(low=0.0, high=1.0, size=(N, 28)).astype(np.float32)
    lg = rng.uniform(low=14.0, high=110.0, size=(N, 7)).astype(np.float32)
    ivcv = np.zeros((N, 7, 6, 17), dtype=np.float32)
    vgs = np.linspace(0.0, 0.8, 17).astype(np.float32)

    for i in range(N):
        p = params[i]
        for j in range(7):
            L = lg[i, j]
            for k in range(6):
                a = p[(k * 4) % 28] * (1.0 + 0.3 * np.sin(L / 30.0))
                b = p[(k * 4 + 1) % 28] * (1.0 + 0.2 * np.cos(L / 25.0))
                baseline = a * (vgs ** (1.0 + 0.2 * b)) + b * np.exp(vgs * 2.0)
                noise = rng.normal(scale=0.01, size=vgs.shape)
                ivcv[i, j, k, :] = baseline + noise
    return ivcv, lg, params

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    ivcv, lg, params = gen_syn_dataset(N=1000)
    np.savez_compressed('data/processed/demo_data.npz', ivcv=ivcv, lg=lg, params=params)
    print('Saved demo data to data/processed/demo_data.npz')
