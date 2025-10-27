import os

import numpy as np

# 待修改

def load_single_sample(path):
    data = np.load(path)
    return data['ivcv'], data['lg'], data['params']


def main(src_dir, dst_path='data/processed/dataset.npz'):
    samples = []
    for fname in sorted(os.listdir(src_dir)):
        if not fname.lower().endswith(('.npz', '.npz.gz')):
            continue
        ivcv, lg, params = load_single_sample(os.path.join(src_dir, fname))
        samples.append((ivcv, lg, params))

    ivcv = np.stack([s[0] for s in samples], axis=0)
    lg = np.stack([s[1] for s in samples], axis=0)
    params = np.stack([s[2] for s in samples], axis=0)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.savez_compressed(dst_path, ivcv=ivcv, lg=lg, params=params)
    print('Saved processed dataset to', dst_path)


if __name__ == '__main__':
    main('data/raw')
