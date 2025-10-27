import argparse

import numpy as np
import torch

from models.param_extractor import ParamExtractorNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(path, device=device):
    ck = torch.load(path, map_location=device)
    return ck

def predict(model_ckpt, sample_npz, out_txt='pred_params.txt'):
    ck = load_checkpoint(model_ckpt)
    model = ParamExtractorNet()
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    norm_meta = ck.get('norm_meta', None)

    data = np.load(sample_npz)
    ivcv, lg = data['ivcv'], data['lg']

    ivcv_t = torch.from_numpy(ivcv.astype('float32')).to(device)
    lg_t = torch.from_numpy(lg.astype('float32')).to(device)
    with torch.no_grad():
        pred = model(ivcv_t, lg_t).cpu().numpy()
    if norm_meta is not None:
        pmin = np.array(norm_meta['params_min'], dtype=np.float32)
        pmax = np.array(norm_meta['params_max'], dtype=np.float32)
        real_params = pmin + pred * (pmax - pmin)
    else:
        real_params = pred
    # save: if batch, save first sample
    if real_params.ndim == 2:
        real_params = real_params[0]
    np.savetxt(out_txt, real_params, fmt='%0.6e')
    print('Saved predicted params to', out_txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='experiments/exp_001/best_model.pth')
    parser.add_argument('--sample', type=str, default='data/processed/sample.npz')
    parser.add_argument('--out', type=str, default='pred_params.txt')
    args = parser.parse_args()
    predict(args.model, args.sample, args.out)

