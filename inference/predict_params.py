import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.param_extractor import ParamExtractorNet
from bsim_datasets.bsmi_dataset import BSIMDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(model_ckpt, sample_npz, out_txt='pred_params.txt'):
    # 加载归一化信息
    norm_meta_path = os.path.join(os.path.dirname(model_ckpt), 'norm_meta.json')
    with open(norm_meta_path, 'r') as f:
        norm_meta = json.load(f)

    # 加载数据
    data = np.load(sample_npz)
    ivcv, lg = data['ivcv'], data['lg']

    dataset = BSIMDataset(ivcv, lg, np.zeros((len(ivcv), len(norm_meta['params_min']))))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    model = ParamExtractorNet(hidden_size=1024, num_hidden=3).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            ivcv_t = batch['ivcv'].to(device)
            lg_t = batch['lg'].to(device)
            pred = model(ivcv_t, lg_t).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    # 反归一化
    pmin = np.array(norm_meta['params_min'], dtype=np.float32)
    pmax = np.array(norm_meta['params_max'], dtype=np.float32)
    real_params = pmin + preds * (pmax - pmin)

    # 保存：如果是 batch，默认保存每个样本一行
    np.savetxt(out_txt, real_params, fmt='%0.6e')
    print('Saved predicted params to', out_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--sample', type=str, required=True, help='Path to sample npz')
    parser.add_argument('--out', type=str, default='pred_params.txt')
    args = parser.parse_args()
    predict(args.model, args.sample, args.out)
