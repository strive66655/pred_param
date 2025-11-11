import os
import sys

import torch
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bsim_datasets.config import config
from models.param_extractor_iv import ParamExtractorIVNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict(model_path, sample_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = ParamExtractorIVNet(input_dim=config.input_dim, hidden_layers=config.mlp_layers,
                                output_dim=config.output_dim, dropout=config.dropout_rate).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    data = np.load(sample_path)
    iv = data["ivcv"].astype(np.float32)
    iv = (iv - iv.min()) / (iv.max() - iv.min() + 1e-12)

    with torch.no_grad():
        out = model(torch.from_numpy(iv).to(DEVICE))
    out = out.cpu().numpy()
    np.savetxt("pred_iv_params.txt", out, fmt="%.6f")
    print("Saved predictions to pred_iv_params.txt")

if __name__ == "__main__":
    predict("F:\pred_param\experiments\exp_MLP_20251106_165846\models\\best_iv_extractor.pth", "data/processed/converted_dataset.npz")
