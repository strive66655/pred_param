# train/train_iv_extractor.py
import os
import json
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bsim_datasets.bsim_iv_dataset import BSIMIVDataset
from models.param_extractor_iv import ParamExtractorIVNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 200
PATIENCE = 15
MODEL_SAVE = "best_iv_extractor.pth"
NORMALIZE_META = "iv_norm_meta.json"

def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for batch in loader:
        iv = batch["iv"].to(DEVICE)
        params = batch["params"].to(DEVICE)
        pred = model(iv)
        loss = loss_fn(pred, params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * iv.size(0)
    return total / len(loader.dataset)

def eval_model(model, loader, loss_fn):
    model.eval()
    total = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["iv"].to(DEVICE)
            params = batch["params"].to(DEVICE)
            pred = model(iv)
            loss = loss_fn(pred, params)
            total += loss.item() * iv.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(params.cpu().numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    return total / len(loader.dataset), preds, trues

def main():
    data = np.load("data/processed/converted_dataset.npz")
    iv, params = data["ivcv"], data["params"]

    dataset = BSIMIVDataset(iv, params)
    with open(NORMALIZE_META, "w") as f:
        json.dump(dataset.norm_meta, f, indent=2)

    n = len(dataset)
    n_val = int(0.1 * n)
    n_test = int(0.1 * n)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    print(f"Dataset split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader =DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ParamExtractorIVNet(input_dim=iv.shape[1], output_dim=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss = 1e9
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        val_loss, preds, trues = eval_model(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({"model": model.state_dict(), "norm_meta": dataset.norm_meta}, MODEL_SAVE)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break
    print("Training done, best val loss =", best_loss)

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Saved training curve: loss_curve.png")

    # 测试集评估
    ckpt = torch.load(MODEL_SAVE, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    test_loss, preds, trues = eval_model(model, test_loader, loss_fn)
    print(f"Test loss = {test_loss:.6f}")

    # 绘制预测 vs 真实参数散点图
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(trues[:, i], preds[:, i], s=15, alpha=0.7)
        plt.plot([trues[:, i].min(), trues[:, i].max()],
                 [trues[:, i].min(), trues[:, i].max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Param {i + 1}")
    plt.tight_layout()
    plt.savefig("pred_vs_true.png")
    print("Saved prediction vs true scatter plot: pred_vs_true.png")

if __name__ == "__main__":
    main()
