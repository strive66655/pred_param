import os
import json
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bsim_datasets.bsim_iv_dataset import BSIMIVDataset
from models.param_extractor_iv import ParamExtractorIVNet
from bsim_datasets.data_parser import split_train_val_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 300
PATIENCE = 20
MODEL_SAVE = "best_iv_extractor.pth"

def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        iv = batch["iv"].to(DEVICE)
        params = batch["params"].to(DEVICE)
        pred = model(iv)
        loss = loss_fn(pred, params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * iv.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["iv"].to(DEVICE)
            params = batch["params"].to(DEVICE)
            pred = model(iv)
            loss = loss_fn(pred, params)
            total_loss += loss.item() * iv.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(params.cpu().numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    return total_loss / len(loader.dataset), preds, trues

def main():
    # 直接加载预处理后的数据
    data_path = "data/processed/converted_dataset.npz"
    norm_path = "data/processed/norm_stats.json"

    data = np.load(data_path)
    iv, params = data["ivcv"], data["params"]

    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    print(f"Loaded dataset: {iv.shape}, params: {params.shape}")

    # 划分训练与验证集
    x_train, x_val, y_train, y_val = split_train_val_data(iv, params, train_ratio=0.9)

    train_set = BSIMIVDataset(x_train, y_train)
    val_set = BSIMIVDataset(x_val, y_val)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 模型定义
    model = ParamExtractorIVNet(input_dim=iv.shape[1]*iv.shape[2], output_dim=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # 训练
    best_loss = float("inf")
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
            torch.save({"model": model.state_dict(), "norm_stats": norm_stats}, MODEL_SAVE)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break

    print(f"训练完成，最佳验证损失: {best_loss:.6f}")

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    print("Saved training curve: loss_curve.png")

    plt.figure(figsize=(9, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(trues[:, i], preds[:, i], s=20, alpha=0.7)
        plt.plot([trues[:, i].min(), trues[:, i].max()],
                 [trues[:, i].min(), trues[:, i].max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.title(f"Param {i+1}")
    plt.tight_layout()
    plt.savefig("pred_vs_true.png")
    print("Saved: pred_vs_true.png")

if __name__ == "__main__":
    main()
