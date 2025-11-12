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
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bsim_datasets.config import config
from bsim_datasets.bsim_iv_dataset import BSIMIVDataset
from models.param_extractor_iv import ParamExtractorIVNet

DEVICE = config.device
LR = config.learning_rate
BATCH_SIZE = config.batch_size
NUM_EPOCHS = config.epochs
PATIENCE = config.early_stopping_patience
MODEL_SAVE = config.model_dir / "best_iv_extractor.pth"
NORMALIZE_META = config.model_dir / "iv_norm_meta.json"

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


def visualization(train_losses, val_losses, trues, preds):
    param = config.output_params

    # --- 绘制损失曲线 ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color='tab:blue', linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color='tab:orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.plot_dir / "loss_curve.png", dpi=300)
    print("Saved training curve: loss_curve.png")
    plt.close()

    # --- 绘制预测 vs 真值散点图 ---
    plt.figure(figsize=(12, 4))
    for i in range(len(param)):
        plt.subplot(1, len(param), i + 1)
        plt.scatter(trues[:, i], preds[:, i], s=30, alpha=0.7, color='tab:blue', edgecolors='k')
        # 对角线
        min_val = min(trues[:, i].min(), preds[:, i].min())
        max_val = max(trues[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel("True", fontsize=11)
        plt.ylabel("Pred", fontsize=11)
        plt.title(f"{param[i]}", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        # 设置横轴不挤，自动调整
        plt.xticks(rotation=30)
        plt.tight_layout()

    plt.savefig(config.plot_dir / "pred_vs_true.png", dpi=300)
    print("Saved: pred_vs_true.png")
    plt.close()


def main():
    data = np.load("data/processed/converted_dataset.npz")
    iv, params = data["ivcv"], data["params"]

    dataset = BSIMIVDataset(iv, params, save_meta_path=NORMALIZE_META)

    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    train_set, val_set= random_split(dataset, [n_train, n_val])
    print(f"Dataset split: train={n_train}, val={n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ParamExtractorIVNet(input_dim=config.input_dim, hidden_layers=config.mlp_layers,
                                output_dim=config.output_dim, dropout=config.dropout_rate).to(DEVICE)
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
        print(f"Epoch {epoch:03d} | Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({"model": model.state_dict(), "norm_meta": dataset.norm_meta}, MODEL_SAVE)
            print(f"保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break
    print("Training done, best val loss =", best_loss)

    visualization(train_losses, val_losses, trues, preds)
if __name__ == "__main__":
    main()
