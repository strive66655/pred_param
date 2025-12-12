# train/train_iv_extractor.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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

LOSS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0, 1.0]).to(DEVICE)


def weighted_mse_loss(input, target, weights):
    """
    加权均方误差损失函数
    :param input: 模型预测值 [batch, num_params]
    :param target: 真实标签 [batch, num_params]
    :param weights: 权重向量 [num_params]
    """
    # (input - target)^2
    pct_var = (input - target) ** 2
    # 乘以权重 (自动广播)
    out = pct_var * weights.expand_as(target)
    # 返回平均 Loss
    return out.mean()


def r2_score_np(y_true, y_pred):
    """
    计算 R^2 score for numpy arrays。
    针对常数标签 (ss_tot=0) 增加了鲁棒性检查。
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # 显式处理常数标签
    if ss_tot < 1e-12:
        if ss_res < 1e-12:
            return 1.0  # 完美预测
        else:
            return 0.0  # 常数标签，但预测失败

    return 1 - (ss_res / ss_tot)


def mae_np(y_true, y_pred):
    """计算 Mean Absolute Error (MAE) for numpy arrays (返回标量)."""
    return np.mean(np.abs(y_true - y_pred))


def rmse_np(y_true, y_pred):
    """计算 Root Mean Square Error (RMSE) for numpy arrays (返回标量)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        iv = batch["iv"].to(DEVICE)
        params = batch["params"].to(DEVICE)
        pred = model(iv)

        # 使用自定义加权 Loss
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
        for batch in tqdm(loader, desc="Evaluation", leave=False):
            iv = batch["iv"].to(DEVICE)
            params = batch["params"].to(DEVICE)
            pred = model(iv)

            # 使用自定义加权 Loss
            loss = loss_fn(pred, params)

            total += loss.item() * iv.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(params.cpu().numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    return total / len(loader.dataset), preds, trues


def visualization(train_losses, val_losses, trues, preds, r2_scores):
    param = config.output_params

    # --- 绘制损失曲线 ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss (Weighted)", color='tab:blue', linewidth=2)
    plt.plot(val_losses, label="Validation Loss (Weighted)", color='tab:orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Weighted MSE Loss", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    loss_path = config.plot_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    print(f"✓ 训练损失曲线图已保存: {loss_path.name}")
    plt.close()

    num_params = len(param)
    # 使用 2 行 3 列的布局 (如果 num_params >= 4)
    if num_params > 3:
        rows = 2
        cols = int(np.ceil(num_params / rows))
        fig_width = 4 * cols
        fig_height = 4 * rows
    else:
        rows = 1
        cols = num_params
        fig_width = 5 * cols
        fig_height = 5

    plt.figure(figsize=(fig_width, fig_height))

    for i in range(num_params):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(trues[:, i], preds[:, i], s=30, alpha=0.7, color='tab:blue', edgecolors='k')
        # 对角线
        min_val = min(trues[:, i].min(), preds[:, i].min())
        max_val = max(trues[:, i].max(), preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel("True", fontsize=11)
        plt.ylabel("Pred", fontsize=11)

        r2_text = f" (R²: {r2_scores[i]:.4f})"
        plt.title(f"{param[i]}{r2_text}", fontsize=12)

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=30)

    plt.tight_layout()
    pred_path = config.plot_dir / "pred_vs_true.png"
    plt.savefig(pred_path, dpi=300)
    print(f"✓ 预测对比图已保存: {pred_path.name}")
    plt.close()


def main():
    # 打印设备和参数信息
    print("Device:", DEVICE)
    print("Model Parameters:", config.output_params)
    # print("Loss Weights:", LOSS_WEIGHTS)  # 打印权重确认
    print("-" * 50)

    data = np.load("data/processed/converted_dataset.npz")
    iv, params = data["ivcv"], data["params"]

    dataset = BSIMIVDataset(iv, params, save_meta_path=NORMALIZE_META)

    if 'pca_n_components' in dataset.norm_meta:
        # 使用 PCA 后的维度
        INPUT_DIM = dataset.norm_meta['pca_n_components']
    else:
        # 如果没有 PCA meta (例如：在 config.py 中没有开启 PCA 或者数据集未加载 PCA)
        # 使用原始维度
        INPUT_DIM = dataset[0]["iv"].numel()
    print(f"模型输入维度设置为: {INPUT_DIM}")

    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    print(f"Dataset split: train={n_train}, val={n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ParamExtractorIVNet(input_dim=INPUT_DIM, hidden_layers=config.mlp_layers,
                                output_dim=config.output_dim, dropout=config.dropout_rate).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=config.weight_decay)

    if hasattr(config, 'scheduler') and config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    else:
        scheduler = None

    loss_fn = nn.MSELoss() # 不再使用标准 MSE
    # print("Using Custom Weighted MSE Loss.")
    # loss_fn = lambda pred, params: weighted_mse_loss(pred, params, LOSS_WEIGHTS)

    best_loss = 1e9
    patience = 0
    train_losses, val_losses = [], []
    best_preds_norm = None
    best_trues_norm = None

    for epoch in range(NUM_EPOCHS):
        # 传入权重
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        val_loss, preds_norm, trues_norm = eval_model(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f} | LR={current_lr:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({"model": model.state_dict(), "norm_meta": dataset.norm_meta}, MODEL_SAVE)
            best_preds_norm = preds_norm
            best_trues_norm = trues_norm
            print(f"保存最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break
    print("Training done, best val loss =", best_loss)

    if best_preds_norm is None:
        print("警告: 未能找到最佳模型，使用最后一次 epoch 的结果进行报告。")
        best_preds_norm = preds_norm
        best_trues_norm = trues_norm

    val_trues_final = dataset.inverse_transform_params(best_trues_norm)
    val_preds_final = dataset.inverse_transform_params(best_preds_norm)

    r2_scores = []

    print("-" * 50)
    print("最佳模型性能报告 (验证集)")
    print("-" * 50)

    param_names = config.output_params

    for i, name in enumerate(param_names):
        y_true = val_trues_final[:, i]
        y_pred = val_preds_final[:, i]

        r2 = r2_score_np(y_true, y_pred)
        mae = mae_np(y_true, y_pred)
        rmse = rmse_np(y_true, y_pred)

        r2_scores.append(r2)
        std_dev = np.std(y_true)

        print(f"{name}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE:      {mae:.4e}")
        print(f"  RMSE:     {rmse:.4e}")
        print(f"  True Std: {std_dev:.4e}")
        print("")

    print("-" * 50)

    visualization(train_losses, val_losses, val_trues_final, val_preds_final, r2_scores)

    print("✅ 实验完成!")
    print(f"  结果保存在: {config.output_dir.as_posix()}")
    print("-" * 50)


if __name__ == "__main__":
    main()