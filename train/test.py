# train/train_stepwise.py
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

# 导入原有模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bsim_datasets.config import config
from bsim_datasets.bsim_iv_dataset import BSIMIVDataset
from models.param_extractor_iv import ParamExtractorIVNet

# ===== 全局配置 =====
DEVICE = config.device
LR = config.learning_rate
BATCH_SIZE = config.batch_size
TOTAL_EPOCHS = config.epochs  # 总 epoch 数
PHASE_1_RATIO = 0.4  # 第一阶段占总 Epoch 的比例 (例如前 40% 训练 VTH0/U0)
MODEL_SAVE = config.model_dir / "best_iv_extractor_stepwise.pth"
NORMALIZE_META = config.model_dir / "iv_norm_meta.json"


# ⭐ 核心修改: 加权 MSE Loss
def weighted_mse_loss(pred, target, weights):
    """
    pred: (Batch, 3)
    target: (Batch, 3)
    weights: (3,)  例如 [1.0, 1.0, 0.0]
    """
    # 保持维度一致以便广播
    w = weights.view(1, -1).expand_as(pred)
    # 计算平方差
    loss = w * (pred - target) ** 2
    # 返回平均值
    return loss.mean()


def train_one_epoch(model, loader, opt, epoch_weights):
    model.train()
    total_loss = 0
    for batch in loader:
        iv = batch["iv"].to(DEVICE)
        params = batch["params"].to(DEVICE)

        pred = model(iv)

        # 使用加权 Loss
        loss = weighted_mse_loss(pred, params, epoch_weights)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * iv.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, epoch_weights):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["iv"].to(DEVICE)
            params = batch["params"].to(DEVICE)
            pred = model(iv)

            loss = weighted_mse_loss(pred, params, epoch_weights)

            total_loss += loss.item() * iv.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(params.cpu().numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    return total_loss / len(loader.dataset), preds, trues


def visualization(trues, preds, phase_name):
    """画图函数"""
    param_names = config.output_params
    plt.figure(figsize=(12, 4))
    for i in range(len(param_names)):
        plt.subplot(1, len(param_names), i + 1)
        # 绘制散点
        plt.scatter(trues[:, i], preds[:, i], s=10, alpha=0.5, label='Pred')
        # 绘制对角线
        min_v, max_v = trues[:, i].min(), trues[:, i].max()
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal')
        plt.title(f"{param_names[i]}")
        plt.xlabel("True (Z-score)")
        plt.ylabel("Pred (Z-score)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.plot_dir / f"pred_vs_true_{phase_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 {phase_name} 阶段可视化已保存: {save_path}")
    plt.close()


def main():
    # 1. 加载数据
    data_path = "data/processed/converted_dataset.npz"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return

    data = np.load(data_path)
    # 确保载入 float32
    iv, params = data["ivcv"].astype(np.float32), data["params"].astype(np.float32)

    # 2. 创建 Dataset (注意：这里会自动应用 Log 变换，如果您更新了 dataset 代码)
    dataset = BSIMIVDataset(iv, params, save_meta_path=NORMALIZE_META)

    # 3. 划分数据集
    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset: Train={len(train_set)}, Val={len(val_set)}")
    print(f"Params Order: {config.output_params}")  # ['VTH0', 'U0', 'AGS']

    # 4. 初始化模型
    model = ParamExtractorIVNet(
        input_dim=config.input_dim,
        hidden_layers=config.mlp_layers,
        output_dim=config.output_dim,
        dropout=config.dropout_rate
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 定义分步训练的权重
    # 假设参数顺序是 [VTH0, U0, AGS]
    # Phase 1: 只看前两个
    weights_phase1 = torch.tensor([1.0, 1.0, 0.0]).to(DEVICE)
    # Phase 2: 重点看第三个 (AGS)，稍微兼顾前两个防止遗忘
    weights_phase2 = torch.tensor([0.1, 0.1, 10.0]).to(DEVICE)

    phase_switch_epoch = int(TOTAL_EPOCHS * PHASE_1_RATIO)
    best_loss = 1e9

    print(f"\n🚀 开始分步训练 (Stepwise Training)")
    print(f"Phase 1 (Epoch 0-{phase_switch_epoch}): 专注 VTH0, U0")
    print(f"Phase 2 (Epoch {phase_switch_epoch}-{TOTAL_EPOCHS}): 攻克 AGS\n")

    for epoch in range(TOTAL_EPOCHS):
        # --- 动态调整权重 ---
        if epoch < phase_switch_epoch:
            current_weights = weights_phase1
            phase = "Phase1"
        else:
            current_weights = weights_phase2
            phase = "Phase2"

        # --- 训练与评估 ---
        train_loss = train_one_epoch(model, train_loader, opt, current_weights)
        val_loss, preds, trues = eval_model(model, val_loader, current_weights)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} [{phase}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # --- 保存最佳模型 (仅基于当前阶段的 Loss) ---
        # 注意：Phase 1 和 Phase 2 的 Loss 数量级不同，切换阶段时 best_loss 会剧烈波动，这是正常的
        if epoch == phase_switch_epoch:
            print(f"⚠️ 切换到阶段 2，重置 Best Loss 基准...")
            best_loss = 1e9  # 重置 best loss 以适应新的权重量级

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "norm_meta": dataset.norm_meta}, MODEL_SAVE)
            # print(f"  Best model saved.")

    print("\n✅ 训练完成。")

    # 最后做一次可视化评估
    _, preds, trues = eval_model(model, val_loader, weights_phase2)  # 用最后的权重评估
    visualization(trues, preds, "Final_Stepwise")


if __name__ == "__main__":
    main()