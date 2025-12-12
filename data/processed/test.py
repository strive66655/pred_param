import numpy as np
import pandas as pd
from pathlib import Path

# --- 配置 (与 config.py 保持一致) ---
NUM_CURVES = 10
VG_POINTS = 21
INPUT_DIM = VG_POINTS + (NUM_CURVES * VG_POINTS)  # 231
OUTPUT_PARAMS = ['VTH0', 'U0', 'AGS', 'ETA0', 'LU0', 'VSAT']
N_SAMPLES_TO_SHOW = 5  # 显示前 5 个样本

# --- 路径 ---
FEATURES_PATH = Path("data/processed/features.npy")
LABELS_PATH = Path("data/processed/labels.npy")
OUTPUT_CSV_PATH = Path("data/processed/extracted_data_check.csv")


def create_verification_csv():
    """加载 NumPy 数据并创建 CSV 文件用于检查"""
    try:
        # 1. 加载数据
        X = np.load(FEATURES_PATH)
        Y = np.load(LABELS_PATH)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 {FEATURES_PATH.name} 或 {LABELS_PATH.name}。请确保 data_parser.py 已成功运行。")
        return

    print(f"成功加载 {X.shape[0]} 个样本。")

    # 2. 选择前 N 个样本
    X_sample = X[:N_SAMPLES_TO_SHOW]
    Y_sample = Y[:N_SAMPLES_TO_SHOW]

    # 3. 构建列标题
    headers = []
    # 电压列 (V_P1 to V_P21)
    headers.extend([f"V_P{i + 1}" for i in range(VG_POINTS)])

    # 电流列 (I_C1_P1 to I_C10_P21)
    for c in range(NUM_CURVES):
        headers.extend([f"I_C{c + 1}_P{p + 1}" for p in range(VG_POINTS)])

    # 参数列 (VTH0, U0, ...)
    headers.extend(OUTPUT_PARAMS)

    # 4. 合并特征和标签
    X_sample_flat = X_sample.reshape(N_SAMPLES_TO_SHOW, INPUT_DIM)
    data_to_combine = np.hstack([X_sample_flat, Y_sample])

    # 5. 创建 DataFrame 并保存
    df = pd.DataFrame(data_to_combine, columns=headers)

    # 确保输出目录存在
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n✅ 成功将前 {N_SAMPLES_TO_SHOW} 个样本的数据保存到:")
    print(f"文件路径: {OUTPUT_CSV_PATH}")
    print(f"数据形状 (样本数, 总列数): {df.shape}")
    print("\n请检查 CSV 文件中的数据，确保 I-V 数据和参数正确对齐。")


if __name__ == '__main__':
    create_verification_csv()