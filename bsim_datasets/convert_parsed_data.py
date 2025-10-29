# datasets/convert_parsed_data.py
import numpy as np
import os

def convert(features_path='data/processed/features.npy',
            labels_path='data/processed/labels.npy',
            out_path='data/processed/converted_dataset.npz'):
    """
    将 data_parser.py 输出的 features.npy 和 labels.npy
    转换为 (ivcv, params) 格式以便神经网络训练。
    """
    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"加载完成: features {features.shape}, labels {labels.shape}")

    # 检查特征数量一致
    assert features.shape[0] == labels.shape[0], "样本数量不一致"

    # 重命名为符合旧结构的字段
    ivcv = features.astype(np.float32)
    params = labels.astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, ivcv=ivcv, params=params)
    print(f"✅ 已保存到 {out_path}")

if __name__ == "__main__":
    convert()
