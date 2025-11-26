from pathlib import Path
import numpy as np
from tqdm import tqdm

from data_parser import HspiceLisParser  # 确保可以正确导入
from config import config


def process_folder_merged(lis_folder_path: Path, output_dir: Path, recursive=True):
    """
    批量解析文件夹下所有 .lis 文件，将所有数据横向拼接
    - .lis 文件中的第一列只保留第一个文件，后续文件不重复
    - 样本数与标签保持一致
    :param lis_folder_path: 待解析的 .lis 文件夹路径
    :param output_dir: 保存 .npy 的目录
    :param recursive: 是否递归子目录
    """
    if not lis_folder_path.exists() or not lis_folder_path.is_dir():
        print(f"❌ 错误: 文件夹不存在或不是目录: {lis_folder_path}")
        return

    lis_files = list(lis_folder_path.rglob("*.lis") if recursive else lis_folder_path.glob("*.lis"))
    if not lis_files:
        print(f"❌ 错误: 未找到任何 .lis 文件: {lis_folder_path}")
        return

    print(f"🔍 找到 {len(lis_files)} 个 .lis 文件，开始解析并横向合并...")

    parser = HspiceLisParser(output_params_list=config.output_params)

    feature_list = []
    label_ref = None
    n_samples = None

    for idx, lis_file in enumerate(tqdm(sorted(lis_files), desc="解析 .lis 文件")):
        try:
            content = lis_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = lis_file.read_text(encoding='latin1')
        except Exception as e:
            print(f"❌ 读取文件 {lis_file.name} 失败: {e}")
            continue

        features, labels = parser.parse(content)
        if features is None or labels is None:
            print(f"⚠️ 文件 {lis_file.name} 未提取到有效数据，跳过。")
            continue

        # 样本数一致性检查
        if label_ref is None:
            label_ref = labels
            n_samples = features.shape[0]
        else:
            if features.shape[0] != n_samples:
                print(f"⚠️ 样本数不一致 ({features.shape[0]} vs {n_samples})，跳过 {lis_file.name}")
                continue
            # 标签一致性检查
            if not np.allclose(labels, label_ref):
                print(f"⚠️ 标签与参考标签不一致，跳过 {lis_file.name}")
                continue

        # ⚡ 保留第一个文件的第一列，其余文件去掉第一列
        if idx == 0:
            feature_list.append(features)
        else:
            if features.shape[1] > 1:
                feature_list.append(features[:, config.vg_points:])
            else:
                print(f"⚠️ 文件 {lis_file.name} 只有一列特征，跳过。")
                continue

    if not feature_list or label_ref is None:
        print("❌ 未提取到任何有效数据。")
        return

    # 横向拼接
    all_features_np = np.hstack(feature_list)
    all_labels_np = label_ref

    # 保存输出
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'features.npy', all_features_np)
    np.save(output_dir / 'labels.npy', all_labels_np)

    print(f"\n✅ 所有文件解析完成并横向合并保存到 {output_dir}")
    print(f"  特征 (X) 形状: {all_features_np.shape}")
    print(f"  标签 (Y) 形状: {all_labels_np.shape}")
    print(f"  合并了 {len(feature_list)} 个文件，每个文件特征维度分别为 {[f.shape[1] for f in feature_list]}")


if __name__ == "__main__":
    LIS_FOLDER_PATH = Path(r"F:\pred_param\bsim_datasets\-")  # 待解析的 .lis 文件夹
    NPY_OUTPUT_DIR = Path(r"F:\pred_param\data\processed")   # 输出保存目录

    process_folder_merged(LIS_FOLDER_PATH, NPY_OUTPUT_DIR, recursive=True)
