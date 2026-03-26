# data_parser.py
import os
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

if __package__ in (None, ""):
    from config import config
else:
    from .config import config


VALUE_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:meg|[fpnumkaxgt])?)"


def parse_value(value_str: str) -> float:
    """
    支持 HSPICE 所有的工程单位后缀，包括 meg
    """
    value_str = value_str.strip().lower()
    suffixes = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'x': 1e6, 'meg': 1e6, 'g': 1e9, 't': 1e12,
        'a': 1e-18, 'f': 1e-15
    }

    # 特殊处理 meg (3字符)
    if value_str.endswith('meg'):
        try:
            return float(value_str[:-3]) * 1e6
        except ValueError:
            return 0.0

    try:
        suffix = value_str[-1]
        if suffix in suffixes:
            return float(value_str[:-1]) * suffixes[suffix]
        return float(value_str)
    except ValueError:
        return 0.0


class HspiceLisParser:
    def __init__(self, output_params_list):
        # 完整的参数映射池：HSPICE内部变量名 -> 你的模型标准名
        # 以后想多提参数，只需在此池中添加一行映射，无需修改下方逻辑
        self.full_param_pool = {
            'vth0_stamos_val': 'VTH0', 'voff_stamos_val': 'VOFF', 'nfactor_stamos_val': 'NFACTOR',
            'k1_stamos_val': 'K1', 'k2_stamos_val': 'K2', 'u0_stamos_val': 'U0', 'ua_stamos_val': 'UA',
            'ub_stamos_val': 'UB', 'uc_stamos_val': 'UC', 'rdsw_stamos_val': 'RDSW','ags_stamos_val': 'AGS',
            'a0_stamos_val': 'A0', 'keta_stamos_val': 'KETA',
        }
        self.output_order = output_params_list

        self.re_mc_block = re.compile(
            r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
            re.DOTALL
        )

        # 优化后的 XY 块匹配正则
        self.re_xy_block = re.compile(
            r"x\s*\n\s*volt\s+param[\s\S]*?\n([\s\S]*?)\s*y",
            re.IGNORECASE
        )

    def parse(self, lis_content: str):
        features_list = []
        labels_list = []
        vd_biases = config.vd_values

        mc_blocks = self.re_mc_block.findall(lis_content)
        if not mc_blocks:
            print("❌ 错误: 未找到 Monte Carlo 块。")
            return None, None

        print(f"🔍 找到 {len(mc_blocks)} 个样本。目标参数: {self.output_order}")

        for index, block_content in tqdm(mc_blocks, desc="解析 .lis"):
            # --- 1. 查找并拼接分段表格 ---
            xy_matches = self.re_xy_block.findall(block_content)

            # 使用字典存储电流曲线，方便按 Vd 顺序排列
            # 即使 .lis 里分段顺序乱了也能自动对应
            vd_id_map = {}
            full_voltages = []

            # 获取当前块内所有的 i_d_xx 表头
            re_vd_header = re.compile(r"i_d_([\d.]+)", re.IGNORECASE)

            for xy_text in xy_matches:
                lines = xy_text.strip().split('\n')
                if not lines: 
                    continue

                # 从第一行提取当前段包含哪些 Vd
                local_vds = [float(v) for v in re_vd_header.findall(lines[0])]

                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    try:
                        v = parse_value(parts[0])
                        if v not in full_voltages:
                            full_voltages.append(v)

                        for i, id_val_str in enumerate(parts[1:]):
                            if i >= len(local_vds):
                                break
                            vd = local_vds[i]
                            if vd not in vd_id_map:
                                vd_id_map[vd] = []
                            vd_id_map[vd].append(parse_value(id_val_str))
                    except (IndexError, ValueError):
                        continue

            # --- 2. 构造特征 [Vg_vec, Vd_vec, Id_vec] ---
            # 这种向量化排列与 Dataset 类中的 Log 变换逻辑完美契合
            combined_features = []
            pts = config.vg_points

            success_construct = True
            for vd_bias in vd_biases:
                if vd_bias not in vd_id_map:
                    success_construct = False
                    break

                id_curve = vd_id_map[vd_bias]
                if len(id_curve) < pts:
                    success_construct = False
                    break

                # combined_features.extend(full_voltages[:pts])  # Vg 段
                # combined_features.extend([vd_bias] * pts)  # Vd 段
                combined_features.extend(id_curve[:pts])  # Id 段

            if not success_construct or len(combined_features) != config.input_dim:
                continue

            # --- 3. 动态参数提取 (解耦核心) ---
            current_label_dict = {}
            for lis_name, bsim_name in self.full_param_pool.items():
                # 只搜索当前 MC 块
                reg = re.compile(rf"{re.escape(lis_name)}=\s*{VALUE_PATTERN}", re.IGNORECASE)
                match = reg.search(block_content)
                if match:
                    current_label_dict[bsim_name] = parse_value(match.group(1))

            # 根据 config.output_params 的顺序构建向量
            label_vector = []
            for target_p in self.output_order:
                val = current_label_dict.get(target_p, 0.0)  # 没找到则补0
                label_vector.append(val)

            features_list.append(combined_features)
            labels_list.append(label_vector)

        return np.array(features_list), np.array(labels_list)


def main(lis_file_path: Path, output_dir: Path):
    print(f"📄 开始解析 .lis 文件: {lis_file_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        content = lis_file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = lis_file_path.read_text(encoding='latin1')

    parser = HspiceLisParser(output_params_list=config.output_params)
    features, labels = parser.parse(content)

    if features is not None:
        np.save(output_dir / 'features.npy', features)
        np.save(output_dir / 'labels.npy', labels)
        print(f"\n✓ 数据已保存: {output_dir}")

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

    if features.ndim == 3:
        features_transposed = np.transpose(features, (0, 2, 1))
        ivcv = features_transposed.reshape(features_transposed.shape[0], -1).astype(np.float32)
        print(f"✅ 特征已从 {features.shape} 转换为展平的 MLP 格式: {ivcv.shape}")
    else:
        ivcv = features.astype(np.float32)
        print(f"⚠️ 特征已经是 2D 格式: {ivcv.shape}，跳过展平。")

    params = labels.astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, ivcv=ivcv, params=params)
    print(f"✅ 已保存到 {out_path}")

if __name__ == "__main__":
    L_FILE_PATH = Path("bsim_datasets/mc.lis")  # 确认路径
    NPY_OUTPUT_DIR = Path("data/processed")
    main(L_FILE_PATH, NPY_OUTPUT_DIR)
    convert()
