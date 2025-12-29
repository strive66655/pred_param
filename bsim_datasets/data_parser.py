# data_parser.py
"""
HSPICE .lis 文件解析器 (修复版)
- 1. 使用通用正则匹配 x...y 数据块。
- 2. 自动拼接分段表格 (Split Tables)。
- 3. 构造 [Vg, Vd, Id] 格式特征。
"""
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from config import config


def parse_value(value_str: str) -> float:
    value_str = value_str.strip()
    suffixes = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'x': 1e6, 'meg': 1e6, 'g': 1e9, 't': 1e12,
        'a': 1e-18, 'f': 1e-15
    }
    try:
        suffix = value_str[-1].lower()
        if suffix in suffixes:
            return float(value_str[:-1]) * suffixes[suffix]
        return float(value_str)
    except:
        return 0.0


class HspiceLisParser:
    def __init__(self, output_params_list):
        self.param_map = {
            'vth0_value': 'VTH0', 'u0_param': 'U0', 'ags_param': 'AGS',
            'vsat_value': 'VSAT', 'ub_value': 'UB', 'voff_value': 'VOFF',
            'nfactor_value': 'NFACTOR', 'a0_value': 'A0', 'ua_value': 'UA'
        }
        self.target_lis_params = list(self.param_map.keys())
        self.output_order = output_params_list

        self.re_mc_block = re.compile(
            r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
            re.DOTALL
        )

        # [通用正则] 只要是 x 开头 y 结尾的块都抓出来，不管是分段的哪一部分
        self.re_xy_block = re.compile(
            r"x\s*\n"
            r"\s*volt\s+param\s*(?:\s+param\s*)*\n"
            # 匹配第二行表头 (m1, m2, m3, ...)
            r"\s*(?:\s+\w+)?\s*(?:\s*i_d_\d+(?:\.\d+)?\s*)*\n"
            r"(.*?)"  # 捕获中间数据
            r"\s*y",  # y 行结尾
            re.DOTALL | re.IGNORECASE
        )

        self.re_params = []
        for param_name in self.target_lis_params:
            self.re_params.append(
                (param_name, re.compile(r"{}=\s*([\d.+-]+[pnumkaxfg]?)".format(param_name), re.IGNORECASE))
            )

    def parse(self, lis_content: str):
        features_list = []
        labels_list = []
        vd_biases = config.vd_values

        mc_blocks = self.re_mc_block.findall(lis_content)
        if not mc_blocks:
            print("❌ 错误: 未找到 Monte Carlo 块。")
            return None, None

        print(f"🔍 找到 {len(mc_blocks)} 个样本。目标: {len(vd_biases)} 条曲线 (Vd={vd_biases})...")

        for index, block_content in tqdm(mc_blocks, desc="解析 .lis"):

            # --- 1. 查找并拼接所有分段表格 ---
            xy_blocks = self.re_xy_block.findall(block_content)

            full_voltages = []
            full_currents_list = []

            for xy_data in xy_blocks:
                lines = xy_data.strip().split('\n')
                data_start_idx = -1
                for i, line in enumerate(lines):
                    # 只有以数字/负号开头的行才是数据
                    if re.match(r"^\s*[\d.-]", line.strip()):
                        data_start_idx = i
                        break

                if data_start_idx == -1: continue

                segment_volts = []
                segment_currs = []

                for line in lines[data_start_idx:]:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            v = parse_value(parts[0])
                            c_vals = [parse_value(x) for x in parts[1:]]
                            segment_volts.append(v)
                            segment_currs.append(c_vals)
                        except:
                            pass

                if not segment_volts: continue

                if not full_voltages:
                    full_voltages = segment_volts

                if segment_currs:
                    cols = list(map(list, zip(*segment_currs)))
                    full_currents_list.extend(cols)

            if len(full_currents_list) != len(vd_biases):
                continue

            # --- 2. 构造 [Vg, Vd, Id] 特征 ---
            combined_features = []
            pts = config.vg_points

            for i in range(len(full_currents_list)):
                current_curve = full_currents_list[i]
                vd_val = vd_biases[i]

                if len(full_voltages) < pts: break

                vg_vec = full_voltages[:pts]
                vd_vec = [vd_val] * pts  # Vd 显式特征
                id_vec = current_curve[:pts]

                combined_features.extend(vg_vec)
                combined_features.extend(vd_vec)
                combined_features.extend(id_vec)

            if len(combined_features) != config.input_dim:
                continue

            features_list.append(combined_features)

            # --- 3. 提取参数 ---
            label_dict_raw = {}
            for param_name, re_c in self.re_params:
                param_match = re_c.search(block_content)
                if param_match:
                    label_dict_raw[param_name] = parse_value(param_match.group(1))

            if not label_dict_raw:
                features_list.pop()
                continue

            label_ordered = []
            for out_param in self.output_order:
                found = False
                for lis_name, bsim_name in self.param_map.items():
                    if bsim_name == out_param:
                        if lis_name in label_dict_raw:
                            label_ordered.append(label_dict_raw[lis_name])
                            found = True
                            break
                if not found: label_ordered.append(0.0)

            labels_list.append(label_ordered)

        if not features_list:
            print("❌ 未提取到数据，请检查文件格式。")
            return None, None

        return np.array(features_list), np.array(labels_list)


def main(lis_file_path: Path, output_dir: Path):
    print(f"📄 开始解析 .lis 文件: {lis_file_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        content = lis_file_path.read_text(encoding='utf-8')
    except:
        content = lis_file_path.read_text(encoding='latin1')

    parser = HspiceLisParser(output_params_list=config.output_params)
    features, labels = parser.parse(content)

    if features is not None:
        np.save(output_dir / 'features.npy', features)
        np.save(output_dir / 'labels.npy', labels)
        print(f"\n✓ 数据已保存: {output_dir}")


if __name__ == "__main__":
    L_FILE_PATH = Path("bsim_datasets/mc.lis")  # 确认路径
    NPY_OUTPUT_DIR = Path("data/processed")
    main(L_FILE_PATH, NPY_OUTPUT_DIR)