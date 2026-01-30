# data_parser.py
"""
HSPICE .lis 文件解析器 (分段表格修复版)
- 1. 支持多段 x...y 数据块自动拼接。
- 2. 自动对齐 Vg 和多列 Vd 数据。
- 3. 构造 [Vg, Vd, Id] 格式特征。
"""
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from config import config


def parse_value(value_str: str) -> float:
    """解析 HSPICE 科学计数法后缀 (如 254.35m, 1.2k, 50meg, 1.9a)"""
    value_str = value_str.strip().lower()
    suffixes = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'x': 1e6, 'meg': 1e6, 'g': 1e9, 't': 1e12,
        'a': 1e-18, 'f': 1e-15
    }
    if value_str.endswith('meg'):
        return float(value_str[:-3]) * 1e6

    last_char = value_str[-1]
    if last_char in suffixes:
        try:
            return float(value_str[:-1]) * suffixes[last_char]
        except:
            return 0.0
    try:
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
        self.output_order = output_params_list

        # --- 正则表达式 ---
        self.re_mc_block = re.compile(r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
                                      re.DOTALL)

        # 匹配块内所有的 x...y 结构
        self.re_x_y_block = re.compile(r"x\s*\n\s*volt\s+param[\s\S]*?\n([\s\S]*?)\s*y", re.IGNORECASE)

        # 提取表头中的 Vd 值 (用于识别当前分段属于哪些 Vd)
        self.re_vd_header = re.compile(r"i_d_([\d.]+)", re.IGNORECASE)

        self.re_params = []
        for lis_name, bsim_name in self.param_map.items():
            if bsim_name in self.output_order:
                reg = re.compile(r"{}=\s*([\d.+-]+[pnumk]?)".format(lis_name))
                self.re_params.append((bsim_name, reg))

    def parse(self, content):
        # 1. 预扫描：获取全局所有不重复的 Vd 数值并排序
        all_vd_matches = self.re_vd_header.findall(content[:50000])
        all_vds = sorted(list(set([float(v) for v in all_vd_matches])))

        print(f"🔍 全局检测到 {len(all_vds)} 个漏压点 (Vd): {all_vds}")
        if not all_vds:
            print("❌ 错误: 无法提取 Vd 数值")
            return None, None

        # 2. 提取 MC 块
        mc_blocks = self.re_mc_block.findall(content)
        if not mc_blocks:
            print("❌ 错误: 未找到 Monte Carlo 数据块")
            return None, None

        features_list = []
        labels_list = []

        for index, block_content in tqdm(mc_blocks, desc="解析进度"):
            # 使用字典存储：{Vg: {Vd1: Id1, Vd2: Id2, ...}} 确保分段数据能正确合入同一 Vg
            sample_data_map = {}

            # 查找块内所有的 x...y 区域
            xy_matches = self.re_x_y_block.findall(block_content)

            for xy_text in xy_matches:
                lines = xy_text.strip().split('\n')
                if not lines: continue

                # 第一行通常是包含 i_d_x.x 的表头行
                header_line = lines[0]
                local_vds = [float(v) for v in self.re_vd_header.findall(header_line)]

                # 从第二行开始是数值
                for data_line in lines[1:]:
                    parts = data_line.split()
                    if len(parts) < 2: continue

                    try:
                        vg = parse_value(parts[0])
                        currents = [parse_value(p) for p in parts[1:]]

                        if vg not in sample_data_map:
                            sample_data_map[vg] = {}

                        # 将当前段的电流填入对应的 Vd 位置
                        for i, id_val in enumerate(currents):
                            if i < len(local_vds):
                                sample_data_map[vg][local_vds[i]] = id_val
                    except:
                        continue

            # 按照 Vg 排序并展平为 [Vg, Vd, Id] 序列
            flattened_feature = []
            sorted_vgs = sorted(sample_data_map.keys())

            for vg in sorted_vgs:
                for vd in all_vds:
                    id_val = sample_data_map[vg].get(vd, 0.0)  # 缺失点补 0
                    flattened_feature.extend([vg, vd, id_val])

            if not flattened_feature:
                continue

            # B. 提取参数标签
            label_dict = {}
            for bsim_name, reg in self.re_params:
                m = reg.search(block_content)
                label_dict[bsim_name] = parse_value(m.group(1)) if m else 0.0

            ordered_label = [label_dict.get(p, 0.0) for p in self.output_order]

            features_list.append(flattened_feature)
            labels_list.append(ordered_label)

        return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)


def main():
    lis_path = Path(config.INPUT_LIS)
    out_path = Path(config.OUTPUT_NPZ)

    if not lis_path.exists():
        print(f"❌ 找不到输入文件: {lis_path}")
        return

    print(f"📄 正在读取: {lis_path}")
    try:
        content = lis_path.read_text(encoding='utf-8')
    except:
        content = lis_path.read_text(encoding='latin1')

    parser = HspiceLisParser(config.output_params)
    ivcv, params = parser.parse(content)

    if ivcv is not None and ivcv.shape[1] > 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, ivcv=ivcv, params=params)

        print(f"\n✅ 转换成功!")
        print(f"📂 保存路径: {out_path}")
        print(f"📊 特征形状: {ivcv.shape} (每行长度应为 Vg步数 * Vd列数 * 3)")
        print(f"🏷️ 标签形状: {params.shape}")

        print("\n--- 数据对齐预览 (第一个样本) ---")
        # 打印前两组 [Vg, Vd, Id] 验证 Vd 是否递增
        print(f"组1 [Vg, Vd, Id]: {ivcv[0][0:3]}")
        print(f"组2 [Vg, Vd, Id]: {ivcv[0][3:6]}")
    else:
        print("\n❌ 错误: 未提取到有效数据，请检查正则。")


if __name__ == "__main__":
    main()