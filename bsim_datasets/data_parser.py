# data_parser.py
"""
HSPICE .lis 文件解析器
- 专为解析蒙特卡洛 (mc) .lis 文件而设计。
- 使用正则表达式 (re) 提取每个 index 的 I-V (特征) 和参数 (标签)。
- 将数据保存为 features.npy 和 labels.npy 供模型训练。
"""
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# 导入全局配置
from config import config


def parse_value(value_str: str) -> float:
    """
    将HSPICE的科学计数法 (如 '254.3500m', '93.9859k', '50.7286p') 转换为浮点数
    """
    value_str = value_str.strip()
    suffixes = {
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'x': 1e6,  # 'x' 或 'meg'
        'meg': 1e6,
        'g': 1e9,
        't': 1e12,
    }
    # 检查最后一个字符是否是已知的后缀
    suffix = value_str[-1].lower()
    if suffix in suffixes:
        num_str = value_str[:-1]
        return float(num_str) * suffixes[suffix]
    else:
        # 可能是 'e+' 或 'e-' 格式
        try:
            return float(value_str)
        except ValueError:
            print(f"警告: 无法解析的值 '{value_str}'，返回 0.0")
            return 0.0


class HspiceLisParser:
    """
    (已更新) 解析 mc.lis 文件的主类
    - 支持 i_linear 和 i_sat 两列数据
    """

    def __init__(self, output_params_list):
        self.param_map = {
            'vth0_value': 'VTH0',
            'u0_param': 'U0',
            'ags_param': 'AGS',
            'vsat_value': 'VSAT',
            'ub_value': 'UB',
            'voff_value': 'VOFF',
            'nfactor_value': 'NFACTOR',
            'a0_value': 'A0',
            'ua_value': 'UA'
        }

        self.target_lis_params = list(self.param_map.keys())
        self.output_order = output_params_list

        # 1. 匹配 MC index 块 (不变)
        self.re_mc_block = re.compile(
            r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
            re.DOTALL
        )

        # 2. 匹配 I-V 数据 (x 块) -- 修改后的表头
        self.re_iv_data = re.compile(
            r"x\s*\n"  
            r"\s*volt\s+param\s*(?:\s+param\s*)*\n"
            # 匹配第二行表头 (m1, m2, m3, ...)
            r"\s*(?:\s+\w+)?\s*(?:\s*i_d_\d+(?:\.\d+)?\s*)*\n"
            r"(.*?)"  # 捕获中间数据
            r"\s*y",  # y 行结尾
            re.DOTALL | re.IGNORECASE
        )

        # 3. 匹配参数数据 (y 块)
        self.re_params = []
        for param_name in self.target_lis_params:
            # 匹配形式 param_name= 数值(带单位)
            self.re_params.append(
                (param_name, re.compile(r"{}=\s*([\d.+-]+[pnumk]?)".format(param_name)))
            )

    def parse(self, lis_content: str):
        """
        解析 .lis 文件，提取 Monte Carlo 样本的 I-V 特征和参数标签。
        支持单个 MC 块内包含多个 I-V 数据块。
        """
        features_list = []
        labels_list = []

        mc_blocks = self.re_mc_block.findall(lis_content)
        if not mc_blocks:
            print("❌ 错误: 未在文件中找到任何 '*** monte carlo index = ... ***' 块。")
            return None, None

        print(f"🔍 找到 {len(mc_blocks)} 个 Monte Carlo 样本。开始解析...")

        for index, block_content in tqdm(mc_blocks, desc="解析 .lis 文件"):

            # --- 提取 I-V 数据 (可能多个) ---
            # 使用 findall 找到所有匹配的 I-V 数据块
            iv_data_matches = self.re_iv_data.findall(block_content)

            if not iv_data_matches:
                print(f"⚠️ 警告: Index {index} 中未找到任何 I-V 数据块，跳过...")
                continue

            # 用于存储当前 MC 样本的所有特征
            mc_voltages = []
            mc_currents_list = []  # 存储所有曲线的电流值

            # 遍历所有的 I-V 数据块
            for iv_data_str in iv_data_matches:

                iv_data_str = iv_data_str.strip()
                voltages = []
                currents_list_in_block = []  # 当前块中的电流曲线

                # --- 解析 I-V 数据行 ---
                for line in iv_data_str.split('\n'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # 假设第一列始终是电压
                            voltages.append(parse_value(parts[0]))

                            # 动态扩展电流曲线列表
                            while len(currents_list_in_block) < len(parts) - 1:
                                currents_list_in_block.append([])

                            for i, val in enumerate(parts[1:]):
                                currents_list_in_block[i].append(parse_value(val))
                        except Exception as e:
                            print(f"⚠️ Index {index} 解析行失败: {line} -> {e}")

                # 将当前块的I-V数据加入到 MC 样本的总列表
                # 注意：如果电压在每个块中都是相同的，可以选择只存储一次
                # 为了特征向量统一，我们将所有电压和电流都平铺
                mc_voltages.extend(voltages)
                for curve in currents_list_in_block:
                    mc_currents_list.append(curve)

            if not mc_voltages or not mc_currents_list:
                print(f"⚠️ Index {index} 未提取到有效 I-V 数据，跳过...")
                continue

            combined_features = mc_voltages[:config.vg_points]  # 仅取第一个曲线的电压作为共享特征
            # 检查总电流点数是否符合预期
            if len(mc_currents_list) != config.num_curves:
                print(
                    f"❌ Index {index} 提取的曲线数 ({len(mc_currents_list)}) 与 config.num_curves ({config.num_curves}) 不符，跳过...")
                continue

            for curve in mc_currents_list:
                combined_features.extend(curve)

            features_list.append(combined_features)

            # --- 提取参数数据 (只提取一次) ---
            label_dict_raw = {}
            for param_name, re_c in self.re_params:
                param_match = re_c.search(block_content)
                if param_match:
                    # 注意：如果参数可能在多个地方重复，这里只会取第一次匹配到的
                    label_dict_raw[param_name] = parse_value(param_match.group(1))

            if not label_dict_raw:
                print(f"⚠️ Index {index} 中未找到任何参数，移除特征并跳过...")
                features_list.pop()
                continue

            # --- 按 output_order 排序标签 ---
            label_ordered = []
            for out_param in self.output_order:
                found = False
                for lis_name, bsim_name in self.param_map.items():
                    if bsim_name == out_param:
                        if lis_name in label_dict_raw:
                            label_ordered.append(label_dict_raw[lis_name])
                            found = True
                            break
                if not found:
                    # 如果参数不在 lis 文件中，或者未在 param_map 中定义，填充 0.0
                    label_ordered.append(0.0)

            labels_list.append(label_ordered)

            # --- 最终检查 ---
            if len(combined_features) != config.input_dim:
                print(
                    f"❌ Index {index} 特征维度 ({len(combined_features)}) 与 config.input_dim ({config.input_dim}) 不匹配，移除特征和标签并跳过...")
                if features_list: features_list.pop()
                if labels_list: labels_list.pop()
                continue

        if not features_list or not labels_list:
            print("❌ 错误: 解析完成，但未提取到任何有效数据。")
            return None, None

        print(f"\n✓ 解析成功! 提取了 {len(features_list)} 组数据。")

        features_np = np.array(features_list)
        labels_np = np.array(labels_list)

        print(f"  特征 (X) 形状: {features_np.shape}")
        print(f"  标签 (Y) 形状: {labels_np.shape}")

        return features_np, labels_np

def main(lis_file_path: Path, output_dir: Path):
    """
    主函数：读取 .lis, 解析, 保存 .npy
    """
    print(f"📄 开始解析 .lis 文件: {lis_file_path}")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        content = lis_file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # 如果 utf-8 失败，尝试 latin1
        print("⚠️ UTF-8 读取失败，尝试使用 latin1 编码...")
        content = lis_file_path.read_text(encoding='latin1')
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 {lis_file_path}")
        return
    except Exception as e:
        print(f"❌ 错误: 读取文件时出错: {e}")
        return

    # 初始化解析器
    # 我们从 config.py 传入期望的参数列表
    parser = HspiceLisParser(output_params_list=config.output_params)
    features, labels = parser.parse(content)

    if features is not None and labels is not None:
        # 保存 .npy 文件
        feature_path = output_dir / 'features.npy'
        label_path = output_dir / 'labels.npy'

        np.save(feature_path, features)
        np.save(label_path, labels)

        print(f"\n✓ 数据已保存:")
        print(f"  特征 -> {feature_path}")
        print(f"  标签 -> {label_path}")


if __name__ == "__main__":
    # --- 如何运行 ---
    # 1. 把你的 mc.lis 文件放到一个地方, 例如 'data/' 目录
    # 2. 在下面设置路径
    # 3. 直接运行 `python data_parser.py`

    L_FILE_PATH = Path("bsim_datasets/mc.lis")  # <--- 修改这里: 你的.lis文件路径
    NPY_OUTPUT_DIR = Path("data/processed")  # <--- 修改这里: .npy的保存路径

    main(L_FILE_PATH, NPY_OUTPUT_DIR)