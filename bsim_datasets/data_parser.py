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


# data_parser.py (只替换这个类)

class HspiceLisParser:
    """
    (已更新) 解析 mc.lis 文件的主类
    - 支持 i_linear 和 i_sat 两列数据
    """

    def __init__(self, output_params_list):
        self.param_map = {
            'vth0_value': 'VTH0',
            'u0_param': 'U0',
            'vsat_param': 'VSAT',
        }

        self.target_lis_params = list(self.param_map.keys())
        self.output_order = output_params_list

        # --- 正则表达式 ---

        # 1. 匹配 MC index 块 (不变)
        self.re_mc_block = re.compile(
            r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
            re.DOTALL
        )

        # 2. (!!! 关键修改 !!!) 匹配 I-V 数据 (x 块)
        # 匹配新的表头: volt param param \n i_linear i_sat
        self.re_iv_data = re.compile(
            r"x\n\n *volt *param *param *\n *i_linear *i_sat *\n(.*?)\ny\n",
            re.DOTALL | re.IGNORECASE  # re.IGNORECASE 增加鲁棒性
        )

        # 3. 匹配参数数据 (y 块) (不变)
        self.re_params = []
        for param_name in self.target_lis_params:
            self.re_params.append(
                (param_name, re.compile(r"{}=\s*([\w.+-]+)".format(param_name)))
            )

    def parse(self, lis_content: str):
        """
        执行解析
        """
        features_list = []
        labels_list = []

        mc_blocks = self.re_mc_block.findall(lis_content)
        if not mc_blocks:
            print("❌ 错误: 未在文件中找到任何 '*** monte carlo index = ... ***' 块。")
            return None, None

        print(f"🔍 找到 {len(mc_blocks)} 个 Monte Carlo 样本。开始解析...")

        for index, block_content in tqdm(mc_blocks, desc="解析 .lis 文件"):

            # 2. (!!! 关键修改 !!!) 提取 I-V 数据
            iv_match = self.re_iv_data.search(block_content)
            if not iv_match:
                print(f"警告: 在 Index {index} 中未找到 I-V 数据块 (x...y)。跳过...")
                continue

            iv_data_str = iv_match.group(1).strip()

            # (!!! 关键修改 !!!)
            # 我们现在需要为两条曲线分别创建列表
            voltages = []
            linear_currents = []
            sat_currents = []

            # 2.1 解析 I-V 数据行
            for line in iv_data_str.split('\n'):
                parts = line.strip().split()
                if len(parts) == 3:  # 检查是否有 3 列 (volt, i_linear, i_sat)
                    try:
                        voltages.append(parse_value(parts[0]))
                        linear_currents.append(parse_value(parts[1]))  # 第 2 列
                        sat_currents.append(parse_value(parts[2]))  # 第 3 列
                    except Exception as e:
                        print(f"警告: 在 Index {index} 中解析行失败: {line} -> {e}")

            # 2.2 (!!! 关键修改 !!!)
            # 将两条曲线拼接成一个特征向量
            # [i_linear_1, ..., i_linear_21, i_sat_1, ..., i_sat_21]
            if voltages and linear_currents and sat_currents:
                combined_features = voltages + linear_currents + sat_currents
                features_list.append(combined_features)
            else:
                print(f"警告: 在 Index {index} 中未提取到足够的 I-V 数据。跳过...")
                continue

            # 3. 提取参数数据 (不变)
            label_dict_raw = {}
            for param_name, re_c in self.re_params:
                param_match = re_c.search(block_content)
                if param_match:
                    label_dict_raw[param_name] = parse_value(param_match.group(1))

            if not label_dict_raw:
                print(f"警告: 在 Index {index} 中未找到任何参数 (y 块)。跳过...")
                # 移除刚刚添加的特征，确保 X 和 Y 对齐
                features_list.pop()
                continue

            # 4. 按 config.py 中的顺序排列标签 (不变)
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
                    label_ordered.append(0.0)

            labels_list.append(label_ordered)

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

    L_FILE_PATH = Path("G:\DeepLearning\data_set\mc.lis")  # <--- 修改这里: 你的.lis文件路径
    NPY_OUTPUT_DIR = Path("data/processed")  # <--- 修改这里: .npy的保存路径

    main(L_FILE_PATH, NPY_OUTPUT_DIR)