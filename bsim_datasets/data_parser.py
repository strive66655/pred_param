import pickle
import numpy as np
import re


def split_train_val_data(x_data, y_data, train_ratio=0.9, shuffle=True, random_state=42):
    """
    划分训练集和验证集

    参数:
        x_data: 输入特征数据，形状为 (n_samples, ...)
        y_data: 输出标签数据，形状为 (n_samples, ...)
        train_ratio: 训练集比例，默认0.9（90%）
        shuffle: 是否在划分前打乱数据，默认True
        random_state: 随机种子，保证可重复性

    返回:
        x_train, x_val, y_train, y_val: 划分后的训练集和验证集
    """
    n_samples = len(x_data)

    # 确保输入数据长度一致
    assert len(x_data) == len(y_data), "x_data和y_data的样本数量必须一致"

    # 生成索引
    indices = np.arange(n_samples)

    # 是否打乱数据
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    # 计算训练集大小
    train_size = int(n_samples * train_ratio)

    # 划分索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 根据索引划分数据
    x_train = x_data[train_indices]
    x_val = x_data[val_indices]
    y_train = y_data[train_indices]
    y_val = y_data[val_indices]

    # 打印划分结果
    print(f"数据集划分完成:")
    print(f"总样本数: {n_samples}")
    print(f"训练集样本数: {len(x_train)} ({len(x_train) / n_samples * 100:.1f}%)")
    print(f"验证集样本数: {len(x_val)} ({len(x_val) / n_samples * 100:.1f}%)")

    return x_train, x_val, y_train, y_val


def parse_hspice_mc_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式分割每次蒙特卡洛仿真产生的数据
    mc_blocks = re.split(r'\*\*\* monte carlo\s+index\s*=\s*\d+\s*\*\*\*', content)[1:]

    all_curves = []
    all_params = []

    for block in mc_blocks:
        # 提取曲线数据 (x部分)
        x_section = re.search(r'x\s+(.*?)y', block, re.DOTALL)
        if x_section:
            curve_data = []
            lines = x_section.group(1).strip().split('\n')

            # 跳过表头行，直接读取数据行
            for line in lines:
                line = line.strip()
                if line and not any(keyword in line.lower() for keyword in ['volt', 'current', 'vd_linear', 'vd_sat']):
                    # 分割行数据
                    parts = line.split()
                    if len(parts) >= 3:
                        # 处理电压值
                        volt_str = parts[0].lower()
                        if volt_str.endswith('.'):
                            volt = float(volt_str)
                        elif 'm' in volt_str:
                            volt = float(volt_str.replace('m', '')) * 1e-3
                        else:
                            volt = float(volt_str)

                        # 处理线性区电流值
                        current_linear_str = parts[1].lower()
                        current_linear = convert_current_unit(current_linear_str)

                        # 处理饱和区电流值
                        current_sat_str = parts[2].lower()
                        current_sat = convert_current_unit(current_sat_str)

                        curve_data.append([volt, current_linear, current_sat])

            # 将当前块的曲线数据添加到总列表中
            if curve_data:
                all_curves.append(curve_data)

        # 提取参数数据 (y部分)
        y_section = re.search(r'y\s+(.*?)(?:\*\*\*|$)', block, re.DOTALL)
        if y_section:
            params = []
            lines = y_section.group(1).strip().split('\n')

            for line in lines:
                line = line.strip()
                if 'vth0_value=' in line:
                    match = re.search(r'vth0_value=\s*([-\d.]+[mku]?)', line)
                    if match:
                        value_str = match.group(1).lower()
                        if 'm' in value_str:
                            vth0 = float(value_str.replace('m', '')) * 1e-3
                        else:
                            vth0 = float(value_str)
                        params.append(vth0)

                elif 'u0_param=' in line:
                    match = re.search(r'u0_param=\s*([-\d.]+[mku]?)', line)
                    if match:
                        value_str = match.group(1).lower()
                        if 'm' in value_str:
                            u0 = float(value_str.replace('m', '')) * 1e-3
                        else:
                            u0 = float(value_str)
                        params.append(u0)

                elif 'vsat_param=' in line:
                    match = re.search(r'vsat_param=\s*([-\d.]+[mku]?)', line)
                    if match:
                        value_str = match.group(1).lower()
                        if 'k' in value_str:
                            vsat = float(value_str.replace('k', '')) * 1e3
                        else:
                            vsat = float(value_str)
                        params.append(vsat)

            if len(params) == 3:
                all_params.append(params)

    return all_curves, all_params


def convert_current_unit(current_str):
    """转换电流单位"""
    if 'p' in current_str:
        return float(current_str.replace('p', '')) * 1e-12
    elif 'n' in current_str:
        return float(current_str.replace('n', '')) * 1e-9
    elif 'u' in current_str:
        return float(current_str.replace('u', '')) * 1e-6
    elif 'm' in current_str:
        return float(current_str.replace('m', '')) * 1e-3
    else:
        return float(current_str)


def prepare_deep_learning_data(curves, params):
    # 转换为numpy数组
    x_num = np.array(curves, dtype=np.float32)  # 形状: (样本数, 21, 3)
    y_num = np.array(params, dtype=np.float32)  # 形状: (样本数, 3)

    print(f"输入数据形状: {x_num.shape}")  # 应该是 (n_samples, 21, 3)
    print(f"输出数据形状: {y_num.shape}")  # 应该是 (n_samples, 3)

    print(f"样本电压范围: {x_num[:, :, 0].min():.3f}V 到 {x_num[:, :, 0].max():.3f}V")
    print(f"样本线性区电流范围: {x_num[:, :, 1].min():.3e}A 到 {x_num[:, :, 1].max():.3e}A")
    print(f"样本饱和区电流范围: {x_num[:, :, 2].min():.3e}A 到 {x_num[:, :, 2].max():.3e}A")
    print(f"样本Vth0范围: {y_num[:, 0].min():.6f} 到 {y_num[:, 0].max():.6f}")
    print(f"样本U0范围: {y_num[:, 1].min():.6f} 到 {y_num[:, 1].max():.6f}")
    print(f"样本Vsat范围: {y_num[:, 2].min():.2f} 到 {y_num[:, 2].max():.2f}")

    return x_num, y_num


def normalize_monte_carlo_data(all_curves, all_params):
    """
    对蒙特卡洛数据进行对数化和Z-Score标准化处理

    参数:
        all_curves: 从parse_hspice_mc_data返回的曲线数据
        all_params: 从parse_hspice_mc_data返回的参数数据

    返回:
        normalized_curves: 归一化后的曲线数据
        normalized_params: 归一化后的参数数据
        normalization_stats: 用于逆变换的统计量字典
    """
    # 转换为numpy数组以便处理
    curves_array = np.array(all_curves)  # 形状: (n_simulations, n_points, 3)
    params_array = np.array(all_params)  # 形状: (n_simulations, 3)

    # 分离各个分量
    volt_data = curves_array[:, :, 0]  # 电压
    i_linear_data = curves_array[:, :, 1]  # 线性区电流
    i_sat_data = curves_array[:, :, 2]  # 饱和区电流

    # 1. 电压归一化 (Min-Max到[0,1])
    volt_min = np.min(volt_data)
    volt_max = np.max(volt_data)
    volt_normalized = (volt_data - volt_min) / (volt_max - volt_min)

    # 2. 电流对数变换 + Z-Score标准化
    # 为避免对数0，给一个很小的偏移量
    epsilon = 1e-20

    # 线性区电流处理
    log_i_linear = np.log10(i_linear_data + epsilon)
    log_i_linear_mean = np.mean(log_i_linear)
    log_i_linear_std = np.std(log_i_linear)
    i_linear_normalized = (log_i_linear - log_i_linear_mean) / log_i_linear_std

    # 饱和区电流处理
    log_i_sat = np.log10(i_sat_data + epsilon)
    log_i_sat_mean = np.mean(log_i_sat)
    log_i_sat_std = np.std(log_i_sat)
    i_sat_normalized = (log_i_sat - log_i_sat_mean) / log_i_sat_std

    # 3. 参数归一化 (Z-Score)
    params_mean = np.mean(params_array, axis=0)
    params_std = np.std(params_array, axis=0)
    params_normalized = (params_array - params_mean) / params_std

    # 组合归一化后的曲线数据
    normalized_curves = np.stack([volt_normalized, i_linear_normalized, i_sat_normalized], axis=2)

    # 保存用于逆变换的统计量
    normalization_stats = {
        'volt': {'min': volt_min, 'max': volt_max},
        'i_linear': {'log_mean': log_i_linear_mean, 'log_std': log_i_linear_std},
        'i_sat': {'log_mean': log_i_sat_mean, 'log_std': log_i_sat_std},
        'params': {'mean': params_mean, 'std': params_std},
        'epsilon': epsilon
    }

    return normalized_curves, params_normalized, normalization_stats


# 使用示例
if __name__ == "__main__":

    curves, params = parse_hspice_mc_data(r"bsim_datasets/mc.lis")

    x_data, y_data = prepare_deep_learning_data(curves, params)

    x_normalized, y_normalized, stats = normalize_monte_carlo_data(curves, params)

    x_train, x_val, y_train, y_val = split_train_val_data(x_normalized, y_normalized, train_ratio=0.9)

    print(f"\n训练集输入形状: {x_train.shape}")
    print(f"训练集输出形状: {y_train.shape}")
    print(f"验证集输入形状: {x_val.shape}")
    print(f"验证集输出形状: {y_val.shape}")