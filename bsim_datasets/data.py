import pickle
import numpy as np
import re
import os
import glob


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
    """
    解析HSPICE蒙特卡洛数据文件
    新格式：只有电压和i_d_0.1电流数据
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式分割每次蒙特卡洛仿真产生的数据
    mc_blocks = re.split(r'\*\*\* monte carlo\s+index\s*=\s*\d+\s*\*\*\*', content)

    # 如果第一个块是空或文件头，则跳过
    if mc_blocks and not mc_blocks[0].strip():
        mc_blocks = mc_blocks[1:]



    all_curves = []
    all_params = []
    skipped_blocks = 0

    for block_idx, block in enumerate(mc_blocks):
        # 提取曲线数据 (x部分)
        x_section = re.search(r'x\s+(.*?)y', block, re.DOTALL)
        if not x_section:
            # 尝试另一种模式，可能没有明确的y部分
            x_section = re.search(r'x\s+(.*?)(?:\*\*\*|$)', block, re.DOTALL)

        if x_section:
            curve_data = []
            lines = x_section.group(1).strip().split('\n')

            # 跳过表头行，找到数据开始位置
            data_started = False
            valid_points = 0

            for line in lines:
                line = line.strip()

                # 检查是否是数据行（包含数字和单位）
                if re.search(r'\d+\.?\d*[mku]?', line) and not any(
                        keyword in line.lower() for keyword in ['volt', 'param', 'i_d_0.1']):
                    data_started = True
                    parts = line.split()
                    if len(parts) >= 2:
                        # 处理电压值
                        volt_str = parts[0].lower()
                        try:
                            volt = convert_unit(volt_str)

                            # 处理电流值 (i_d_0.1)
                            current_str = parts[1].lower()
                            current = convert_unit(current_str)

                            # 新格式只有电压和电流两个值
                            curve_data.append([volt, current])
                            valid_points += 1
                        except ValueError as e:
                            print(f"    - 警告: 无法解析数据行: {line} - {e}")

                # 如果是表头行但还没有开始数据，继续
                elif not data_started and ('volt' in line.lower() or 'i_d_0.1' in line.lower()):
                    continue

            # 检查是否有足够的数据点
            if valid_points >= 10:  # 假设至少有10个有效数据点
                all_curves.append(curve_data)

            else:

                skipped_blocks += 1
                continue

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
                        try:
                            vth0 = convert_unit(value_str)
                            params.append(vth0)
                        except ValueError as e:
                            print(f"    - 警告: 无法解析vth0值: {line} - {e}")

                elif 'u0_param=' in line:
                    match = re.search(r'u0_param=\s*([-\d.]+[mku]?)', line)
                    if match:
                        value_str = match.group(1).lower()
                        try:
                            u0 = convert_unit(value_str)
                            params.append(u0)
                        except ValueError as e:
                            print(f"    - 警告: 无法解析u0值: {line} - {e}")

                elif 'ags_param=' in line:
                    match = re.search(r'ags_param=\s*([-\d.]+[mku]?)', line)
                    if match:
                        value_str = match.group(1).lower()
                        try:
                            ags = convert_unit(value_str)
                            params.append(ags)
                        except ValueError as e:
                            print(f"    - 警告: 无法解析ags值: {line} - {e}")

            if len(params) == 3:
                all_params.append(params)

            else:
                print(f"    - 块 {block_idx + 1}: 参数不完整 - 找到 {len(params)} 个参数")
                # 如果曲线数据已添加但参数不完整，移除对应的曲线数据
                if len(all_curves) > len(all_params):
                    all_curves.pop()
                skipped_blocks += 1
        else:
            print(f"    - 块 {block_idx + 1}: 未找到参数部分")
            # 如果曲线数据已添加但没有参数，移除对应的曲线数据
            if len(all_curves) > len(all_params):
                all_curves.pop()
            skipped_blocks += 1

    # 确保曲线和参数数量一致
    min_length = min(len(all_curves), len(all_params))
    if min_length < len(all_curves):
        all_curves = all_curves[:min_length]
    if min_length < len(all_params):
        all_params = all_params[:min_length]

    print(f"  - 成功解析 {min_length} 个样本，跳过 {skipped_blocks} 个块")

    return all_curves, all_params


def convert_unit(value_str):
    """转换单位"""
    value_str = value_str.lower().strip()

    # 移除可能的尾随点
    if value_str.endswith('.'):
        value_str = value_str[:-1]

    if 'p' in value_str:
        return float(value_str.replace('p', '')) * 1e-12
    elif 'n' in value_str:
        return float(value_str.replace('n', '')) * 1e-9
    elif 'u' in value_str:
        return float(value_str.replace('u', '')) * 1e-6
    elif 'm' in value_str:
        return float(value_str.replace('m', '')) * 1e-3
    elif 'k' in value_str:
        return float(value_str.replace('k', '')) * 1e3
    else:
        return float(value_str)


def prepare_deep_learning_data(curves, params):
    # 转换为numpy数组
    x_num = np.array(curves, dtype=np.float32)  # 形状: (样本数, n_points, 2)
    y_num = np.array(params, dtype=np.float32)  # 形状: (样本数, 3)

    print(f"输入数据形状: {x_num.shape}")  # 应该是 (n_samples, n_points, 2)
    print(f"输出数据形状: {y_num.shape}")  # 应该是 (n_samples, 3)

    print(f"样本电压范围: {x_num[:, :, 0].min():.3f}V 到 {x_num[:, :, 0].max():.3f}V")
    print(f"样本电流范围: {x_num[:, :, 1].min():.3e}A 到 {x_num[:, :, 1].max():.3e}A")
    print(f"样本Vth0范围: {y_num[:, 0].min():.6f} 到 {y_num[:, 0].max():.6f}")
    print(f"样本U0范围: {y_num[:, 1].min():.6f} 到 {y_num[:, 1].max():.6f}")
    print(f"样本Ags范围: {y_num[:, 2].min():.6f} 到 {y_num[:, 2].max():.6f}")

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
    curves_array = np.array(all_curves)  # 形状: (n_simulations, n_points, 2)
    params_array = np.array(all_params)  # 形状: (n_simulations, 3)

    # 分离各个分量
    volt_data = curves_array[:, :, 0]  # 电压
    current_data = curves_array[:, :, 1]  # 电流

    # 1. 电压归一化 (Min-Max到[0,1])
    volt_min = np.min(volt_data)
    volt_max = np.max(volt_data)
    volt_normalized = (volt_data - volt_min) / (volt_max - volt_min)

    # 2. 电流对数变换 + Z-Score标准化
    # 为避免对数0，给一个很小的偏移量
    epsilon = 1e-20

    # 电流处理
    log_current = np.log10(current_data + epsilon)
    log_current_mean = np.mean(log_current)
    log_current_std = np.std(log_current)
    current_normalized = (log_current - log_current_mean) / log_current_std

    # 3. 参数归一化 (Z-Score)
    params_mean = np.mean(params_array, axis=0)
    params_std = np.std(params_array, axis=0)
    params_normalized = (params_array - params_mean) / params_std

    # 组合归一化后的曲线数据
    normalized_curves = np.stack([volt_normalized, current_normalized], axis=2)

    # 保存用于逆变换的统计量
    normalization_stats = {
        'volt': {'min': volt_min, 'max': volt_max},
        'current': {'log_mean': log_current_mean, 'log_std': log_current_std},
        'params': {'mean': params_mean, 'std': params_std},
        'epsilon': epsilon
    }

    return normalized_curves, params_normalized, normalization_stats


def process_folder(folder_path, file_pattern="*.lis"):
    """
    处理整个文件夹中的HSPICE蒙特卡洛数据文件


    返回:
        all_curves: 所有文件的曲线数据
        all_params: 所有文件的参数数据
        file_info: 文件信息列表
    """
    all_curves = []
    all_params = []
    file_info = []

    # 获取所有匹配的文件
    file_paths = glob.glob(os.path.join(folder_path, file_pattern))

    if not file_paths:
        print(f"在文件夹 {folder_path} 中没有找到匹配 {file_pattern} 的文件")
        return all_curves, all_params, file_info

    print(f"找到 {len(file_paths)} 个文件，开始处理...")

    for i, file_path in enumerate(file_paths):
        print(f"处理文件 {i + 1}/{len(file_paths)}: {os.path.basename(file_path)}")

        try:
            curves, params = parse_hspice_mc_data(file_path)

            if curves and params and len(curves) == len(params):
                all_curves.extend(curves)
                all_params.extend(params)
                file_info.append({
                    'filename': os.path.basename(file_path),
                    'samples': len(curves),
                    'file_path': file_path
                })
                print(f"  - 成功提取 {len(curves)} 个样本")
            else:
                print(f"  - 警告: 文件中没有找到有效数据或数据不匹配")
                print(f"    - 曲线数: {len(curves) if curves else 0}")
                print(f"    - 参数数: {len(params) if params else 0}")

        except Exception as e:
            print(f"  - 错误: 处理文件时发生异常 - {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n文件夹处理完成:")
    print(f"总文件数: {len(file_paths)}")
    print(f"成功处理文件数: {len(file_info)}")
    print(f"总样本数: {len(all_curves)}")

    # 验证数据一致性
    if len(all_curves) != len(all_params):
        print(f"警告: 曲线数据数量 ({len(all_curves)}) 和参数数据数量 ({len(all_params)}) 不匹配!")
        # 取较小值以确保一致性
        min_length = min(len(all_curves), len(all_params))
        all_curves = all_curves[:min_length]
        all_params = all_params[:min_length]
        print(f"已截断数据至 {min_length} 个样本以确保一致性")

    return all_curves, all_params, file_info


# 使用示例
if __name__ == "__main__":
    # 处理整个文件夹
    folder_path = r"F:\pred_param\bsim_datasets"
    all_curves, all_params, file_info = process_folder(folder_path)

    if all_curves and all_params:
        # 准备深度学习数据
        x_data, y_data = prepare_deep_learning_data(all_curves, all_params)

        # 数据归一化
        x_normalized, y_normalized, stats = normalize_monte_carlo_data(x_data, y_data)

        # 划分训练集和验证集
        x_train, x_val, y_train, y_val = split_train_val_data(x_normalized, y_normalized)

        print("数据处理完成！")

        # 保存为 npz 文件（推荐使用压缩格式）
        save_path = r"F:\pred_param\bsim_datasets\dataset_processed.npz"
        np.savez_compressed(
            save_path,
            x_train=x_train,
            x_val=x_val,
            y_train=y_train,
            y_val=y_val,
            stats=stats  # 可选：保存归一化参数以便反归一化
        )
        print(f"✅ 数据已保存到: {save_path}")
