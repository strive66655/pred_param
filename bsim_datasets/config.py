# config.py
"""
实验配置中心
- 集中管理所有超参数和项目设置。
- 当前任务: 仅I-V数据, 预测部分参数, 使用MLP模型。
"""

import torch
from pathlib import Path
from datetime import datetime


class ExperimentConfig:
    """实验配置"""

    def __init__(self):
        # ===== 项目信息 =====
        self.project_name = "DL_Parameter_Extraction"
        self.experiment_name = f"exp_MLP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ===== 设备配置 =====
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ===== 路径配置 =====
        self.output_dir = Path("experiments") / self.experiment_name
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.plot_dir = self.output_dir / "plots"

        # ===== 数据配置 (当前任务) =====
        # (基于 mc.lis 文件的真实结构)
        self.num_curves = 3  # .lis 文件每个index只有1条 I-V 曲线
        self.vg_points = 21  # 每条曲线有 21 个点 (0V 到 1.0V)
        self.num_lg = 1  # 这个 .lis 文件似乎是单个Lg的MC，而不是全局的
        # 总输入特征维度 = 1 * 21 * 1 = 21
        self.input_dim = self.num_curves * self.vg_points * self.num_lg

        # 我们只提取 .lis 文件中真实存在的参数
        self.output_params = ['VTH0', 'U0', 'VSAT']  # 必须与 data_parser.py 的映射一致
        self.output_dim = len(self.output_params)  # output_dim 现在是 3

        # ===== 数据预处理配置 =====
        # 对电流使用log变换非常重要，尤其是亚阈值区域
        self.normalization = "minmax"  # 'minmax' 或 'standard'
        self.log_transform = True  # 对特征(电流)进行log10变换
        self.clip_min_current = 1e-12  # log变换前的最小电流值

        # ===== 模型配置 (当前任务) =====
        self.model_type = "mlp"  # 先从MLP开始 [cite: 201]

        # MLP配置
        self.mlp_layers = [1024, 512, 256, 128]  # 隐藏层 [cite: 201]
        self.dropout_rate = 0.2

        # ===== 训练配置 =====
        self.batch_size = 64
        self.epochs = 200
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5  # L2正则化

        # 损失函数: MSE (均方误差)
        self.loss_function = "mse"

        # 学习率调度
        self.scheduler = "plateau"  # 'plateau', 'cosine', 'step', None
        self.scheduler_patience = 10
        self.scheduler_factor = 0.5

        # 早停
        self.early_stopping = True
        self.early_stopping_patience = 25

        # ===== 目录创建 =====
        self._create_dirs()

    def _create_dirs(self):
        """创建实验目录"""
        for dir_path in [self.model_dir, self.log_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save(self):
        """保存配置到json文件"""
        import json
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"✓ 配置已保存: {self.output_dir / 'config.json'}")


# 创建一个全局可用的配置实例
config = ExperimentConfig()