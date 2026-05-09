# config.py
"""
Experiment configuration.
"""

from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


class ExperimentConfig:
    """Centralized experiment configuration."""

    def __init__(self):
        # Project info
        self.project_name = "DL_Parameter_Extraction"
        self.experiment_name = f"exp_MLP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Device
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        # Paths
        self.output_dir = Path("experiments") / self.experiment_name
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.plot_dir = self.output_dir / "plots"

        self.INPUT_LIS = [
            r"bsim_datasets/L=0.18u.lis",
            r"bsim_datasets/L=0.5u.lis",
            r"bsim_datasets/L=1.2u.lis",
            r"bsim_datasets/L=5u.lis",
            r"bsim_datasets/L=10u.lis",
            r"bsim_datasets/L=20u.lis",
        ]
        self.OUTPUT_NPZ = r"data/processed/converted_dataset.npz"

        # Data settings
        self.vg_points = 23
        self.joint_l_input = True
        self.num_lg = len(self.INPUT_LIS) if self.joint_l_input else 1
        # self.vd_values = [0.05, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.vd_values = [0.05, 1.8]

        self.vb_values = [0, -0.45, -0.90, -1.35, -1.80]
        self.num_vb = len(self.vb_values) if self.vb_values else 1
        self.mc_indices_per_vb = None
        self.num_curves = self.num_lg * self.num_vb * len(self.vd_values)

        self.include_raw_id = True
        self.include_log_id = True 
        self.include_gm_id = False
        self.include_log_gm = False
        self.include_log_curvature = False
        self.raw_input_dim = self.num_curves * self.vg_points
        self.include_l_feature = True
        self.l_feature_transform = "log10_um"
        self.extra_input_dim = self.num_lg if self.include_l_feature and self.joint_l_input else int(self.include_l_feature)
        self.feature_channels = self._count_feature_channels()
        self.input_dim = self.num_curves * self.feature_channels * self.vg_points + self.extra_input_dim

        self.output_params = [
            "VTH0",
            "VOFF",
            "NFACTOR",
            "K1",
            "K2",
            "U0",
            "UA",
            "UB",
            "UC",
            "AGS",
            "A0",
            "KETA",
            "DVT0",
            "DVT1",
            "DVT2",
            "LPE0",
            "LINT",
            "LUA",
            "LUB",
            "LUC",
            "DSUB",
            "ETA0",
            "ETAB",
            "LAGS",
            "LA0",
            "DROUT",
            "PDIBLC1",
            "PCLM",
            "RDSW",
            "LU0",
            "LNFACTOR",
        ]
        self.output_dim = len(self.output_params)

        # Preprocessing
        # Supported values: "minmax" and "zscore".
        self.normalization = "zscore"
        self.log_transform = True
        self.clip_min_current = 1e-13

        # Model
        self.model_type = "residual_mlp"
        self.mlp_layers = [1024, 512, 256]
        self.residual_hidden_dim = 256
        self.residual_blocks = 3
        self.dropout_rate = 0.1

        # Training
        self.batch_size = 64
        self.epochs = 3000
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        self.loss_function = "mse"
        self.scheduler = "plateau"
        self.scheduler_patience = 10
        self.scheduler_factor = 0.5
        self.early_stopping = True
        self.early_stopping_patience = 15

    def _count_feature_channels(self):
        return sum(
            int(flag)
            for flag in [
                self.include_raw_id,
                self.include_log_id,
                self.include_gm_id,
                self.include_log_gm,
                self.include_log_curvature,
            ]
        )

    def _create_dirs(self):
        """Create experiment directories."""
        for dir_path in [self.model_dir, self.log_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save(self):
        """Save configuration to a JSON file."""
        import json

        self._create_dirs()

        config_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Config saved: {self.output_dir / 'config.json'}")


config = ExperimentConfig()
