import argparse
import json
import sys
from dataclasses import dataclass
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.param_extractor_iv import ParamExtractorIVNet
from models.residual_param_extractor import ResidualMLPParamExtractor


@dataclass
class InferenceConfig:
    experiment_name: str
    model_type: str
    num_curves: int
    vg_points: int
    vd_values: list
    include_raw_id: bool
    include_log_id: bool
    include_gm_id: bool
    include_log_gm: bool
    include_log_curvature: bool
    clip_min_current: float
    output_params: list
    output_dim: int
    residual_hidden_dim: int
    residual_blocks: int
    dropout_rate: float
    mlp_layers: list

    @property
    def raw_input_dim(self) -> int:
        return self.num_curves * self.vg_points

    @property
    def feature_channels(self) -> int:
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

    @property
    def input_dim(self) -> int:
        return self.raw_input_dim * self.feature_channels


def load_experiment_config(exp_dir: Path) -> InferenceConfig:
    with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    allowed = {f.name for f in fields(InferenceConfig)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return InferenceConfig(**filtered)


def find_best_matching_experiment(experiments_dir: Path) -> Path:
    candidates = []
    for exp_dir in sorted(experiments_dir.glob("exp_*")):
        config_path = exp_dir / "config.json"
        model_path = exp_dir / "models" / "best_iv_extractor.pth"
        norm_path = exp_dir / "models" / "iv_norm_meta.json"
        if not (config_path.exists() and model_path.exists() and norm_path.exists()):
            continue

        try:
            cfg = load_experiment_config(exp_dir)
        except Exception:
            continue

        if cfg.model_type == "residual_mlp" and cfg.num_curves == 10 and cfg.vg_points == 37:
            candidates.append(exp_dir)

    if not candidates:
        raise FileNotFoundError("No matching experiment checkpoint was found under experiments/.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_model(cfg: InferenceConfig) -> torch.nn.Module:
    if cfg.model_type == "residual_mlp":
        return ResidualMLPParamExtractor(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            hidden_dim=cfg.residual_hidden_dim,
            num_blocks=cfg.residual_blocks,
            dropout=cfg.dropout_rate,
        )

    return ParamExtractorIVNet(
        input_dim=cfg.input_dim,
        hidden_layers=cfg.mlp_layers,
        output_dim=cfg.output_dim,
        dropout=cfg.dropout_rate,
    )


def build_features(raw_iv: np.ndarray, cfg: InferenceConfig) -> np.ndarray:
    raw_iv = np.asarray(raw_iv, dtype=np.float32).reshape(cfg.num_curves, cfg.vg_points)
    blocks = []

    if cfg.include_raw_id:
        blocks.append(raw_iv.copy())

    log_iv = None
    if cfg.include_log_id or cfg.include_log_gm or cfg.include_log_curvature:
        log_iv = np.log10(np.clip(raw_iv, a_min=cfg.clip_min_current, a_max=None)).astype(np.float32)

    if cfg.include_log_id:
        blocks.append(log_iv)

    if cfg.include_gm_id:
        blocks.append(np.gradient(raw_iv, axis=1).astype(np.float32))

    if cfg.include_log_gm:
        blocks.append(np.gradient(log_iv, axis=1).astype(np.float32))

    if cfg.include_log_curvature:
        blocks.append(np.gradient(np.gradient(log_iv, axis=1), axis=1).astype(np.float32))

    if not blocks:
        raise ValueError("No feature blocks enabled in the checkpoint config.")

    return np.stack(blocks, axis=1).reshape(-1).astype(np.float32)


def normalize_features(features: np.ndarray, norm_meta: dict) -> np.ndarray:
    iv_mu = np.asarray(norm_meta["iv_mu"], dtype=np.float32)
    iv_sigma = np.asarray(norm_meta["iv_sigma"], dtype=np.float32)
    iv_sigma[iv_sigma == 0] = 1.0
    return (features - iv_mu) / iv_sigma


def inverse_params(pred: np.ndarray, norm_meta: dict) -> np.ndarray:
    p_mu = np.asarray(norm_meta["params_mu"], dtype=np.float32)
    p_sigma = np.asarray(norm_meta["params_sigma"], dtype=np.float32)
    p_sigma[p_sigma == 0] = 1.0
    return pred * p_sigma + p_mu


def load_sample_from_dataset(dataset_path: Path, sample_index: int) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(dataset_path)
    raw_iv = np.asarray(data["ivcv"][sample_index], dtype=np.float32)
    true_params = None
    if "params" in data:
        true_params = np.asarray(data["params"][sample_index], dtype=np.float32)
    return raw_iv, true_params


def plot_prediction(raw_iv: np.ndarray, pred_params: np.ndarray, cfg: InferenceConfig, out_path: Path,
                    true_params: np.ndarray | None = None) -> None:
    raw_iv_2d = raw_iv.reshape(cfg.num_curves, cfg.vg_points)
    vg_axis = np.arange(cfg.vg_points)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    for i, vd in enumerate(cfg.vd_values):
        ax1.plot(vg_axis, raw_iv_2d[i], label=f"Vd={vd:g}V", linewidth=1.8)
    ax1.set_title("Input Id-Vg Curves")
    ax1.set_xlabel("Vg Point Index")
    ax1.set_ylabel("Id (A)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8, ncol=2)

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(len(cfg.output_params))
    width = 0.36 if true_params is not None else 0.6
    ax2.bar(x - (width / 2 if true_params is not None else 0), pred_params, width=width, label="Pred")
    if true_params is not None:
        ax2.bar(x + width / 2, true_params, width=width, label="True")
    ax2.set_title("Predicted BSIM Parameters")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfg.output_params, rotation=35, ha="right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    if true_params is not None:
        ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BSIM parameter prediction using the best checkpoint.")
    parser.add_argument("--dataset", default="data/processed/converted_dataset.npz")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--experiment-dir", default=None, help="Optional experiment directory override.")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction artifacts.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else find_best_matching_experiment(Path("experiments"))
    cfg = load_experiment_config(experiment_dir)

    checkpoint_path = experiment_dir / "models" / "best_iv_extractor.pth"
    norm_meta_path = experiment_dir / "models" / "iv_norm_meta.json"

    with open(norm_meta_path, "r", encoding="utf-8") as f:
        norm_meta = json.load(f)

    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    raw_iv, true_params = load_sample_from_dataset(Path(args.dataset), args.sample_index)
    features = build_features(raw_iv, cfg)
    features = normalize_features(features, norm_meta)

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(features).unsqueeze(0)).cpu().numpy()[0]
    pred_params = inverse_params(pred_norm, norm_meta)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / f"inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment_dir": str(experiment_dir.as_posix()),
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "sample_index": args.sample_index,
        "raw_input_format": {
            "shape": [cfg.num_curves, cfg.vg_points],
            "flat_length": cfg.raw_input_dim,
            "description": "10 Id-Vg curves stacked by Vd, each curve has 37 current points.",
            "vd_values": cfg.vd_values,
            "channel_order": ["raw_id", "log_id", "d2log_id_dvg2"],
        },
        "predicted_params": {
            name: float(value) for name, value in zip(cfg.output_params, pred_params)
        },
    }
    if true_params is not None:
        result["true_params"] = {
            name: float(value) for name, value in zip(cfg.output_params, true_params)
        }

    result_path = output_dir / "prediction.json"
    figure_path = output_dir / "prediction_visualization.png"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_prediction(raw_iv, pred_params, cfg, figure_path, true_params=true_params)

    print(f"Using experiment: {experiment_dir.as_posix()}")
    print(f"Prediction JSON: {result_path.as_posix()}")
    print(f"Visualization: {figure_path.as_posix()}")
    print("Required raw input:")
    print(f"  shape = ({cfg.num_curves}, {cfg.vg_points})")
    print(f"  flat_length = {cfg.raw_input_dim}")
    print(f"  vd_values = {cfg.vd_values}")
    print("Predicted params:")
    for name, value in zip(cfg.output_params, pred_params):
        print(f"  {name}: {value:.6e}")


if __name__ == "__main__":
    main()
