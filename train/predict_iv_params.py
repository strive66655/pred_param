import argparse
import json
import re
import sys
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import field
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
    vb_values: list = field(default_factory=list)

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

        candidates.append(exp_dir)

    if not candidates:
        raise FileNotFoundError("No matching experiment checkpoint was found under experiments/.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_mea_value(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_mea_idvg(mea_path: Path) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """
    Parse measured Id-Vg pages keyed by (Vbs, Vds).

    Expected pages look like:
      Page (name=Ids_Vgs_Vds,x=Vgs,p=Vbs,y=Ids){Vds=...}
      curve{Vbs}
      Vgs Ids
    """
    text = mea_path.read_text(encoding="utf-8", errors="ignore")
    curves: dict[tuple[float, float], tuple[list[float], list[float]]] = {}
    current_vd = None
    current_vb = None
    in_idvg_page = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Page"):
            in_idvg_page = "name=Ids_Vgs_Vds" in line and "x=Vgs" in line
            current_vd = parse_mea_value(line, "Vds") if in_idvg_page else None
            current_vb = None
            continue

        if not in_idvg_page or current_vd is None:
            continue

        curve_match = re.match(r"curve\{([^}]+)\}", line, re.IGNORECASE)
        if curve_match:
            current_vb = float(curve_match.group(1))
            curves.setdefault((current_vb, current_vd), ([], []))
            continue

        if current_vb is None:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            vgs = float(parts[0])
            ids = float(parts[1])
        except ValueError:
            continue

        x_values, y_values = curves[(current_vb, current_vd)]
        x_values.append(vgs)
        y_values.append(ids)

    return {
        key: (np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32))
        for key, (x_values, y_values) in curves.items()
    }


def find_curve(
    curves: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    vb: float,
    vd: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray] | None:
    for (curve_vb, curve_vd), curve in curves.items():
        if abs(curve_vb - vb) <= tol and abs(curve_vd - vd) <= tol:
            return curve
    return None


def load_sample_from_mea(mea_path: Path, cfg: InferenceConfig) -> np.ndarray:
    curves = parse_mea_idvg(mea_path)
    if not curves:
        raise ValueError(f"No Id-Vg measured curves found in {mea_path}.")

    vb_values = cfg.vb_values or [0.0]
    if cfg.num_curves != len(vb_values) * len(cfg.vd_values):
        raise ValueError(
            "Checkpoint config is inconsistent: "
            f"num_curves={cfg.num_curves}, len(vb_values)*len(vd_values)="
            f"{len(vb_values) * len(cfg.vd_values)}."
        )

    available = sorted(curves.keys(), key=lambda item: (item[0], item[1]))
    raw_iv = []
    missing = []

    for vb in vb_values:
        for vd in cfg.vd_values:
            curve = find_curve(curves, float(vb), float(vd))
            if curve is None:
                missing.append((float(vb), float(vd)))
                continue

            vgs, ids = curve
            order = np.argsort(vgs)
            vgs = vgs[order]
            ids = ids[order]

            if len(vgs) == cfg.vg_points:
                sampled_ids = ids
            else:
                target_vgs = np.linspace(float(vgs.min()), float(vgs.max()), cfg.vg_points, dtype=np.float32)
                sampled_ids = np.interp(target_vgs, vgs, ids).astype(np.float32)

            raw_iv.extend(sampled_ids.tolist())

    if missing:
        raise ValueError(
            "Measured file does not contain all checkpoint-required (Vbs, Vds) curves. "
            f"Missing: {missing}. Available: {available}"
        )

    raw_iv = np.asarray(raw_iv, dtype=np.float32)
    if raw_iv.size != cfg.raw_input_dim:
        raise ValueError(f"Parsed measured input length {raw_iv.size}, expected {cfg.raw_input_dim}.")
    return raw_iv


def _median_filter_1d(values: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1 or values.size == 0:
        return values.astype(np.float32, copy=True)

    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    filtered = np.empty_like(values, dtype=np.float32)
    for idx in range(values.size):
        filtered[idx] = np.median(padded[idx:idx + kernel_size])
    return filtered


def preprocess_measured_curve(ids: np.ndarray, cfg: InferenceConfig) -> tuple[np.ndarray, dict]:
    ids = np.asarray(ids, dtype=np.float32).copy()
    stats = {
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(ids))),
        "negative_count": int(np.count_nonzero(ids < 0)),
        "below_clip_count_before": int(np.count_nonzero(ids < cfg.clip_min_current)),
    }

    finite_mask = np.isfinite(ids)
    if not np.all(finite_mask):
        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size == 0:
            ids.fill(cfg.clip_min_current)
        else:
            ids[~finite_mask] = np.interp(
                np.flatnonzero(~finite_mask),
                finite_idx,
                ids[finite_idx],
            ).astype(np.float32)

    positive = ids[ids > cfg.clip_min_current]
    if positive.size:
        weak_floor = float(np.percentile(positive, 15))
        noise_floor = max(cfg.clip_min_current, weak_floor * 0.25)
        noise_floor = min(noise_floor, float(np.max(positive)) * 1e-3)
        noise_floor = max(noise_floor, cfg.clip_min_current)
    else:
        noise_floor = cfg.clip_min_current

    ids = np.maximum(ids, noise_floor).astype(np.float32)
    log_ids = np.log10(ids).astype(np.float32)

    smoothed_log = _median_filter_1d(log_ids, kernel_size=3)
    smoothed_log = _median_filter_1d(smoothed_log, kernel_size=3)

    max_current = float(np.max(ids)) if ids.size else noise_floor
    weak_threshold = max(noise_floor * 40.0, max_current * 1e-3, cfg.clip_min_current)
    weak_mask = ids <= weak_threshold

    monotonic_log = np.maximum.accumulate(smoothed_log)
    processed_log = log_ids.copy()
    processed_log[weak_mask] = monotonic_log[weak_mask]

    processed_ids = np.power(10.0, processed_log, dtype=np.float32)
    stats.update(
        {
            "noise_floor": float(noise_floor),
            "weak_threshold": float(weak_threshold),
            "weak_points": int(np.count_nonzero(weak_mask)),
            "below_clip_count_after": int(np.count_nonzero(processed_ids < cfg.clip_min_current)),
            "max_abs_log_shift": float(np.max(np.abs(processed_log - log_ids))) if ids.size else 0.0,
        }
    )
    return processed_ids.astype(np.float32), stats


def preprocess_measured_raw_iv(raw_iv: np.ndarray, cfg: InferenceConfig) -> tuple[np.ndarray, dict]:
    raw_iv_2d = np.asarray(raw_iv, dtype=np.float32).reshape(cfg.num_curves, cfg.vg_points)
    processed_curves = []
    curve_stats = []

    for curve_idx, curve in enumerate(raw_iv_2d):
        processed_curve, stats = preprocess_measured_curve(curve, cfg)
        stats["curve_index"] = curve_idx
        processed_curves.append(processed_curve)
        curve_stats.append(stats)

    processed = np.stack(processed_curves, axis=0).astype(np.float32)
    summary = {
        "method": "adaptive_floor_log_median_monotonic",
        "curve_stats": curve_stats,
        "total_negative_count": int(sum(item["negative_count"] for item in curve_stats)),
        "total_nonfinite_count": int(sum(item["nonfinite_count"] for item in curve_stats)),
        "total_weak_points": int(sum(item["weak_points"] for item in curve_stats)),
        "max_abs_log_shift": float(max(item["max_abs_log_shift"] for item in curve_stats)) if curve_stats else 0.0,
    }
    return processed.reshape(-1), summary


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
    iv_min = np.asarray(norm_meta["iv_min"], dtype=np.float32)
    iv_max = np.asarray(norm_meta["iv_max"], dtype=np.float32)
    iv_range = iv_max - iv_min
    iv_range[iv_range == 0] = 1.0
    return (features - iv_min) / iv_range


def inverse_params(pred: np.ndarray, norm_meta: dict) -> np.ndarray:
    p_min = np.asarray(norm_meta["params_min"], dtype=np.float32)
    p_max = np.asarray(norm_meta["params_max"], dtype=np.float32)
    return pred * (p_max - p_min) + p_min


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
    vb_values = cfg.vb_values or [None]
    curve_labels = []
    for vb in vb_values:
        for vd in cfg.vd_values:
            if vb is None:
                curve_labels.append(f"Vd={vd:g}V")
            else:
                curve_labels.append(f"Vbs={vb:g}V, Vds={vd:g}V")

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(cfg.num_curves):
        label = curve_labels[i] if i < len(curve_labels) else f"Curve {i}"
        ax1.plot(vg_axis, raw_iv_2d[i], label=label, linewidth=1.8)
    ax1.set_title("Input Id-Vg Curves (Linear Scale)")
    ax1.set_xlabel("Vg Point Index")
    ax1.set_ylabel("Id (A)")
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


def get_channel_names(cfg: InferenceConfig) -> list[str]:
    return [
        name for flag, name in [
            (cfg.include_raw_id, "raw_id"),
            (cfg.include_log_id, "log_id"),
            (cfg.include_gm_id, "gm_id"),
            (cfg.include_log_gm, "dlog_id_dvg"),
            (cfg.include_log_curvature, "d2log_id_dvg2"),
        ] if flag
    ]


def plot_model_input(features: np.ndarray, cfg: InferenceConfig, out_path: Path) -> None:
    """Plot the exact normalized features passed into the model."""
    channel_names = get_channel_names(cfg)
    feature_array = np.asarray(features, dtype=np.float32).reshape(
        cfg.num_curves,
        len(channel_names),
        cfg.vg_points,
    )
    vg_axis = np.arange(cfg.vg_points)
    vb_values = cfg.vb_values or [None]
    curve_labels = []
    for vb in vb_values:
        for vd in cfg.vd_values:
            if vb is None:
                curve_labels.append(f"Vd={vd:g}V")
            else:
                curve_labels.append(f"Vbs={vb:g}V, Vds={vd:g}V")

    fig, axes = plt.subplots(
        len(channel_names),
        cfg.num_curves,
        figsize=(3.2 * cfg.num_curves, 3.0 * len(channel_names)),
        squeeze=False,
        sharex=True,
    )

    for channel_idx, channel_name in enumerate(channel_names):
        for curve_idx in range(cfg.num_curves):
            ax = axes[channel_idx, curve_idx]
            ax.plot(vg_axis, feature_array[curve_idx, channel_idx], linewidth=1.6)
            if channel_idx == 0:
                title = curve_labels[curve_idx] if curve_idx < len(curve_labels) else f"Curve {curve_idx}"
                ax.set_title(title, fontsize=9)
            if curve_idx == 0:
                ax.set_ylabel(f"{channel_name}\nnormalized")
            ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Vg Point Index")

    fig.suptitle("Model Input After Training-Time Preprocessing", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BSIM parameter prediction using the best checkpoint.")
    parser.add_argument("--dataset", default="data/processed/converted_dataset.npz")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mea", default=None, help="Measured .mea file to predict instead of a dataset sample.")
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

    true_params = None
    preprocessing = None
    if args.mea:
        raw_iv = load_sample_from_mea(Path(args.mea), cfg)
        raw_iv, preprocessing = preprocess_measured_raw_iv(raw_iv, cfg)
        source = {"type": "mea", "path": str(Path(args.mea).as_posix())}
    else:
        raw_iv, true_params = load_sample_from_dataset(Path(args.dataset), args.sample_index)
        source = {"type": "dataset", "path": str(Path(args.dataset).as_posix()), "sample_index": args.sample_index}

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
        "source": source,
        "raw_input_format": {
            "shape": [cfg.num_curves, cfg.vg_points],
            "flat_length": cfg.raw_input_dim,
            "description": "Id-Vg curves stacked by Vbs then Vds.",
            "vb_values": cfg.vb_values,
            "vd_values": cfg.vd_values,
            "channel_order": [
                name for flag, name in [
                    (cfg.include_raw_id, "raw_id"),
                    (cfg.include_log_id, "log_id"),
                    (cfg.include_gm_id, "gm_id"),
                    (cfg.include_log_gm, "dlog_id_dvg"),
                    (cfg.include_log_curvature, "d2log_id_dvg2"),
                ] if flag
            ],
        },
        "predicted_params": {
            name: float(value) for name, value in zip(cfg.output_params, pred_params)
        },
    }
    if preprocessing is not None:
        result["mea_preprocessing"] = preprocessing
    if true_params is not None:
        result["true_params"] = {
            name: float(value) for name, value in zip(cfg.output_params, true_params)
        }

    result_path = output_dir / "prediction.json"
    figure_path = output_dir / "prediction_visualization.png"
    model_input_figure_path = output_dir / "model_input_visualization.png"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_prediction(raw_iv, pred_params, cfg, figure_path, true_params=true_params)
    plot_model_input(features, cfg, model_input_figure_path)

    print(f"Using experiment: {experiment_dir.as_posix()}")
    print(f"Prediction JSON: {result_path.as_posix()}")
    print(f"Visualization: {figure_path.as_posix()}")
    print(f"Model input visualization: {model_input_figure_path.as_posix()}")
    print("Required raw input:")
    print(f"  shape = ({cfg.num_curves}, {cfg.vg_points})")
    print(f"  flat_length = {cfg.raw_input_dim}")
    print(f"  vb_values = {cfg.vb_values}")
    print(f"  vd_values = {cfg.vd_values}")
    print("Predicted params:")
    for name, value in zip(cfg.output_params, pred_params):
        print(f"  {name}: {value:.6e}")


if __name__ == "__main__":
    main()
