import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_mea_value(text: str, key: str) -> float | None:
    match = re.search(
        rf"{re.escape(key)}\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def parse_mea_idvg(mea_path: Path) -> dict[tuple[float, float], dict]:
    text = mea_path.read_text(encoding="utf-8", errors="ignore")
    curves: dict[tuple[float, float], dict] = {}
    current_vd = None
    current_vb = None
    current_w = None
    current_l = None
    current_t = None
    in_idvg_page = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Page"):
            in_idvg_page = "name=Ids_Vgs_Vds" in line and "x=Vgs" in line
            current_vd = parse_mea_value(line, "Vds") if in_idvg_page else None
            current_w = parse_mea_value(line, "W") if in_idvg_page else None
            current_l = parse_mea_value(line, "L") if in_idvg_page else None
            current_t = parse_mea_value(line, "T") if in_idvg_page else None
            current_vb = None
            continue

        if not in_idvg_page or current_vd is None:
            continue

        curve_match = re.match(r"curve\{([^}]+)\}", line, re.IGNORECASE)
        if curve_match:
            current_vb = float(curve_match.group(1))
            curves[(current_vb, current_vd)] = {
                "vgs": [],
                "ids": [],
                "w": current_w,
                "l": current_l,
                "t": current_t,
            }
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

        curve = curves[(current_vb, current_vd)]
        curve["vgs"].append(vgs)
        curve["ids"].append(ids)

    for curve in curves.values():
        curve["vgs"] = np.asarray(curve["vgs"], dtype=np.float32)
        curve["ids"] = np.asarray(curve["ids"], dtype=np.float32)
    return curves


def load_raw_iv_from_mea(mea_path: Path, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    curves = parse_mea_idvg(mea_path)
    vb_values = cfg.get("vb_values") or [0.0]
    vd_values = cfg["vd_values"]
    vg_points = int(cfg["vg_points"])

    raw_iv = []
    details = []
    missing = []

    for vb in vb_values:
        for vd in vd_values:
            key = None
            for curve_key in curves:
                if abs(curve_key[0] - float(vb)) <= 1e-9 and abs(curve_key[1] - float(vd)) <= 1e-9:
                    key = curve_key
                    break

            if key is None:
                missing.append((float(vb), float(vd)))
                continue

            curve = curves[key]
            order = np.argsort(curve["vgs"])
            vgs = curve["vgs"][order]
            ids = curve["ids"][order]
            interpolated = False

            if len(vgs) == vg_points:
                sampled_vgs = vgs
                sampled_ids = ids
            else:
                sampled_vgs = np.linspace(float(vgs.min()), float(vgs.max()), vg_points, dtype=np.float32)
                sampled_ids = np.interp(sampled_vgs, vgs, ids).astype(np.float32)
                interpolated = True

            raw_iv.extend(sampled_ids.tolist())
            details.append(
                {
                    "vb": float(vb),
                    "vd": float(vd),
                    "vgs": sampled_vgs,
                    "ids": sampled_ids,
                    "interpolated": interpolated,
                    "original_points": int(len(vgs)),
                    "w": curve.get("w"),
                    "l": curve.get("l"),
                    "t": curve.get("t"),
                }
            )

    if missing:
        available = sorted(curves.keys(), key=lambda item: (item[0], item[1]))
        raise ValueError(f"Missing measured curves: {missing}. Available curves: {available}")

    raw_iv = np.asarray(raw_iv, dtype=np.float32)
    expected = int(cfg["num_curves"]) * int(cfg["vg_points"])
    if raw_iv.size != expected:
        raise ValueError(f"Parsed measured input length {raw_iv.size}, expected {expected}.")
    return raw_iv, details


def build_features(raw_iv: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    raw_iv = np.asarray(raw_iv, dtype=np.float32).reshape(int(cfg["num_curves"]), int(cfg["vg_points"]))
    blocks = []
    names = []

    if cfg.get("include_raw_id", True):
        blocks.append(raw_iv.copy())
        names.append("raw_id")

    log_iv = None
    if cfg.get("include_log_id") or cfg.get("include_log_gm") or cfg.get("include_log_curvature"):
        clip_min = float(cfg.get("clip_min_current", 1e-13))
        log_iv = np.log10(np.clip(raw_iv, a_min=clip_min, a_max=None)).astype(np.float32)

    if cfg.get("include_log_id"):
        blocks.append(log_iv)
        names.append("log_id")

    if cfg.get("include_gm_id"):
        blocks.append(np.gradient(raw_iv, axis=1).astype(np.float32))
        names.append("gm_id")

    if cfg.get("include_log_gm"):
        blocks.append(np.gradient(log_iv, axis=1).astype(np.float32))
        names.append("dlog_id_dvg")

    if cfg.get("include_log_curvature"):
        blocks.append(np.gradient(np.gradient(log_iv, axis=1), axis=1).astype(np.float32))
        names.append("d2log_id_dvg2")

    if not blocks:
        raise ValueError("No feature blocks enabled in the experiment config.")

    features = np.stack(blocks, axis=1).reshape(-1).astype(np.float32)
    return features, names


def normalize_features(features: np.ndarray, norm_meta: dict) -> np.ndarray:
    normalization = norm_meta.get("normalization", "minmax").lower()
    if normalization == "minmax":
        iv_min = np.asarray(norm_meta["iv_min"], dtype=np.float32)
        iv_max = np.asarray(norm_meta["iv_max"], dtype=np.float32)
        iv_range = iv_max - iv_min
        iv_range[iv_range == 0] = 1.0
        return (features - iv_min) / iv_range

    if normalization in ("zscore", "z-score"):
        iv_mean = np.asarray(norm_meta["iv_mean"], dtype=np.float32)
        iv_std = np.asarray(norm_meta["iv_std"], dtype=np.float32)
        iv_std[iv_std == 0] = 1.0
        return (features - iv_mean) / iv_std

    raise ValueError(f"Unsupported normalization: {normalization}")


def feature_tensor(flat_features: np.ndarray, cfg: dict, channel_count: int) -> np.ndarray:
    return flat_features.reshape(int(cfg["num_curves"]), channel_count, int(cfg["vg_points"]))


def summarize_scores(scores: np.ndarray) -> dict:
    abs_scores = np.abs(scores)
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "rms": float(np.sqrt(np.mean(scores**2))),
        "max_abs": float(np.max(abs_scores)),
        "ratio_gt_2": float(np.mean(abs_scores > 2.0)),
        "ratio_gt_3": float(np.mean(abs_scores > 3.0)),
        "ratio_gt_5": float(np.mean(abs_scores > 5.0)),
        "count_gt_3": int(np.count_nonzero(abs_scores > 3.0)),
        "count_gt_5": int(np.count_nonzero(abs_scores > 5.0)),
        "total": int(scores.size),
    }


def print_summary(title: str, summary: dict) -> None:
    print(title)
    print(f"  score min/max: {summary['min']:.3f} / {summary['max']:.3f}")
    print(f"  score mean/std/rms: {summary['mean']:.3f} / {summary['std']:.3f} / {summary['rms']:.3f}")
    print(f"  max_abs_score: {summary['max_abs']:.3f}")
    print(
        "  ratio |score|>2/3/5: "
        f"{summary['ratio_gt_2']:.2%} / {summary['ratio_gt_3']:.2%} / {summary['ratio_gt_5']:.2%}"
    )
    print(f"  count |score|>3/5: {summary['count_gt_3']}/{summary['total']} / {summary['count_gt_5']}/{summary['total']}")


def compute_train_feature_bounds(train_npz: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(train_npz)
    raw = data["ivcv"]
    feats = []
    for sample in raw:
        sample_features, _ = build_features(sample, cfg)
        feats.append(sample_features)
    feat_matrix = np.stack(feats, axis=0)
    return feat_matrix.min(axis=0), feat_matrix.max(axis=0)


def verdict(summary: dict) -> str:
    if summary["ratio_gt_5"] > 0 or summary["ratio_gt_3"] > 0.05 or summary["max_abs"] > 5.0:
        return "OUT_OF_DISTRIBUTION"
    if summary["ratio_gt_3"] > 0.01 or summary["max_abs"] > 3.0:
        return "SUSPICIOUS"
    return "OK"


def print_metric_explanations(normalization: str) -> None:
    score_name = "z-score" if normalization in ("zscore", "z-score") else "minmax score"
    print("\nMetric meanings:")
    print(f"  score: normalized input feature value using training metadata ({score_name}).")
    print("  max_abs_score: worst single-point deviation; large values reveal local curve mismatch.")
    print("  rms: whole-input deviation energy; high rms means the full curve is shifted, not just one point.")
    print("  ratio |score|>2: fraction of mildly unusual points.")
    print("  ratio |score|>3: fraction of clearly unusual points; continuous regions above this are important.")
    print("  ratio |score|>5: severe outliers; predictions should be treated as extrapolation.")
    print("  out_of_train_minmax: count of feature points outside the training set pointwise envelope.")
    print("  clipped_log_points: raw Id points at or below clip_min_current before log10; too many means lost off-current detail.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether measured IV input matches the training distribution.")
    parser.add_argument("--mea", required=True, type=Path, help="Measured .mea file to inspect.")
    parser.add_argument("--exp", required=True, type=Path, help="Experiment directory containing config.json and models/iv_norm_meta.json.")
    parser.add_argument("--train-npz", default=None, type=Path, help="Optional training NPZ for pointwise min/max envelope checks.")
    parser.add_argument("--top-k", default=20, type=int, help="Number of worst feature points to print.")
    parser.add_argument("--explain", action="store_true", help="Print metric explanations.")
    args = parser.parse_args()

    cfg = load_json(args.exp / "config.json")
    norm_meta = load_json(args.exp / "models" / "iv_norm_meta.json")
    raw_iv, curve_details = load_raw_iv_from_mea(args.mea, cfg)
    features, feature_names = build_features(raw_iv, cfg)
    scores = normalize_features(features, norm_meta)

    score_tensor = feature_tensor(scores, cfg, len(feature_names))
    feature_tensor_values = feature_tensor(features, cfg, len(feature_names))
    raw_tensor = raw_iv.reshape(int(cfg["num_curves"]), int(cfg["vg_points"]))

    summary = summarize_scores(scores)
    print(f"Experiment: {args.exp}")
    print(f"Measured file: {args.mea}")
    print(f"Normalization: {norm_meta.get('normalization', 'minmax')}")
    print(f"Feature blocks: {feature_names}")
    print(f"Verdict: {verdict(summary)}")
    print_summary("\nOverall input distribution:", summary)

    clip_min = float(cfg.get("clip_min_current", 1e-13))
    clipped_count = int(np.count_nonzero(raw_iv <= clip_min))
    print(f"  clipped_log_points: {clipped_count}/{raw_iv.size} at or below {clip_min:.1e}")

    train_min = train_max = None
    if args.train_npz is not None:
        train_min, train_max = compute_train_feature_bounds(args.train_npz, cfg)
        below = features < train_min
        above = features > train_max
        print(f"  out_of_train_minmax: {int(np.count_nonzero(below | above))}/{features.size}")
        print(f"    below training min: {int(np.count_nonzero(below))}")
        print(f"    above training max: {int(np.count_nonzero(above))}")

    print("\nBy curve:")
    for curve_idx, detail in enumerate(curve_details):
        curve_scores = score_tensor[curve_idx].reshape(-1)
        curve_summary = summarize_scores(curve_scores)
        ids = detail["ids"]
        print(
            f"  Vb={detail['vb']:g}, Vd={detail['vd']:g}: "
            f"Vg=[{float(detail['vgs'][0]):g},{float(detail['vgs'][-1]):g}], "
            f"points={detail['original_points']}, interpolated={detail['interpolated']}, "
            f"W/L/T={detail['w']}/{detail['l']}/{detail['t']}, "
            f"Id=[{float(ids.min()):.4e},{float(ids.max()):.4e}], "
            f"max_abs={curve_summary['max_abs']:.3f}, "
            f"rms={curve_summary['rms']:.3f}, "
            f"|score|>3={curve_summary['count_gt_3']}/{curve_summary['total']}"
        )

    print(f"\nWorst {args.top_k} points:")
    flat_order = np.argsort(np.abs(scores))[::-1][: args.top_k]
    vg_points = int(cfg["vg_points"])
    channel_count = len(feature_names)
    for flat_idx in flat_order:
        curve_idx = flat_idx // (channel_count * vg_points)
        rem = flat_idx % (channel_count * vg_points)
        channel_idx = rem // vg_points
        vg_idx = rem % vg_points
        detail = curve_details[curve_idx]
        extra = ""
        if train_min is not None and (features[flat_idx] < train_min[flat_idx] or features[flat_idx] > train_max[flat_idx]):
            extra = " outside_train_minmax"
        print(
            f"  score={scores[flat_idx]: .3f}{extra} | "
            f"feature={feature_names[channel_idx]} value={feature_tensor_values[curve_idx, channel_idx, vg_idx]:.4e} | "
            f"Vb={detail['vb']:g}, Vd={detail['vd']:g}, Vg={float(detail['vgs'][vg_idx]):.3g}, "
            f"Id={raw_tensor[curve_idx, vg_idx]:.4e}"
        )

    if args.explain:
        print_metric_explanations(norm_meta.get("normalization", "minmax").lower())


if __name__ == "__main__":
    main()
