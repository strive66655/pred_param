# data_parser.py
import os
import re
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

if __package__ in (None, ""):
    from config import config
else:
    from .config import config


VALUE_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:meg|[fpnumkaxgt])?)"


def parse_l_um_from_path(path) -> float:
    """Parse device length from names like L=0.5u.lis or *_L0p18_*.lis."""
    name = Path(path).stem.lower()
    match = re.search(r"(?:^|[_\s-])l\s*=\s*([\d.]+|[\d]+p[\d]+)\s*([a-z]*)", name)
    if match is None:
        match = re.search(r"(?:^|[_\s-])l([\d.]+|[\d]+p[\d]+)\s*([a-z]*)", name)
    if match is None:
        raise ValueError(f"Cannot parse L from file name: {path}")

    value = float(match.group(1).replace("p", "."))
    unit = match.group(2)
    if unit in ("", "u", "um"):
        return value
    if unit in ("n", "nm"):
        return value * 1e-3
    if unit in ("m", "meter", "meters"):
        return value * 1e6
    raise ValueError(f"Unsupported L unit '{unit}' in file name: {path}")


def build_l_feature(l_um: float) -> float:
    transform = getattr(config, "l_feature_transform", "log10_um").lower()
    if transform == "log10_um":
        if l_um <= 0:
            raise ValueError(f"L must be positive for log10 transform, got {l_um}.")
        return float(np.log10(l_um))
    if transform in ("um", "raw_um"):
        return float(l_um)
    raise ValueError(f"Unsupported L feature transform: {transform}")


def parse_value(value_str: str) -> float:
    """Parse HSPICE numeric values, including engineering suffixes."""
    value_str = value_str.strip().lower()
    suffixes = {
        "a": 1e-18,
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "x": 1e6,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }

    if value_str.endswith("meg"):
        try:
            return float(value_str[:-3]) * suffixes["meg"]
        except ValueError:
            return 0.0

    try:
        suffix = value_str[-1]
        if suffix in suffixes:
            return float(value_str[:-1]) * suffixes[suffix]
        return float(value_str)
    except ValueError:
        return 0.0


class HspiceLisParser:
    def __init__(self, output_params_list):
        self.full_param_pool = {
               "vth0": "VTH0",
                "voff": "VOFF",
                "nfactor": "NFACTOR",
                "k1": "K1",
                "k2": "K2",
                "u0": "U0",
                "ua": "UA",
                "ub": "UB",
                "uc": "UC",
                "ags": "AGS",
                "a0": "A0",
                "keta": "KETA",
                "dvt0": "DVT0",
                "dvt1": "DVT1",
                "dvt2": "DVT2",
                "lpe0": "LPE0",
                "lint": "LINT",
                "lua": "LUA",
                "lub": "LUB",
                "luc": "LUC",
                "dsub": "DSUB",
                "eta0": "ETA0",
                "etab": "ETAB",
                "lags": "LAGS",
                "la0": "LA0",
                "drout": "DROUT",
                "pdiblc1": "PDIBLC1",
                "pclm": "PCLM",
                "rdsw": "RDSW",
                "lu0": "LU0",
                "lnfactor": "LNFACTOR", 
        }
        self.output_order = output_params_list

        self.re_mc_block = re.compile(
            r"\*\*\* monte carlo +index = +(\d+) \*\*\*(.*?)(?=\*\*\* monte carlo|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        self.re_xy_block = re.compile(
            r"x\s*\n\s*volt\s+param[\s\S]*?\n([\s\S]*?)\s*y",
            re.IGNORECASE,
        )
        self.re_vd_header = re.compile(r"i_d_([\d.]+)", re.IGNORECASE)

    def _parse_mc_block(self, block_content: str):
        vd_id_map = {}

        for xy_text in self.re_xy_block.findall(block_content):
            lines = xy_text.strip().splitlines()
            if not lines:
                continue

            local_vds = [float(v) for v in self.re_vd_header.findall(lines[0])]
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                for i, id_val_str in enumerate(parts[1:]):
                    if i >= len(local_vds):
                        break
                    vd = local_vds[i]
                    vd_id_map.setdefault(vd, []).append(parse_value(id_val_str))

        features = []
        for vd_bias in config.vd_values:
            if vd_bias not in vd_id_map:
                return None, None

            id_curve = vd_id_map[vd_bias]
            if len(id_curve) < config.vg_points:
                return None, None

            features.extend(id_curve[:config.vg_points])

        label_dict = {}
        for lis_name, bsim_name in self.full_param_pool.items():
            reg = re.compile(rf"{re.escape(lis_name)}=\s*{VALUE_PATTERN}", re.IGNORECASE)
            match = reg.search(block_content)
            if match:
                label_dict[bsim_name] = parse_value(match.group(1))

        labels = [label_dict.get(param_name, 0.0) for param_name in self.output_order]
        return features, labels

    def _parse_records(self, lis_content: str):
        records = []
        mc_blocks = self.re_mc_block.findall(lis_content)
        if not mc_blocks:
            print("Error: no Monte Carlo blocks found.")
            return records

        print(f"Found {len(mc_blocks)} Monte Carlo blocks. Target params: {self.output_order}")
        for index, block_content in tqdm(mc_blocks, desc="Parse .lis"):
            features, labels = self._parse_mc_block(block_content)
            if features is None:
                continue
            records.append(
                {
                    "index": int(index),
                    "features": features,
                    "labels": labels,
                }
            )

        return records

    def _merge_vb_records(self, records):
        vb_values = getattr(config, "vb_values", None)
        num_vb = len(vb_values) if vb_values else int(getattr(config, "num_vb", 1))
        indices_per_vb = getattr(config, "mc_indices_per_vb", None)

        if num_vb <= 1:
            return (
                np.array([record["features"] for record in records], dtype=np.float32),
                np.array([record["labels"] for record in records], dtype=np.float32),
            )

        if indices_per_vb is None:
            if len(records) % num_vb != 0:
                raise ValueError(
                    f"Cannot split {len(records)} records into {num_vb} Vb groups. "
                    "Set config.mc_indices_per_vb explicitly."
                )
            indices_per_vb = len(records) // num_vb

        expected_records = indices_per_vb * num_vb
        if len(records) < expected_records:
            raise ValueError(
                f"Need at least {expected_records} records for {num_vb} Vb groups, "
                f"but only parsed {len(records)}."
            )

        features_list = []
        labels_list = []
        expected_single_vb_dim = len(config.vd_values) * config.vg_points

        # Expected record order:
        # [Vb1 index 1..N], [Vb2 index 1..N], [Vb3 index 1..N], ...
        for local_idx in range(indices_per_vb):
            grouped_features = []
            grouped_labels = []

            for vb_idx in range(num_vb):
                record = records[vb_idx * indices_per_vb + local_idx]
                if len(record["features"]) != expected_single_vb_dim:
                    raise ValueError(
                        f"Unexpected single-Vb feature length at local index {local_idx + 1}: "
                        f"{len(record['features'])}, expected {expected_single_vb_dim}."
                    )
                grouped_features.extend(record["features"])
                grouped_labels.append(np.asarray(record["labels"], dtype=np.float32))

            first_label = grouped_labels[0]
            for label in grouped_labels[1:]:
                if not np.allclose(first_label, label, rtol=1e-4, atol=1e-8):
                    print(
                        f"Warning: labels differ inside Vb group at local index {local_idx + 1}; "
                        "using the first Vb label."
                    )
                    break

            features_list.append(grouped_features)
            labels_list.append(first_label)

        return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)

    def parse(self, lis_content: str):
        records = self._parse_records(lis_content)
        if not records:
            return None, None
        return self._merge_vb_records(records)

    def _parse_file_without_l_feature(self, lis_file_path):
        lis_file_path = Path(lis_file_path)
        print(f"Reading .lis file: {lis_file_path}")
        try:
            content = lis_file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = lis_file_path.read_text(encoding="latin1")

        records = self._parse_records(content)
        if not records:
            return None, None

        return self._merge_vb_records(records)

    def parse_file(self, lis_file_path):
        features, labels = self._parse_file_without_l_feature(lis_file_path)
        if features is None:
            return None, None

        if getattr(config, "include_l_feature", False):
            l_um = parse_l_um_from_path(lis_file_path)
            l_feature = build_l_feature(l_um)
            l_column = np.full((features.shape[0], 1), l_feature, dtype=np.float32)
            features = np.concatenate([features, l_column], axis=1)
            print(f"Added L feature from file name: L={l_um:g} um, feature={l_feature:g}")

        return features, labels

    def parse_files(self, lis_file_paths):
        if getattr(config, "joint_l_input", False):
            return self.parse_files_joint_l(lis_file_paths)

        all_features = []
        all_labels = []

        for lis_file_path in lis_file_paths:
            features, labels = self.parse_file(lis_file_path)
            if features is None:
                continue
            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            return None, None
        return (
            np.concatenate(all_features, axis=0).astype(np.float32),
            np.concatenate(all_labels, axis=0).astype(np.float32),
        )

    def parse_files_joint_l(self, lis_file_paths):
        per_l_features = []
        per_l_labels = []
        l_features = []
        min_samples = None

        for lis_file_path in lis_file_paths:
            features, labels = self._parse_file_without_l_feature(lis_file_path)
            if features is None:
                raise ValueError(f"No valid samples parsed from joint-L file: {lis_file_path}")

            l_um = parse_l_um_from_path(lis_file_path)
            l_feature = build_l_feature(l_um)
            per_l_features.append(features)
            per_l_labels.append(labels)
            l_features.append(l_feature)
            min_samples = features.shape[0] if min_samples is None else min(min_samples, features.shape[0])
            print(f"Queued joint L input: L={l_um:g} um, feature={l_feature:g}")

        if not per_l_features:
            return None, None

        if min_samples is None or min_samples <= 0:
            return None, None

        if any(features.shape[0] != min_samples for features in per_l_features):
            print(
                "Warning: L files produced different sample counts; "
                f"truncating all to {min_samples} aligned samples."
            )

        aligned_features = [features[:min_samples] for features in per_l_features]
        aligned_labels = [labels[:min_samples] for labels in per_l_labels]
        joint_features = np.concatenate(aligned_features, axis=1).astype(np.float32)

        if getattr(config, "include_l_feature", False):
            l_row = np.asarray(l_features, dtype=np.float32).reshape(1, -1)
            l_array = np.repeat(l_row, min_samples, axis=0)
            joint_features = np.concatenate([joint_features, l_array], axis=1).astype(np.float32)

        first_labels = aligned_labels[0]
        for labels in aligned_labels[1:]:
            if not np.allclose(first_labels, labels, rtol=1e-4, atol=1e-8):
                print(
                    "Warning: labels differ across L files for the same sample index; "
                    "using labels from the first L file."
                )
                break

        print(
            "Built joint-L dataset: "
            f"samples={joint_features.shape[0]}, flat_dim={joint_features.shape[1]}, "
            f"num_l={len(per_l_features)}"
        )
        return joint_features, first_labels.astype(np.float32)


def main(lis_file_path, output_dir: Path):
    print(f"Start parsing .lis file(s): {lis_file_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = HspiceLisParser(output_params_list=config.output_params)
    if isinstance(lis_file_path, (list, tuple)):
        features, labels = parser.parse_files(lis_file_path)
    else:
        features, labels = parser.parse_file(lis_file_path)

    if features is not None:
        np.save(output_dir / "features.npy", features)
        np.save(output_dir / "labels.npy", labels)
        print(f"\nSaved parsed arrays to {output_dir}")


def convert(
    features_path="data/processed/features.npy",
    labels_path="data/processed/labels.npy",
    out_path="data/processed/converted_dataset.npz",
):
    """Convert parser outputs to the (ivcv, params) format used by training."""
    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Loaded: features {features.shape}, labels {labels.shape}")

    assert features.shape[0] == labels.shape[0], "Sample count mismatch."

    if features.ndim == 3:
        features_transposed = np.transpose(features, (0, 2, 1))
        ivcv = features_transposed.reshape(features_transposed.shape[0], -1).astype(np.float32)
        print(f"Flattened features from {features.shape} to {ivcv.shape}")
    else:
        ivcv = features.astype(np.float32)
        print(f"Features are already 2D: {ivcv.shape}")

    params = labels.astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, ivcv=ivcv, params=params)
    print(f"Saved converted dataset to {out_path}")


if __name__ == "__main__":
    L_FILE_PATH = config.INPUT_LIS
    NPY_OUTPUT_DIR = Path("data/processed")
    main(L_FILE_PATH, NPY_OUTPUT_DIR)
    convert()
