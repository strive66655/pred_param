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
            "vth0_stamos_val": "VTH0",
            "voff_stamos_val": "VOFF",
            "nfactor_stamos_val": "NFACTOR",
            "k1_stamos_val": "K1",
            "k2_stamos_val": "K2",
            "u0_stamos_val": "U0",
            "ua_stamos_val": "UA",
            "ub_stamos_val": "UB",
            "uc_stamos_val": "UC",
            "rdsw_stamos_val": "RDSW",
            "ags_stamos_val": "AGS",
            "a0_stamos_val": "A0",
            "keta_stamos_val": "KETA",
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

    def parse_files(self, lis_file_paths):
        all_records = []

        for lis_file_path in lis_file_paths:
            lis_file_path = Path(lis_file_path)
            print(f"Reading .lis file: {lis_file_path}")
            try:
                content = lis_file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = lis_file_path.read_text(encoding="latin1")
            all_records.extend(self._parse_records(content))

        if not all_records:
            return None, None
        return self._merge_vb_records(all_records)


def main(lis_file_path, output_dir: Path):
    print(f"Start parsing .lis file(s): {lis_file_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = HspiceLisParser(output_params_list=config.output_params)
    if isinstance(lis_file_path, (list, tuple)):
        features, labels = parser.parse_files(lis_file_path)
    else:
        lis_file_path = Path(lis_file_path)
        try:
            content = lis_file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = lis_file_path.read_text(encoding="latin1")
        features, labels = parser.parse(content)

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
