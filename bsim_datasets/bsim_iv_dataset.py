import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

if __package__ in (None, ""):
    from config import config
else:
    from .config import config


class BSIMIVDataset(Dataset):
    def __init__(self, iv_data, params, norm_meta=None, save_meta_path=None):
        """
        BSIM parameter extraction dataset.
        Expected raw input shape: [N, num_curves * vg_points]
        """
        assert iv_data.shape[0] == params.shape[0], "Sample count mismatch."

        self.iv_data = iv_data.astype(np.float32)
        self.params = params.astype(np.float32)
        self.save_meta_path = save_meta_path

        self._build_input_features()

        if norm_meta is None:
            self.norm_meta = self._compute_norm_meta()
            if self.save_meta_path:
                self._save_norm_meta(self.save_meta_path)
        else:
            self.norm_meta = norm_meta

        self._apply_norm()

    def _build_input_features(self):
        """Build model inputs and flatten them for the MLP."""
        expected_dim = getattr(config, "raw_input_dim", config.num_curves * config.vg_points)
        current_dim = self.iv_data.shape[1]

        if current_dim != expected_dim:
            print(
                "Warning: raw feature dim "
                f"({current_dim}) does not match expected dim ({expected_dim})."
            )

        raw_iv = self.iv_data.reshape(-1, config.num_curves, config.vg_points)
        feature_blocks = []
        feature_names = []

        if getattr(config, "include_raw_id", True):
            feature_blocks.append(raw_iv.copy())
            feature_names.append("raw_id")

        if getattr(config, "include_log_id", False):
            total_count = int(raw_iv.size)
            nonpositive_count = int(np.count_nonzero(raw_iv <= 0))
            if nonpositive_count:
                nonpositive_ratio = nonpositive_count / total_count if total_count else 0.0
                print(
                    "Non-positive currents before log transform: "
                    f"{nonpositive_count}/{total_count} "
                    f"({nonpositive_ratio:.2%})"
                )

            clipped_mask = raw_iv < config.clip_min_current
            clipped_count = int(np.count_nonzero(clipped_mask))
            clipped_ratio = clipped_count / total_count if total_count else 0.0
            print(
                "Log feature clipping: "
                f"threshold={config.clip_min_current:.1e}, "
                f"clipped={clipped_count}/{total_count} "
                f"({clipped_ratio:.2%})"
            )

            log_iv = np.clip(raw_iv, a_min=config.clip_min_current, a_max=None)
            log_iv = np.log10(log_iv)
            feature_blocks.append(log_iv.astype(np.float32))
            feature_names.append("log_id")
        else:
            log_iv = None

        if getattr(config, "include_gm_id", False):
            gm_iv = np.gradient(raw_iv, axis=2)
            feature_blocks.append(gm_iv.astype(np.float32))
            feature_names.append("gm_id")

        if getattr(config, "include_log_gm", False):
            if log_iv is None:
                log_iv = np.clip(raw_iv, a_min=config.clip_min_current, a_max=None)
                log_iv = np.log10(log_iv)
            log_gm_iv = np.gradient(log_iv, axis=2)
            feature_blocks.append(log_gm_iv.astype(np.float32))
            feature_names.append("dlog_id_dvg")

        if getattr(config, "include_log_curvature", False):
            if log_iv is None:
                log_iv = np.clip(raw_iv, a_min=config.clip_min_current, a_max=None)
                log_iv = np.log10(log_iv)
            log_curvature_iv = np.gradient(np.gradient(log_iv, axis=2), axis=2)
            feature_blocks.append(log_curvature_iv.astype(np.float32))
            feature_names.append("d2log_id_dvg2")

        if not feature_blocks:
            raise ValueError("At least one IV feature block must be enabled.")

        structured_iv = np.stack(feature_blocks, axis=2).astype(np.float32)
        self.structured_iv_data = structured_iv
        self.iv_data = structured_iv.reshape(structured_iv.shape[0], -1)
        print(
            "Input features ready, structured_shape="
            f"{self.structured_iv_data.shape[1:]}, flat_dim={self.iv_data.shape[1]}"
        )
        print(f"Enabled feature blocks: {feature_names}")

    def _compute_norm_meta(self):
        """Compute normalization statistics."""
        normalization = getattr(config, "normalization", "minmax").lower()
        if normalization == "minmax":
            return {
                "normalization": normalization,
                "iv_min": self.iv_data.min(axis=0).tolist(),
                "iv_max": self.iv_data.max(axis=0).tolist(),
                "params_min": self.params.min(axis=0).tolist(),
                "params_max": self.params.max(axis=0).tolist(),
            }
        if normalization in ("zscore", "z-score"):
            return {
                "normalization": "zscore",
                "iv_mean": self.iv_data.mean(axis=0).tolist(),
                "iv_std": self.iv_data.std(axis=0).tolist(),
                "params_mean": self.params.mean(axis=0).tolist(),
                "params_std": self.params.std(axis=0).tolist(),
            }
        raise ValueError(f"Unsupported normalization: {normalization}")

    def _apply_norm(self):
        """Apply configured normalization to both inputs and outputs."""
        normalization = self.norm_meta.get("normalization", "minmax").lower()
        if normalization == "minmax":
            iv_min = np.array(self.norm_meta["iv_min"], dtype=np.float32)
            iv_max = np.array(self.norm_meta["iv_max"], dtype=np.float32)
            iv_range = iv_max - iv_min
            iv_range[iv_range == 0] = 1.0
            self.iv_data = (self.iv_data - iv_min) / iv_range

            p_min = np.array(self.norm_meta["params_min"], dtype=np.float32)
            p_max = np.array(self.norm_meta["params_max"], dtype=np.float32)
            p_range = p_max - p_min
            p_range[p_range == 0] = 1.0
            self.params = (self.params - p_min) / p_range
            return

        if normalization in ("zscore", "z-score"):
            iv_mean = np.array(self.norm_meta["iv_mean"], dtype=np.float32)
            iv_std = np.array(self.norm_meta["iv_std"], dtype=np.float32)
            iv_std[iv_std == 0] = 1.0
            self.iv_data = (self.iv_data - iv_mean) / iv_std

            p_mean = np.array(self.norm_meta["params_mean"], dtype=np.float32)
            p_std = np.array(self.norm_meta["params_std"], dtype=np.float32)
            p_std[p_std == 0] = 1.0
            self.params = (self.params - p_mean) / p_std
            return

        raise ValueError(f"Unsupported normalization: {normalization}")

    def inverse_transform_params(self, normalized_params):
        """Restore normalized output parameters to the original scale."""
        normalized_params = np.asarray(normalized_params, dtype=np.float32)
        normalization = self.norm_meta.get("normalization", "minmax").lower()
        if normalization == "minmax":
            p_min = np.array(self.norm_meta["params_min"], dtype=np.float32)
            p_max = np.array(self.norm_meta["params_max"], dtype=np.float32)
            return normalized_params * (p_max - p_min) + p_min
        if normalization in ("zscore", "z-score"):
            p_mean = np.array(self.norm_meta["params_mean"], dtype=np.float32)
            p_std = np.array(self.norm_meta["params_std"], dtype=np.float32)
            return normalized_params * p_std + p_mean
        raise ValueError(f"Unsupported normalization: {normalization}")

    def _save_norm_meta(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.norm_meta, f, indent=2)
        print(f"Normalization meta saved to {path}")

    def __len__(self):
        return self.iv_data.shape[0]

    def __getitem__(self, idx):
        return {
            "iv": torch.from_numpy(self.iv_data[idx]),
            "params": torch.from_numpy(self.params[idx]),
        }
