# BSIM Parameter Extractor (PyTorch)

## Overview

This project trains a neural network to regress BSIM model parameters from HSPICE Monte Carlo I-V curves.

Current target parameters:

- `VTH0`
- `VOFF`
- `NFACTOR`
- `K1`
- `K2`
- `U0`
- `UA`
- `UB`
- `UC`
- `RDSW`
- `AGS`
- `A0`
- `KETA`

## Data Pipeline

The dataset is built from `bsim_datasets/mc.lis`.

1. `bsim_datasets/data_parser.py`
   Parses Monte Carlo blocks from the `.lis` file.
2. Extracts 10 `Id-Vg` curves corresponding to:
   `Vd = [0.1, 0.2, ..., 1.0]`
3. Each curve contains 51 `Vg` points.
4. The final raw input shape is:
   `10 * 51 = 510`
5. Parsed arrays are saved to:
   `data/processed/features.npy`
   `data/processed/labels.npy`
6. The converter writes:
   `data/processed/converted_dataset.npz`

## Preprocessing

Implemented in [bsim_iv_dataset.py](/f:/pred_param/bsim_datasets/bsim_iv_dataset.py).

- Input current features are clipped with `clip_min_current` before `log10`.
- Input features are Min-Max normalized.
- Output parameters are normalized with Z-score.
- If `pca_enabled=True`, PCA is fit on the training split only.
- Validation data reuses the training normalization and PCA metadata.

This avoids validation leakage and keeps train/validation features in the same space.

## Model

Configured in [config.py](/f:/pred_param/bsim_datasets/config.py).

Two model paths are supported:

- `mlp`: plain multi-layer perceptron
- `residual_mlp`: linear head + residual MLP blocks

Current default:

- `model_type = "residual_mlp"`
- `pca_enabled = False` by default in config
- `pca_n_components = 30` when PCA is enabled

Residual MLP is implemented in [residual_param_extractor.py](/f:/pred_param/models/residual_param_extractor.py).

## Training Flow

Training entry:

- [train_iv_extractor.py](/f:/pred_param/train/train_iv_extractor.py)

Current flow:

1. Load `data/processed/converted_dataset.npz`
2. Split raw samples into train/validation sets
3. Build `BSIMIVDataset` for training
4. Reuse training normalization/PCA metadata for validation
5. Build model from `config.model_type`
6. Train with `AdamW + MSELoss`
7. Optionally reduce LR with `ReduceLROnPlateau`
8. Optionally stop early with `early_stopping`
9. Save:
   `experiments/<exp_name>/config.json`
   `experiments/<exp_name>/models/best_iv_extractor.pth`
   `experiments/<exp_name>/models/iv_norm_meta.json`
   `experiments/<exp_name>/plots/loss_curve.png`
   `experiments/<exp_name>/plots/pred_vs_true.png`

## Run

Generate processed data:

```bash
python bsim_datasets/data_parser.py
```

Train:

```bash
python train/train_iv_extractor.py
```

## Notes

- `config.normalization` is currently informational only; the implemented behavior is fixed to Min-Max for inputs and Z-score for labels.
- Experiment outputs are written under `experiments/`.
