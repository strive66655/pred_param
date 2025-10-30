import os
import numpy as np
import json
from data_parser import (
    parse_hspice_mc_data,
    prepare_deep_learning_data,
    normalize_monte_carlo_data,
)

SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)


curves, params = parse_hspice_mc_data("bsim_datasets/mc.lis")
x_data, y_data = prepare_deep_learning_data(curves, params)
x_norm, y_norm, norm_stats = normalize_monte_carlo_data(curves, params)
np.savez_compressed(os.path.join(SAVE_DIR, "converted_dataset.npz"),
                    ivcv=x_norm, params=y_norm)
with open(os.path.join(SAVE_DIR, "norm_stats.json"), "w") as f:
    json.dump({
        "volt": norm_stats["volt"],
        "i_linear": norm_stats["i_linear"],
        "i_sat": norm_stats["i_sat"],
        "params_mean": norm_stats["params"]["mean"].tolist(),
        "params_std": norm_stats["params"]["std"].tolist(),
    }, f, indent=2)

print("数据已保存：")
print(f"→ {os.path.join(SAVE_DIR, 'converted_dataset.npz')}")
print(f"→ {os.path.join(SAVE_DIR, 'norm_stats.json')}")
