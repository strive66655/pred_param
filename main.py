import torch


# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "best_model.pth"
normalize_meta = "norm_meta.json"

# 超参数
batch_size = 64
lr = 1e-4
epochs = 200
patience = 15
hidden_size = 1024 # 8000
hiddenlayer_nums = 3
dropout = 0.0

