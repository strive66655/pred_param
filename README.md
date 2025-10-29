# BSIM Parameter Extractor (PyTorch)

项目包含:
- 数据集格式与示例合成数据生成
- ParamExtractorNet模型定义
- 训练脚本与工具
- 推理脚本

快速开始:
1. 生成合成数据:
```
python datasets/generate_synthetic.py
```
2. 训练:
```
python train/train.py --data data/processed/demo_data.npz --out_dir experiments/exp_demo
```
3. 推理:
```
python inference/predict_params.py --model experiments/exp_demo/best_model.pth --sample data/processed/demo_data.npz --out pred.txt
```
