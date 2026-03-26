import numpy as np
import pandas as pd
from pathlib import Path

data = np.load(Path(__file__).resolve().with_name('converted_dataset.npz'))

# 查看特征 (ivcv) 的前 5 行
df_features = pd.DataFrame(data['ivcv'])
print("特征数据 (ivcv) 前 5 行:")
print(df_features.head())

# 查看标签 (params) 的前 5 行
df_labels = pd.DataFrame(data['params'], columns=['VTH0', 'VOFF', 'NFACTOR', 'K1', 'K2', 'U0', 'UA', 'UB', 'UC', 'RDSW', 'AGS', 'A0', 'KETA']) # 假设是这三个参数
print("\n标签数据 (params) 前 5 行:")
print(df_labels.head())
