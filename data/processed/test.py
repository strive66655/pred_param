import numpy as np
import pandas as pd

data = np.load(r'F:\pred_param\data\processed\converted_dataset.npz')

# 查看特征 (ivcv) 的前 5 行
df_features = pd.DataFrame(data['ivcv'])
print("特征数据 (ivcv) 前 5 行:")
print(df_features.head())

# 查看标签 (params) 的前 5 行
df_labels = pd.DataFrame(data['params'], columns=['VTH0', 'U0', 'AGS', 'VSAT', 'UB', 'VOFF', 'NFACTOR', 'A0', 'UA']) # 假设是这三个参数
print("\n标签数据 (params) 前 5 行:")
print(df_labels.head())