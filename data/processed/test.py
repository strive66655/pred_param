import numpy as np

data = np.load('F:\pred_param\data\processed\\features.npy', allow_pickle=True)  # 记得改成你的文件路径
print(type(data))
print(data.shape)
print(data)
