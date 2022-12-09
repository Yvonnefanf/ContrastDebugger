import numpy as np
from sklearn.cross_decomposition import CCA
# 建立模型
X = ref_train_data
Y = train_data
cca = CCA(n_components=1)
# 训练数据
cca.fit(X, Y)
# print(X)
X_train_r, Y_train_r = cca.transform(X, Y)
# print(X_train_r)
print('noisy20  epoch 1 vs epoch200 ')
print(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]) #输出相关系数