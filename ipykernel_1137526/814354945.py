cca = CCA(n_components=1)
# 训练数据
cca.fit(X, Y2)
# print(X)
X_train_r, Y_train_r = cca.transform(X, Y2)
# print(X_train_r)

print(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]) #输出相关系数
