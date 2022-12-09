from torch import nn
class TransformationModel(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super(TransformationModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 输入的个数，输出的个数

    def forward(self, x):
        out = self.linear(x)
        return out

trans = TransformationModel(1, 1)
epochs = 100
optimizer = torch.optim.SGD(trans.parameters(), lr=0.001)

class TransformationModel(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super(TransformationModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 输入的个数，输出的个数

    def forward(self, x):
        out = self.linear(x)
        return out

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        print("X",x,y)
        mse_loss = torch.mean(torch.pow((x - y), 2))
        return mse_loss

model_loss = CustomLoss()

cca = CCA(n_components=1)
# 训练数据
X = ref_train_data
Y = train_data
cca.fit(X, Y)
# print(X)
X_train_r, Y_train_r = cca.transform(X, Y)
# print(X_train_r)
print('noisy20  epoch 1 vs epoch200 ')
print(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]) #输出相关系数
# 开始训练模型
for epoch in range(epochs):
    epoch += 1
    # 注意转行成tensor

    
    inputs = torch.from_numpy(data_provider.train_representation(199))
    labels = torch.from_numpy(data_provider.train_representation(199))
    print(inputs,labels)
    
    # 梯度要清零每一次迭代
    optimizer.zero_grad()
    # 前向传播
    outputs: torch.Tensor = trans(inputs)
    # 计算损失
    loss = model_loss(outputs, labels)
    # 返向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))
