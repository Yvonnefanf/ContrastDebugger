from torch import nn


class TransformationModel(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super(TransformationModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 输入的个数，输出的个数

    def forward(self, x):
        out = self.linear(x)
        return out


