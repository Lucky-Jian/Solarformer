import torch
import torch.nn as nn
from sklearn.svm import SVR


class SVRModel(nn.Module):
    def __init__(self, input_size):
        super(SVRModel, self).__init__()
        self.input_size = input_size
        self.svr = SVR()

    def forward(self, x):
        # 将输入形状变为二维（batch_size * seq_len, input_size）
        x = x.view(-1, self.input_size)

        # 使用SVR进行训练和预测
        output = self.svr.fit(x, y).predict(x)

        # 将输出形状变回三维（batch_size, seq_len, output_size）
        output = output.view(-1, x.size(1), self.output_size)

        return output


# 创建一个输入张量，形状为 (batch_size, seq_len, input_size)
input_tensor = torch.randn(32, 10, 64)

# 创建SVR模型实例
svr_model = SVRModel(input_size=64)

# 将输入张量传递给SVR模型进行前向计算
output_tensor = svr_model(input_tensor)

# 打印输出张量的形状
print(output_tensor.shape)
