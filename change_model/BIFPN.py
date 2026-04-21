import math

import numpy as np
import torch
import torch.nn as nn


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)

class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension

        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        print(weight)
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


if __name__ == "__main__":
    # 正确构造输入列表（包含2个张量）
    x1 = torch.randn(1, 16, 64, 64)  # 通道16
    x2 = torch.randn(1, 32, 64, 64)  # 通道32
    inputs = [x1, x2]  # 直接构造列表

    model = BiFPN_Concat2(dimension=1)  # 按通道维度拼接

    try:
        output = model(inputs)
        print("测试通过！输出维度:", output.shape)  # 预期 torch.Size([1, 48, 64, 64])
        print("权重分布:", model.w)  # 查看初始权重
    except Exception as e:
        print("测试失败！错误信息:")
        print(e)