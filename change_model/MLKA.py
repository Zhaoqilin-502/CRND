import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
class LayerNorm(nn.Module):#自定义归一化层
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":#PyTorch 原生实现归一化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":#（NCHW）
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 2 != 0:#因为分成2个尺度并行计算
            raise ValueError("n_feats must be divisible by 2 for MLKA.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')#层归一化
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)#缩放参数

        self.LKA7 = nn.Sequential(                #2条通道分支，每条通道平分1/2的通道数
            nn.Conv2d(n_feats // 2, n_feats // 2, 7, 1, 7 // 2, groups=n_feats // 2),
            nn.Conv2d(n_feats // 2, n_feats // 2, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 2, dilation=4),
            nn.Conv2d(n_feats // 2, n_feats // 2, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 2, n_feats // 2, 5, 1, 5 // 2, groups=n_feats // 2),
            nn.Conv2d(n_feats // 2, n_feats // 2, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 2, dilation=3),
            nn.Conv2d(n_feats // 2, n_feats // 2, 1, 1, 0))

        self.X5 = nn.Conv2d(n_feats // 2, n_feats // 2, 5, 1, 5 // 2, groups=n_feats // 2)
        self.X7 = nn.Conv2d(n_feats // 2, n_feats // 2, 7, 1, 7 // 2, groups=n_feats // 2)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)

        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_2, a_3 = torch.chunk(a, 2, dim=1)

        a = torch.cat([ self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],dim=1)
        #使用contact模块，进行通道上的拼接C/2+C/2=C,通道数有C/2恢复为C。
        x = self.proj_last(x * a) * self.scale + shortcut#
        return x
class MLKA1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        return x
