import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DWConv(nn.Module):
    def __init__(self, dim=1028):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)#

    def forward(self, x, H, W): #
        B, N, C = x.shape

        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        #此时x返回的数据形式(B, C, H, W)
        x = self.dwconv(x)#执行深度卷积
        #此时x返回的数据形式(B, C, H, W)，因此下面要执行转换为(B, N, C)的代码

        x = x.flatten(2).transpose(1, 2)
        return x


class ConvolutionalGLU(nn.Module):# 输入/输出通道数相同
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输出通道数默认与输入相同
        hidden_features = hidden_features or in_features # 隐藏层通道数默认与输入相同
        hidden_features = int(2 * hidden_features / 3)  # 调整隐藏层大小为输入的 2/3（GLU 设计）
        self.fc1 = nn.Linear(in_features, hidden_features * 2) # 线性层，输出双倍隐藏层大小
        self.dwconv = DWConv(hidden_features) # 深度可分离卷积
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) #恢复输出维度
        self.drop = nn.Dropout(drop) # Dropout 层

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)#数据维度转化(B, C, H*W)——————————(B, H*W, C)即(B,N,C)对应上述深度卷积
        x, v = self.fc1(x).chunk(2, dim=-1) #在最后一个维度将维度分离为两个分支，各为 (B, H*W, hidden_features)
        #通过 fc1 将输入通道数扩展为 hidden_features * 2，随后分割为两部分（x 和 v），通过门控乘法增强非线性。
        x = self.act(self.dwconv(x, H, W)) * v#深度卷积 + 激活 → 门控乘法
        x = self.drop(x)
        x = self.fc2(x)#恢复输出通道数 → (B, H*W, out_features)
        x = self.drop(x)#共享drop层
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

