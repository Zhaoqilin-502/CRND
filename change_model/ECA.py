import torch
from torch import nn
from torch.nn import init
from thop import profile


# 定义ECA注意力模块的类
class ECA(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变       stride默认为1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)#保持图像大小不变
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重


    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出.实际为1，256，1，1

        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，1，256，1 为1D卷积准备，变为bs,1,c.实际为1，1，256

        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出,1,1,256

        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重

        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度.1,256,1,1
        
        y = y.expand_as(x)#广播机制实现形状匹配1,256,20,20

        return x * y  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

