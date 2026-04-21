import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
        # 初始化一个平均池化层，设置不包含padding区域的统计
    def forward(self, x):
        # return self.avgpool(x)
        x_exp = torch.exp(x)# 对输入x逐元素做指数运算，放大重要区域

        x_exp_pool = self.avgpool(x_exp)# 对指数化后的特征做平均池化
        x = self.avgpool(x_exp*x) # 对加权后的特征（x_exp*x）做平均池化

        return x/x_exp_pool# 返回归一化结果，相当于加权平均池化
class LIA(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        f = f # 中间层通道数缩减比例，默认16
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1), # 1x1卷积降维，减少计算量,降维通道数到16---》f
            SoftPooling2D(7, stride=3),# 自定义软池化，kernel=7, stride=3大幅下采样  图像大小由20-》5
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),# 3x3卷积进一步下采样图像大小5--》3
            nn.Conv2d(f, channels, 3, padding=1),# 恢复原始通道数，3x3卷积融合局部特征图像，大小不变
            # to heatmap
            nn.Sigmoid(),# 输出[0,1]的注意力权重
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),# 门控信号，将输入映射到[0,1]
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map

        g = self.gate(x[:,:1]) # 取输入的第一个通道生成门控信号，

        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 生成注意力权重图并插值回原尺寸
        return x * w * g# 原始特征与权重、门控逐元素相乘

# 示例使用
