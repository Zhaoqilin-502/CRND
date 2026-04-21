import torch
from torch import nn

from thop import profile
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F



class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))# 全局平均池化到1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))# 高度方向池化（保持高度，宽度为1）
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))# 宽度方向池化（保持高度，高度为1）
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)# 分组归一化
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  #将x分组 b*g,c//g,h,w
        x_h = self.pool_h(group_x) # 高度池化：[b*g, c//g, h, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)# 宽度池化并转置：[b*g, c//g, 1, w]->[b*g, c//g, w, 1]
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# 拼接后卷积处理：[b*g, c//g, h+w, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)# 分割回高度和宽度部分
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) # 调整宽度维度顺序，# 空间权重调整
        x2 = self.conv3x3(group_x)  # 3x3卷积处理分组特征。。空间通道协同注意
        # 分支1：x1的全局通道权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))# [b*g, 1, c/g]
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 分支2：x2的全局通道权重
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))# [b*g, 1, c/g]
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w) # 融合注意力权重 # [b*g, 1, h*w]
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)# 应用权重并恢复原始形状

