import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel, G=8):       #G为分组数
        super().__init__()
        self.G = G#分组数
        self.channel = channel#通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))#分组归一化
        #通道注意力可学习参数，初始化为0和1
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        # 空间注意力可学习参数，初始化为0和1
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):#通道清洗可以理解为通道之间信息的融合以及冗余信息的剔除
        b, c, h, w = x.shape
        # 将通道分为groups组
        x = x.reshape(b, groups, -1, h, w)# shape: [b, g, c/g, h, w]
        x = x.permute(0, 2, 1, 3, 4) # shape: [b, c/g, g, h, w]

        # 扁平化
        x = x.reshape(b, -1, h, w) # shape: [b, c, h, w]

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # 将通道分成子特征，将输入通道分为G组，每组c//G个通道

        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w
        # 沿通道维度将每组分为两部分
        # 通道分割
        x_0, x_1 = x.chunk(2, dim=1)  # 每个部分的shape: [b*G, c/(2G), h, w]

        # 通道注意力
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # 空间注意力
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # 沿通道轴拼接
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # 通道混洗
        out = self.channel_shuffle(out, 2)
        return out
