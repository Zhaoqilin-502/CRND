import torch
from torch import nn
import math
class StdPool(nn.Module):            #标准差归一化
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)# 计算每个通道的标准差
        std = std.reshape(b, c, 1, 1) # 重塑为(b,c,1,1)的张量
        return std #返回(batch_size, channels, 1, 1) 的标准差张量，表示每个通道的空间维度标准差。


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()

        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        # 1D卷积处理序列数据
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        # 可学习的池化权重
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        # 步骤1：多类型池化
        feats = [pool(x) for pool in self.pools]# 获取各池化结果
        # 步骤2：动态权重融合
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)  # 约束权重到(0,1)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
        #步骤3：维度置换 + 卷积处理
        out = out.permute(0, 3, 2, 1).contiguous() # (b,c,1,1) -> (b,1,1,c)
        #out.permute(0, 3, 2, 1) 的作用是重新排列张量的维度顺序。
        #(a, b, c, d)（对应维度0到3），经过此操作后，新张量的形状将变为 (a, d, c, b)。
        out = self.conv(out)# 在序列维度执行1D卷积
        out = out.permute(0, 3, 2, 1).contiguous()# 恢复原始维度顺序 (b,1,1,c)  -> (b,c,1,1)

        out = self.sigmoid(out)# 激活到(0,1)范围
        out = out.expand_as(x) # 应用注意力掩码

        return x * out


class MCA1(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs a MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCA1, self).__init__()
        # 自动计算卷积核大小
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1 # 保证为奇数

        self.h_cw = MCAGate(3)#3*3卷积核大小
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)

    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()

        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)

            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)

        return x_out

from thop import profile
if __name__ == '__main__':


    n_feats = 256
    input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MCA1(n_feats).to(device)
    model.eval()  # 设置为评估模式
    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)
    output = model(dummy_input)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算计算量（FLOPs）
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 打印结果
    print(f"输入尺寸: {input_shape}")
    print(f"总参数量: {total_params / 1e6:.2f} Million")
    print(f"理论计算量: {flops / 1e9:.2f} GFLOPs")
    print(f"实际内存占用: {params / 1e6:.2f} MB")
    print(f"输出尺寸：{output.size()}")