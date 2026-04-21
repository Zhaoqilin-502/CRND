import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
class LayerNorm(nn.Module):#自定义归一化层
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):#channels_last：通道维度在最后（如NHWC格式）。
                                                                              # channels_first：通道维度在前（如NCHW格式）。
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

        x = self.proj_first(x)#扩展通道至2倍（n_feats → 2n_feats），分割为注意力分支和特征分支(N, 2C, H, W)
        a, x = torch.chunk(x, 2, dim=1)#通道分割：将扩展后的特征沿通道维度均分为两部分 各为(N, C, H, W)
        a_2, a_3 = torch.chunk(a, 2, dim=1)#将a分为2等份（每份C/2通道）
        # 各分支处理流程：
        # 1. LKA序列处理（大核+扩张卷积）
        # 2. 与标准深度卷积结果相乘（特征增强）
        a = torch.cat([ self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],dim=1)
        #使用contact模块，进行通道上的拼接C/2+C/2=C,通道数有C/2恢复为C。
        """
        a = torch.cat([
        self.LKA3(a_1) * self.X3(a_1),  # 小尺度分支
        self.LKA5(a_2) * self.X5(a_2),  # 中尺度分支
        self.LKA7(a_3) * self.X7(a_3)   # 大尺度分支
        ], dim=1)  # 合并后形状：(N, C, H, W)
        """
        #此处使用直接相加，
        x = self.proj_last(x * a) * self.scale + shortcut#    x * a :特征分支与注意力分支相乘proj_last恢复通道数，整合加权特征。以及缩放，残差操作
        return x
class MLKA1(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        return x
from thop import profile
if __name__ == '__main__':
    if __name__ == '__main__':
        # 配置参数
        n_feats = 256
        input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)

        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLKA1(n_feats).to(device)
        model.eval()  # 设置为评估模式

        # 创建随机输入
        dummy_input = torch.randn(input_shape).to(device)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 估算模型参数占用内存 (MB)
        mem_MB = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        # 计算计算量（FLOPs）
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 模型输出
        output = model(dummy_input)

        # 打印结果
        print(f"输入尺寸: {input_shape}")
        print(f"总参数量: {params :.2f} Million")  # 百万参数
        print(f"理论计算量: {flops:.2f} FLOPs")
        print(f"参数占用内存: {mem_MB:.2f} MB")
        print(f"输出尺寸: {output.size()}")
