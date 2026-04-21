import torch
import torch.nn as nn
from thop import profile

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6#通过ReLU6实现Sigmoid的近似计算，提高轻量化网络的效率


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)#实现Swish激活函数的轻量化版本，用于替代计算量较大的Sigmoid


class CoordAtt(nn.Module):
    def __init__(self, inp, oup=1, reduction=32):
        super(CoordAtt, self).__init__()
        oup=inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))# 沿高度方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))# 沿宽度方向池化
        mip = max(8, inp // reduction) # 计算中间层通道数，reduction=32将输入通道压缩到1/32（但最小保留8通道），减少计算量。
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)# 输出形状: (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)# 输出形状: (n,c,1,w)----->(n, c, w, 1)
        y = torch.cat([x_h, x_w], dim=2)# 拼接后形状: (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # 分解回高度和宽度注意力
        x_h, x_w = torch.split(y, [h, w], dim=2)# 分割为h和w部分
        x_w = x_w.permute(0, 1, 3, 2)# 恢复宽度维度形状
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

if __name__ == '__main__':
    # 配置参数
    n_feats = 256
    input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoordAtt(n_feats).to(device)
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
