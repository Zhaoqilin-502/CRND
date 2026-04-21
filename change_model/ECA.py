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


# 示例使用
if __name__ == '__main__':
    # 配置参数
    n_feats = 256
    input_shape = (1,256,20,20)  # (batch, channels, height, width)
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECA().to(device)       # ！！！！！！！！ECA模块不需要输入通道数量
    model.eval()  # 设置为评估模式
    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 计算计算量（FLOPs）
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    output=model(dummy_input)
    # 打印结果
    print(f"输入尺寸: {input_shape}")
    print(f"总参数量: {total_params :.2f} Million")
    print(f"理论计算量: {flops :.2f} GFLOPs")
    print(f"实际内存占用: {params :.2f} MB")
    print(f"输出尺寸：{output.size()}")
