import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
from thop import profile
if __name__ == '__main__':
    # 配置参数
    n_feats = 256
    input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBAM(n_feats).to(device)
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