
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class SPPF1(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        self.w = nn.Parameter(torch.ones(7, dtype=torch.float32), requires_grad=True)#对七个通道进行加权融合
        self.epsilon = 0.0001
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 7, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.av=nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        y = [self.cv1(x)]
        y1=[self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))#进行了三次最大池化
        y1.extend(self.av(y1[-1]) for _ in range(3))#进行了三次平均池化
        y2 = [weight[0] * y[0], weight[1] * y[1], weight[2] * y[2],weight[3] * y[3],
              weight[4] * y1[1],weight[5] * y1[2],weight[6] * y1[3]]#使用通道加权对这7个通道带上权重
        #y.extend(y1[1:])#平均池化和最大池化分为两路并行计算


        return self.cv2(torch.cat(y2, 1))#在通道维度进行拼接
if __name__ == "__main__":
    # 参数设置
    c1, c2, k = 64, 32, 5
    batch, H, W = 1, 224, 224

    # 创建模块
    model = SPPF1(c1, c2, k)

    # 生成随机输入 (batch_size, channels, height, width)
    x = torch.randn(batch, c1, H, W)

    # 前向传播
    try:
        output = model(x)
        print("测试通过！输出维度:", output.shape)
        # 预期输出: torch.Size([1, 32, 224, 224])
    except Exception as e:
        print("测试失败！错误信息:")
        print(e)