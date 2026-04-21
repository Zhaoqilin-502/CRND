import torch
import torch.nn as nn
import torchvision


class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
from thop import profile
if __name__ == '__main__':
    if __name__ == '__main__':
        # 配置参数
        n_feats = 256
        input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)

        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = deformable_LKA_Attention(n_feats).to(device)
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
