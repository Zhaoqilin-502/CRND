import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DWConv(nn.Module):
    def __init__(self, dim=1028):
        super(DWConv, self).__init__() ## 输入/输出通道数相同，并且通过k=3，p=s=1保存图像尺寸不变
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)# 深度卷积：每个通道独立卷积

    def forward(self, x, H, W): # 输入x形状: (B, N, C)，其中 N = H*W（序列长度）
        B, N, C = x.shape
        # 将序列数据还原为2D图像格式
        """
        x = x.transpose(1, 2)       # (B, C, N)交换第1维和第2维数据，维度索引从0开始
        x = x.view(B, C, H, W)      # (B, C, H, W)
        x = x.contiguous()          # 确保内存连续
        """
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        #此时x返回的数据形式(B, C, H, W)
        x = self.dwconv(x)#执行深度卷积
        #此时x返回的数据形式(B, C, H, W)，因此下面要执行转换为(B, N, C)的代码

        """
        将结果转换回序列格式
        x = x.flatten(2)            # (B, C, H*W) = (B, C, N)，在第2维也就是H，W合并为为H*W
        具体用法是
            x = torch.randn(4, 3, 28, 28)  # 形状 (4, 3, 28, 28)
            # 从第 2 维（height）开始展平到最后一维（width）
            x_flattened = x.flatten(2)      # 形状变为 (4, 3, 28 * 28=784)
        x = x.transpose(1, 2)       # (B, N, C)
        """
        x = x.flatten(2).transpose(1, 2)
        return x
#因为对应线性层，所以需要将四位数据经过一定处理变为三维数据可以减少参数提升性能

class ConvolutionalGLU(nn.Module):# 输入/输出通道数相同
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输出通道数默认与输入相同
        hidden_features = hidden_features or in_features # 隐藏层通道数默认与输入相同
        hidden_features = int(2 * hidden_features / 3)  # 调整隐藏层大小为输入的 2/3（GLU 设计）
        self.fc1 = nn.Linear(in_features, hidden_features * 2) # 线性层，输出双倍隐藏层大小
        self.dwconv = DWConv(hidden_features) # 深度可分离卷积
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) #恢复输出维度
        self.drop = nn.Dropout(drop) # Dropout 层

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)#数据维度转化(B, C, H*W)——————————(B, H*W, C)即(B,N,C)对应上述深度卷积
        x, v = self.fc1(x).chunk(2, dim=-1) #在最后一个维度将维度分离为两个分支，各为 (B, H*W, hidden_features)
        #通过 fc1 将输入通道数扩展为 hidden_features * 2，随后分割为两部分（x 和 v），通过门控乘法增强非线性。
        x = self.act(self.dwconv(x, H, W)) * v#深度卷积 + 激活 → 门控乘法
        x = self.drop(x)
        x = self.fc2(x)#恢复输出通道数 → (B, H*W, out_features)
        x = self.drop(x)#共享drop层
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

if __name__ == '__main__':
    # 配置参数
    n_feats = 256
    input_shape = (1, 256, 20, 20)  # (batch, channels, height, width)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvolutionalGLU(n_feats).to(device)
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
