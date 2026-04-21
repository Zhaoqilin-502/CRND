import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
def channel_normalize(x, eps=1e-5):
    # 输入形状：(N, C, H, W)
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)

class Ours4(nn.Module):
    def __init__(self, n_feats, f=16):
        super().__init__()
        f = f #
        self.Conv2 = nn.Conv2d(n_feats, 2*n_feats, 1, 1, 0)  # 。
        self.Conv1 = nn.Conv2d(n_feats, 1, 1, 1, 0)  #
        # self.DWConv1 = nn.Conv2d(n_feats//2, n_feats//2, 7, 1, 7 // 2, groups=n_feats//2)  #
        self.DWConv2 = nn.Conv2d(n_feats//2, n_feats//2, 5, 1, 5 // 2, groups=1)  # 。
        self.DWConv3 = nn.Conv2d(n_feats//2, n_feats//2, 3, 1, 3 // 2, groups=1)  #
        # self.Conv3 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):#1,256,20,20
        shortcut=x.clone()

        g=self.Conv1(x)
        g = self.gate(g) #

        x=channel_normalize(x)

        x=self.Conv2(x)#
        x,a=torch.chunk(x, 2, dim=1)#

        a_2, a_3 = torch.chunk(a, 2, dim=1)  #
        a = torch.cat([self.DWConv2(a_2) , self.DWConv3(a_3) ], dim=1)#1,
        w= self.gate(a)#
        return x * w * g+shortcut#


