import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *
from ultralytics.utils.torch_utils import make_divisible

__all__ = ['EMBC2f']


class EMBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = eSE(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc
    
    def forward(self, x):
        return x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))

class EMBC2f(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EMBConv(self.c, self.c, shortcut) for _ in range(n))
