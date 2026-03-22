import copy
import torch.nn as nn

def clones(module, N):
    """
    复制器

    产生 N 个相同的层

    参数：
        module: 待复制的层
        N: 复制个数

    返回：
        N 个相同的 module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])