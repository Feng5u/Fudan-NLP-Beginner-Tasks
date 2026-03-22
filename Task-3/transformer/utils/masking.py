import torch

def subsequent_mask(size):
    """
    产生对未来序列的掩码

    参数：
        size: 目标序列长度

    返回：
        返回一个下三角掩码矩阵
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def padding_mask(seq, pad_idx=0):
    """
    产生 padding 掩码

    参数：
        seq: 序列
        pad_idx: padding token 序号

    返回：
        返回经过 padding 掩码的序列
    """
    return (seq != pad_idx).unsqueeze(-2)