import torch.nn as nn
from ..utils.clones import clones
from ..layers.normalization import LayerNorm, SublayerConnection

class DecoderLayer(nn.Module):
    """
    解码器子层

    每个子层包括 self_attn, src_attn 和 FFN
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        初始化函数

        参数：
            size: 特征维度大小
            self_attn: 自注意力层
            src_attn: 交叉注意力层
            feed_forward: FFN 层
            dropout: dropout 比例
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, X, memory, src_mask, tgt_mask):
        """
        前向传播过程

        参数：
            X: 输入数据
            memory: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码

        返回：
            返回解码器子层输出
        """
        m = memory
        X = self.sublayer[0](X, lambda X: self.self_attn(X, X, X, tgt_mask))
        if self.src_attn is not None:
            X = self.sublayer[1](X, lambda X: self.src_attn(X, m, m, src_mask))
        return self.sublayer[2](X, self.feed_forward)

class Decoder(nn.Module):
    """
    解码器

    包括 N 层解码器层
    """
    def __init__(self, layer, N):
        """
        初始化函数

        参数：
            layer: 解码器层
            N: 层数
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, X, memory, src_mask, tgt_mask):
        """
        前向传播过程

        参数：
            X: 输入数据
            memory: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码

        返回：
            解码器输出
        """
        for layer in self.layers:
            X = layer(X, memory, src_mask, tgt_mask)
        return self.norm(X)
