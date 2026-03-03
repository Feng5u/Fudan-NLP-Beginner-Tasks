import torch.nn as nn
from ..utils.clones import clones
from ..layers.normalization import LayerNorm, SublayerConnection

class EncoderLayer(nn.Module):
    """
    编码器子层

    每个子层包括一个 self-attn 和一个 FFN
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        初始化函数

        参数：
            size: 特征维度大小
            self_attn: 自注意力层
            feed_forward: FFN 层
            dropout: dropout 比例
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, X, mask):
        """
        前向传播过程

        参数：
            X: 输入数据
            mask: 掩码

        返回：
            返回一个编码器层的输出
        """
        X = self.sublayer[0](X, lambda Z: self.self_attn(X, X, X, mask))
        return self.sublayer[1](X, self.feed_forward)

class Encoder(nn.Module):
    """
    编码器

    包括 N 层编码器层
    """
    def __init__(self, layer, N):
        """
        初始化函数

        参数：
            layer: 编码器层
            N: 层数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, X, mask):
        """
        前向传播

        参数：
            X: 输入数据
            mask: 掩码
        """
        for layer in self.layers:
            X = layer(X, mask)
        return self.norm(X)
