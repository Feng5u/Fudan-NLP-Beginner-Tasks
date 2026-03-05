import copy
import torch.nn as nn
from ..core.decoder import Decoder, DecoderLayer
from ..core.generator import Generator
from ..layers.embedding import Embedding
from ..layers.positional import PositionalEncoding
from ..layers.attention import MultiHeadedAttention
from ..layers.feedforward import PositionwiseFeedForward
from .base import BaseModeL

class DecoderOnly(BaseModeL):
    """
    Decoder-Only 架构（如 GPT）
    """
    def __init__(self, decoder, embed, generator):
        """
        初始化函数

        参数：
            decoder: 解码器类
            embed: 嵌入层
            generator: 生成器
        """
        super(DecoderOnly, self).__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator

    def forward(self, X, mask):
        """
        前向传播过程

        参数：
            x: 输入数据
            mask: 掩码

        返回：
            返回解码器输出
        """
        X = self.embed(X)
        X = self.decoder(X, None, None, mask)
        return self.generator(X)

def make_decoder_only_model(vocab_size, N=12, d_model=768,
                            d_ff=3072, h=12, dropout=0.1):
    """
    创建 Decoder-Only 模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = DecoderOnly(
        Decoder(DecoderLayer(d_model, c(attn), None, c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model