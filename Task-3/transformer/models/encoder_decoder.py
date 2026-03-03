import copy
import torch.nn as nn
from ..core.encoder import Encoder, EncoderLayer
from ..core.decoder import Decoder, DecoderLayer
from ..core.generator import Generator
from ..layers.embedding import Embedding
from ..layers.positional import PositionalEncoding
from ..layers.attention import MultiHeadedAttention
from ..layers.feedforward import PositionwiseFeedForward
from .base import BaseModeL

class EncoderDecoder(BaseModeL):
    """
    Encoder-Decoder 架构（原始 Transformer）
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        初始化函数

        参数：
            encoder: 编码器类
            decoder: 解码器类
            src_embed: 源语言嵌入层
            tgt_embed: 目标语言嵌入层
            generator: 生成器
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播过程

        参数：
            src: 源序列
            tgt: 目标序列
            src_mask: 源序列掩码（为了实现 Padding）
            tgt_mask: 目标序列掩码

        返回：
            返回解码器输出
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        编码过程

        参数：
            src: 源序列
            src_mask: 源序列掩码

        返回：
            编码器输出
        """
        return self.encode(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码过程

        参数：
            memory: 编码器的输出
            src_mask: 源序列掩码（防止 Padding 字符影响生成）
            tgt: 目标序列
            tgt_mask: 目标序列掩码
        返回：
            解码器输出
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
               d_ff=2048, h=8, dropout=0.1):
    """
    创建 Encoder-Decoder 模型
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model