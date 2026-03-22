import copy
import torch.nn as nn
from ..core.encoder import Encoder, EncoderLayer
from ..core.generator import Generator
from ..layers.embedding import Embedding
from ..layers.positional import PositionalEncoding
from ..layers.attention import MultiHeadedAttention
from ..layers.feedforward import PositionwiseFeedForward
from .base import BaseModeL

class EncoderOnly(BaseModeL):
    """
    Encoder-Only 架构（如 BERT1）
    """
    def __init__(self, encoder, embed, output_layer=None):
        """
        初始化函数

        参数：
            decoder: 解码器类
            embed: 嵌入层
            output_layer: 生成层
        """
        super(EncoderOnly, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.output_layer = output_layer

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
        X = self.encoder(X, mask)

        if self.output_layer is not None:
            X = self.output_layer(X)

        return X

def make_encoder_only_model(vocab_size, N=12, d_model=768,
                            d_ff=3072, h=12, dropout=0.1,
                            task = 'mlm', num_classes=2,
                            use_relative_position=False, max_relative_position=127):
    """
    创建 Encoder-Only 模型
    
    参数:
        vocab_size: 词表大小
        N: 编码器层数
        d_model: 模型维度
        d_ff: 前馈网络维度
        h: 注意力头数
        dropout: dropout 比例
        task: 任务类型 ('mlm' 或 'classification')
        num_classes: 分类任务类别数
        use_relative_position: 是否使用相对位置编码
        max_relative_position: 最大相对距离
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, use_relative_position, max_relative_position)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 如果使用相对位置编码，不再使用绝对位置编码
    if use_relative_position:
        embed = nn.Sequential(Embedding(d_model, vocab_size), nn.Dropout(p=dropout))
    else:
        position = PositionalEncoding(d_model, dropout)
        embed = nn.Sequential(Embedding(d_model, vocab_size), c(position))

    if task == 'mlm':
        output_layer = nn.Linear(d_model, vocab_size)
    elif task == 'classification':
        output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes)
        )
    else:
        output_layer = None

    model = EncoderOnly(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        embed,
        output_layer
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model