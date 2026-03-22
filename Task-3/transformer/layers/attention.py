import math
import torch
import torch.nn as nn
from ..utils.clones import clones


def attention(query, key, value, mask=None, dropout=None, relative_bias=None):
    """
    计算注意力
    
    参数:
        query: (batch, heads, len_q, d_k)
        key: (batch, heads, len_k, d_k)
        value: (batch, heads, len_v, d_v)
        mask: 可选的掩码
        dropout: 可选的 dropout
        relative_bias: 可选的相对位置偏置 (len_q, len_k, d_model)
    
    返回:
        输出和注意力权重
    """
    d_k = query.size(-1)
    h = query.size(1)
    
    # 基础注意力分数：query * key^T
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 如果有相对位置偏置，添加到分数中
    if relative_bias is not None:
        # relative_bias: (len_q, len_k, d_model)
        # query: (batch, heads, len_q, d_k), where d_k = d_model / h
        # 需要将 d_model 分配到各个头
        batch, heads, len_q, _ = query.shape
        len_k = key.size(2)
        
        # 将 relative_bias 按 d_k 分割到各个头
        # shape: (len_q, len_k, heads, d_k)
        relative_bias_heads = relative_bias.view(len_q, len_k, h, d_k)
        
        # 将 heads 维度移到前面，并添加 batch 维度
        # shape: (1, heads, len_q, len_k, d_k)
        relative_bias_heads = relative_bias_heads.permute(2, 0, 1, 3).unsqueeze(0)
        
        # 计算相对位置分数：query 与相对位置嵌入的点积
        # query: (batch, heads, len_q, d_k)
        # relative_bias_heads: (1, heads, len_q, len_k, d_k)
        # 需要将 query 扩展到 (batch, heads, len_q, 1, d_k) 以便广播
        # result: (batch, heads, len_q, len_k)
        relative_scores = torch.einsum('bhqd,hqkd->bhqk', query, relative_bias_heads.squeeze(0))
        scores = scores + relative_scores
    
    if mask is not None:
        mask = mask.bool() if mask.dtype != torch.bool else mask
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力类
    
    支持相对位置编码
    """
    def __init__(self, h, d_model, dropout=0.1, use_relative_position=False, max_relative_position=127):
        """
        初始化函数

        参数：
            h: 注意力头数
            d_model: 词向量维度
            dropout: dropout 比例
            use_relative_position: 是否使用相对位置编码
            max_relative_position: 最大相对距离
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.use_relative_position = use_relative_position
        
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        # 相对位置编码
        if self.use_relative_position:
            from .positional import RelativePositionalEncoding
            self.relative_position = RelativePositionalEncoding(d_model, max_relative_position)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播过程

        参数：
            query: 查询矩阵
            key: 键矩阵
            value: 值矩阵
            mask: 掩码

        返回：
            返回多头注意力值
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 如果使用相对位置编码，生成相对位置偏置
        relative_bias = None
        if self.use_relative_position:
            # query 的长度
            len_q = query.size(2)
            # key 的长度
            len_k = key.size(2)
            relative_bias = self.relative_position(len_q, len_k)
        
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout, relative_bias=relative_bias
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)