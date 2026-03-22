import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    
    使用可学习的相对位置偏置，在注意力计算中使用。
    基于 Shaw 等人在 "Self-Attention with Relative Position Representations" 中的方法。
    """
    def __init__(self, d_model, max_relative_position=127):
        """
        参数:
            d_model: 模型维度
            max_relative_position: 最大相对距离（对称，实际范围是 -max_relative_position 到 +max_relative_position）
        """
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 相对位置偏置矩阵 (2 * max_relative_position + 1, d_model)
        # 索引 0 对应 -max_relative_position，中间对应 0
        vocab_size = max_relative_position * 2 + 1
        self.relative_embeddings = nn.Embedding(vocab_size, d_model)
        
        # 初始化相对位置嵌入
        nn.init.xavier_uniform_(self.relative_embeddings.weight)
    
    def forward(self, length_q, length_k):
        """
        生成相对位置偏置矩阵
        
        参数:
            length_q: query 序列长度
            length_k: key 序列长度
            
        返回:
            relative_bias: (length_q, length_k, d_model) 相对位置偏置
        """
        # 生成相对位置索引矩阵
        # q_pos: [0, 1, 2, ..., length_q-1]
        # k_pos: [0, 1, 2, ..., length_k-1]
        q_pos = torch.arange(length_q, device=self.relative_embeddings.weight.device)
        k_pos = torch.arange(length_k, device=self.relative_embeddings.weight.device)
        
        # 计算相对位置：q_pos - k_pos
        # shape: (length_q, length_k)
        relative_positions = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
        
        # 将相对位置截断到 [-max_relative_position, max_relative_position] 范围
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 将相对位置映射到 [0, 2*max_relative_position]
        # 这样可以作为 embedding 的索引
        relative_positions = relative_positions + self.max_relative_position
        
        # 获取相对位置嵌入
        # shape: (length_q, length_k, d_model)
        relative_bias = self.relative_embeddings(relative_positions)
        
        return relative_bias