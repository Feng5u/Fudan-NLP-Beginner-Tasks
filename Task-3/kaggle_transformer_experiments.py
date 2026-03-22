"""
Kaggle 可运行的 Transformer 实现文件
包含所有模块：数据处理、模型定义、训练和评估
支持两个任务：
- Task 1: 多位数加法
- Task 2: 语言建模
"""

import os
import json
import copy
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset

# ===============================
# Kaggle 数据集路径配置
# ===============================
# 获取当前文件所在目录的父目录，然后构建数据集路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_ADDITION_PATH = os.path.join(_BASE_DIR, "Data", "dataset", "addition.txt")


# ===============================
# Transformer 核心工具
# ===============================

def clones(module, N):
    """复制 N 个相同的层"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """产生对未来序列的掩码"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def padding_mask(seq, pad_idx=0):
    """产生 padding 掩码"""
    return (seq != pad_idx).unsqueeze(-2)


# ===============================
# Transformer 层实现
# ===============================

class LayerNorm(nn.Module):
    """归一化层"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, X):
        mean = X.mean(-1, keepdim=True)
        std = X.std(-1, keepdim=True)
        return self.a_2 * (X - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """子层连接：残差连接 + 归一化"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, sublayer):
        return X + self.dropout(sublayer(self.norm(X)))


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
    def __init__(self, h, d_model, dropout=0.1, use_relative_position=True, max_relative_position=127):
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
            from transformer.layers.positional import RelativePositionalEncoding
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


class PositionwiseFeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        return self.w_2(self.dropout(self.w_1(X).relu()))


class Embedding(nn.Module):
    """词嵌入层"""
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, X):
        return self.lut(X) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# ===============================
# Transformer 核心模块
# ===============================

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, X, mask):
        X = self.sublayer[0](X, lambda Z: self.self_attn(Z, Z, Z, mask))
        return self.sublayer[1](X, self.feed_forward)


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, X, mask):
        for layer in self.layers:
            X = layer(X, mask)
        return self.norm(X)


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, X, memory, src_mask, tgt_mask):
        m = memory
        X = self.sublayer[0](X, lambda X: self.self_attn(X, X, X, tgt_mask))
        if self.src_attn is not None:
            X = self.sublayer[1](X, lambda X: self.src_attn(X, m, m, src_mask))
        return self.sublayer[2](X, self.feed_forward)


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, X, memory, src_mask, tgt_mask):
        for layer in self.layers:
            X = layer(X, memory, src_mask, tgt_mask)
        return self.norm(X)


class Generator(nn.Module):
    """生成器"""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, X):
        return self.proj(X)


# ===============================
# Transformer 模型
# ===============================

class BaseModeL(nn.Module):
    """模型基类"""
    def __init__(self):
        super(BaseModeL, self).__init__()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class EncoderDecoder(BaseModeL):
    """Encoder-Decoder 架构（原始 Transformer）"""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, use_relative_position=True, max_relative_position=127):
    """
    创建 Encoder-Decoder 模型
    
    参数:
        src_vocab: 源语言词汇表大小
        tgt_vocab: 目标语言词汇表大小
        N: 编码器和解码器的层数
        d_model: 模型维度
        d_ff: 前馈网络维度
        h: 注意力头数
        dropout: dropout 比例
        use_relative_position: 是否使用相对位置编码（默认 True）
        max_relative_position: 最大相对距离（默认 127）
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, use_relative_position, max_relative_position)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 使用相对位置编码，不再使用绝对位置编码
    src_embed = nn.Sequential(Embedding(d_model, src_vocab), nn.Dropout(p=dropout))
    tgt_embed = nn.Sequential(Embedding(d_model, tgt_vocab), nn.Dropout(p=dropout))

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed,
        tgt_embed,
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class DecoderOnly(BaseModeL):
    """Decoder-Only 架构（如 GPT）"""
    def __init__(self, decoder, embed, generator):
        super(DecoderOnly, self).__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator

    def forward(self, X, mask):
        X = self.embed(X)
        X = self.decoder(X, None, None, mask)
        return self.generator(X)


def make_decoder_only_model(vocab_size, N=12, d_model=768, d_ff=3072, h=12, dropout=0.1, use_relative_position=True, max_relative_position=127):
    """
    创建 Decoder-Only 模型
    
    参数:
        vocab_size: 词表大小
        N: 解码器层数
        d_model: 模型维度
        d_ff: 前馈网络维度
        h: 注意力头数
        dropout: dropout 比例
        use_relative_position: 是否使用相对位置编码（默认 True）
        max_relative_position: 最大相对距离（默认 127）
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, use_relative_position, max_relative_position)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 使用相对位置编码，不再使用绝对位置编码
    embed = nn.Sequential(Embedding(d_model, vocab_size), nn.Dropout(p=dropout))

    model = DecoderOnly(
        Decoder(DecoderLayer(d_model, c(attn), None, c(ff), dropout), N),
        embed,
        Generator(d_model, vocab_size)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class EncoderOnly(BaseModeL):
    """Encoder-Only 架构（如 BERT）"""
    def __init__(self, encoder, embed, output_layer=None):
        super(EncoderOnly, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.output_layer = output_layer

    def forward(self, X, mask):
        X = self.embed(X)
        X = self.encoder(X, mask)
        if self.output_layer is not None:
            X = self.output_layer(X)
        return X


def make_encoder_only_model(vocab_size, N=12, d_model=768, d_ff=3072, h=12, dropout=0.1, task='mlm', num_classes=2, use_relative_position=True, max_relative_position=127):
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
        use_relative_position: 是否使用相对位置编码（默认 True）
        max_relative_position: 最大相对距离（默认 127）
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, use_relative_position, max_relative_position)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 使用相对位置编码，不再使用绝对位置编码
    embed = nn.Sequential(Embedding(d_model, vocab_size), nn.Dropout(p=dropout))

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


# ===============================
# Task 1: 多位数加法数据集
# ===============================

class AdditionDataset(Dataset):
    """多位数加法数据集"""
    
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    
    DIGITS = [str(i) for i in range(10)]
    OPERATORS = ['+', '=']
    
    VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + DIGITS + OPERATORS
    
    TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
    ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
    
    PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
    SOS_ID = TOKEN_TO_ID[SOS_TOKEN]
    EOS_ID = TOKEN_TO_ID[EOS_TOKEN]
    UNK_ID = TOKEN_TO_ID[UNK_TOKEN]
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()
        self.data = self._validate_data()
        self._print_stats()
    
    def _load_data(self):
        """从文本文件加载数据"""
        data = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if '+' in line and '=' in line:
                        num1, rest = line.split('+')
                        num2, result = rest.split('=')
                        
                        if num1.isdigit() and num2.isdigit() and result.isdigit():
                            data.append({
                                'id': line_num,
                                'expr': line,
                                'input': f"{num1}+{num2}=",
                                'target': result,
                                'num1': int(num1),
                                'num2': int(num2),
                                'result': int(result),
                                'digits1': len(num1),
                                'digits2': len(num2)
                            })
                except Exception as e:
                    print(f"警告: 第{line_num}行解析失败: {line}, 错误: {e}")
        
        return data
    
    def _validate_data(self):
        """验证数据有效性"""
        valid_data = []
        
        for item in self.data:
            if len(str(item['num1'])) != item['digits1']:
                continue
            if len(str(item['num2'])) != item['digits2']:
                continue
            if item['num1'] + item['num2'] != item['result']:
                continue
            valid_data.append(item)
        
        if len(valid_data) < len(self.data):
            print(f"数据验证: 移除了 {len(self.data) - len(valid_data)} 条无效数据")
        
        return valid_data
    
    def _print_stats(self):
        """打印数据统计信息"""
        if not self.data:
            return

        digits1 = [item['digits1'] for item in self.data]
        digits2 = [item['digits2'] for item in self.data]
        results = [item['result'] for item in self.data]
        
        pair_counts = {}
        for item in self.data:
            pair = (item['digits1'], item['digits2'])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {len(self.data)}")
        print(f"  数字位数范围: {min(digits1)}-{max(digits1)} + {min(digits2)}-{max(digits2)}")
        print(f"  结果范围: {min(results)} - {max(results)}")
        print(f"  组合分布:")
        for pair, count in sorted(pair_counts.items()):
            print(f"    {pair[0]}+{pair[1]}: {count} 条 ({count/len(self.data)*100:.1f}%)")
    
    def encode(self, text):
        """将文本编码为 ID 序列"""
        ids = [self.SOS_ID]
        for char in text:
            if char in self.TOKEN_TO_ID:
                ids.append(self.TOKEN_TO_ID[char])
            else:
                ids.append(self.UNK_ID)
        ids.append(self.EOS_ID)
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """将ID序列解码为文本"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        chars = []
        for id_ in ids:
            token = self.ID_TO_TOKEN[id_]
            if skip_special_tokens and token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                continue
            chars.append(token)
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """返回一个样本"""
        item = self.data[idx]
        
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'expr': item['expr'],
            'num1': item['num1'],
            'num2': item['num2'],
            'result': item['result'],
            'digits1': item['digits1'],
            'digits2': item['digits2']
        }
    
    @classmethod
    def get_vocab_size(cls):
        return len(cls.VOCAB)
    
    @classmethod
    def get_pad_id(cls):
        return cls.PAD_ID


# ===============================
# Task 1: 数据处理器
# ===============================

class BaseProcessor:
    """处理器基类"""
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.name = "base_processor"
    
    def split(self):
        """划分数据集"""
        raise NotImplementedError
    
    def get_train_dataset(self):
        train_idx, _, _ = self.split()
        return Subset(self.dataset, train_idx)
    
    def get_val_dataset(self):
        _, val_idx, _ = self.split()
        return Subset(self.dataset, val_idx)
    
    def get_test_dataset(self):
        _, _, test_idx = self.split()
        return Subset(self.dataset, test_idx)
    
    def get_train_loader(self, batch_size, shuffle=True, collate_fn=None, num_workers=0, pin_memory=False):
        dataset = self.get_train_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def get_val_loader(self, batch_size, shuffle=False, collate_fn=None, num_workers=0, pin_memory=False):
        dataset = self.get_val_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, persistent_workers=num_workers > 0)

    def get_test_loader(self, batch_size, shuffle=False, collate_fn=None, num_workers=0, pin_memory=False):
        dataset = self.get_test_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, num_workers=num_workers,
                         pin_memory=pin_memory, persistent_workers=num_workers > 0)
    
    def get_vocab_size(self):
        return self.dataset.get_vocab_size()
    
    def get_stats(self):
        train_idx, val_idx, test_idx = self.split()
        return {
            'processor_name': self.name,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_size': len(self.dataset)
        }


class AdditionProcessor(BaseProcessor):
    """加法数据处理器"""
    
    VALID_STRATEGIES = ['random', 'digit_pair', 'max_digits', 'result_range', 'carry_complexity']
    
    def __init__(self, dataset, config):
        super().__init__(dataset, config)
        self.split_strategy = config.get('split_strategy', 'random')
        if self.split_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"无效的划分策略: {self.split_strategy}，可选: {self.VALID_STRATEGIES}")
        self.name = f"{self.split_strategy}"
        self.seed = config.get('seed', 42)
        self._cached_split = None
        print(f"创建处理器: {self.name}")
    
    def split(self):
        """根据选择的策略进行划分"""
        if self._cached_split is not None:
            return self._cached_split
        
        if self.split_strategy == 'random':
            result = self._random_split()
        elif self.split_strategy == 'digit_pair':
            result = self._digit_pair_split()
        elif self.split_strategy == 'max_digits':
            result = self._max_digits_split()
        elif self.split_strategy == 'result_range':
            result = self._result_range_split()
        elif self.split_strategy == 'carry_complexity':
            result = self._carry_complexity_split()
        else:
            raise ValueError(f"未知策略: {self.split_strategy}")
        
        self._cached_split = result
        return result
    
    def _random_split(self):
        """随机划分"""
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        total_size = len(self.dataset)
        indices = list(range(total_size))
        
        random.seed(self.seed)
        random.shuffle(indices)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"随机划分: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def _digit_pair_split(self):
        train_pairs = self.config.get('train_pairs', [(3,3), (3,4), (4,3)])
        val_pairs = self.config.get('val_pairs', [(3,5), (5,3)])
        test_pairs = self.config.get('test_pairs', [(4,4)])
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        train_set = set(train_pairs)
        val_set = set(val_pairs)
        test_set = set(test_pairs)
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            pair = (item['digits1'], item['digits2'])
            
            if pair in train_set:
                train_indices.append(idx)
            elif pair in val_set:
                val_indices.append(idx)
            elif pair in test_set:
                test_indices.append(idx)
        
        if len(train_indices) == 0:
            print(f"警告: 训练组合 {train_pairs} 没有对应样本")
        if len(val_indices) == 0:
            print(f"警告: 验证组合 {val_pairs} 没有对应样本")
        if len(test_indices) == 0:
            print(f"警告: 测试组合 {test_pairs} 没有对应样本")
        
        print(f"数字组合划分:")
        print(f"  训练组合 {train_pairs}: {len(train_indices)} 样本")
        print(f"  验证组合 {val_pairs}: {len(val_indices)} 样本")
        print(f"  测试组合 {test_pairs}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _max_digits_split(self):
        train_digits = self.config.get('train_max_digits', [3])
        val_digits = self.config.get('val_max_digits', [4])
        test_digits = self.config.get('test_max_digits', [5])
        
        train_set = set(train_digits)
        val_set = set(val_digits)
        test_set = set(test_digits)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            max_d = max(item['digits1'], item['digits2'])
            
            if max_d in train_set:
                train_indices.append(idx)
            elif max_d in val_set:
                val_indices.append(idx)
            elif max_d in test_set:
                test_indices.append(idx)
        
        if len(train_indices) == 0:
            print(f"警告: 训练位数 {train_digits} 没有对应样本")
        if len(val_indices) == 0:
            print(f"警告: 验证位数 {val_digits} 没有对应样本")
        if len(test_indices) == 0:
            print(f"警告: 测试位数 {test_digits} 没有对应样本")
        
        print(f"最大位数划分:")
        print(f"  训练位数 {train_digits}: {len(train_indices)} 样本")
        print(f"  验证位数 {val_digits}: {len(val_indices)} 样本")
        print(f"  测试位数 {test_digits}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _result_range_split(self):
        train_range = self.config.get('train_range', (0, 1000))
        val_range = self.config.get('val_range', (1001, 5000))
        test_range = self.config.get('test_range', (5001, 20000))
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            result = item['result']
            
            if train_range[0] <= result <= train_range[1]:
                train_indices.append(idx)
            elif val_range[0] <= result <= val_range[1]:
                val_indices.append(idx)
            elif test_range[0] <= result <= test_range[1]:
                test_indices.append(idx)
        
        if len(train_indices) == 0:
            print(f"警告: 训练范围 {train_range} 没有对应样本")
        if len(val_indices) == 0:
            print(f"警告: 验证范围 {val_range} 没有对应样本")
        if len(test_indices) == 0:
            print(f"警告: 测试范围 {test_range} 没有对应样本")
        
        print(f"结果范围划分:")
        print(f"  训练范围 {train_range}: {len(train_indices)} 样本")
        print(f"  验证范围 {val_range}: {len(val_indices)} 样本")
        print(f"  测试范围 {test_range}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _carry_complexity_split(self):
        simple = []
        medium = []
        complex_carry = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            num1, num2 = item['num1'], item['num2']
            
            carry_count = self._count_carries(num1, num2)
            
            if carry_count == 0:
                simple.append(idx)
            elif carry_count <= 2:
                medium.append(idx)
            else:
                complex_carry.append(idx)
        
        random.seed(self.seed)
        random.shuffle(simple)
        random.shuffle(medium)
        random.shuffle(complex_carry)
        
        def split_list(lst, train_ratio=0.7, val_ratio=0.15):
            total = len(lst)
            train_size = int(total * train_ratio)
            val_size = int(total * val_ratio)
            return (lst[:train_size], lst[train_size:train_size+val_size], lst[train_size+val_size:])
        
        simple_train, simple_val, simple_test = split_list(simple)
        medium_train, medium_val, medium_test = split_list(medium)
        complex_train, complex_val, complex_test = split_list(complex_carry)
        
        train_indices = simple_train + medium_train + complex_train
        val_indices = simple_val + medium_val + complex_val
        test_indices = simple_test + medium_test + complex_test
        
        if len(train_indices) == 0:
            print(f"警告: 训练集为空")
        if len(val_indices) == 0:
            print(f"警告: 验证集为空")
        if len(test_indices) == 0:
            print(f"警告: 测试集为空")
        
        print(f"进位复杂度划分:")
        print(f"  无进位样本: {len(simple)}")
        print(f"  中等进位样本: {len(medium)}")
        print(f"  复杂进位样本: {len(complex_carry)}")
        print(f"  最终: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def _count_carries(self, num1, num2):
        """计算加法中的进位次数"""
        carry_count = 0
        n1, n2 = str(num1)[::-1], str(num2)[::-1]
        max_len = max(len(n1), len(n2))
        n1 = n1.ljust(max_len, '0')
        n2 = n2.ljust(max_len, '0')
        
        carry = 0
        for d1, d2 in zip(n1, n2):
            s = int(d1) + int(d2) + carry
            if s >= 10:
                carry_count += 1
                carry = 1
            else:
                carry = 0
        
        return carry_count
    
    def get_stats(self):
        """获取详细的统计信息"""
        train_idx, val_idx, test_idx = self.split()
        
        stats = {
            'processor_name': self.name,
            'split_strategy': self.split_strategy,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_size': len(self.dataset)
        }
        
        if self.split_strategy == 'random':
            stats.update({
                'train_ratio': self.config.get('train_ratio', 0.7),
                'val_ratio': self.config.get('val_ratio', 0.15),
                'seed': self.seed
            })
        
        elif self.split_strategy == 'digit_pair':
            stats.update({
                'train_pairs': self.config.get('train_pairs'),
                'val_pairs': self.config.get('val_pairs'),
                'test_pairs': self.config.get('test_pairs')
            })
        
        elif self.split_strategy == 'max_digits':
            stats.update({
                'train_digits': self.config.get('train_max_digits'),
                'val_digits': self.config.get('val_max_digits'),
                'test_digits': self.config.get('test_max_digits')
            })
        
        elif self.split_strategy == 'result_range':
            stats.update({
                'train_range': self.config.get('train_range'),
                'val_range': self.config.get('val_range'),
                'test_range': self.config.get('test_range')
            })
        
        return stats


# ===============================
# Task 1: 配置和工具函数
# ===============================

TASK1_PARAM_CONFIGS = {
    'small': {
        'N': 2,
        'd_model': 128,
        'd_ff': 512,
        'h': 4,
        'dropout': 0.1
    },
    'medium': {
        'N': 4,
        'd_model': 256,
        'd_ff': 1024,
        'h': 8,
        'dropout': 0.1
    },
    'large': {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'dropout': 0.1
    }
}

TASK1_TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'warmup_steps': 2000,
    'clip_grad': 1.0
}

TASK1_SPLIT_CONFIGS = {
    'random': {
        'split_strategy': 'random',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'seed': 42
    },
    'digit_pair': {
        'split_strategy': 'digit_pair',
        'train_pairs': [(1, 1), (1, 2), (2, 1), (2, 2)],
        'val_pairs': [(2, 3), (3, 2), (3, 3)],
        'test_pairs': [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5)]
    },
    'max_digits': {
        'split_strategy': 'max_digits',
        'train_max_digits': [3],
        'val_max_digits': [4],
        'test_max_digits': [5]
    },
    'result_range': {
        'split_strategy': 'result_range',
        'train_range': (0, 1000),
        'val_range': (1001, 5000),
        'test_range': (5001, 20000)
    },
    'carry_complexity': {
        'split_strategy': 'carry_complexity',
        'seed': 42
    }
}


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def task1_collate_fn(batch):
    """数据批次整理函数"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    max_input_len = max(len(ids) for ids in input_ids)
    padded_input = torch.full((len(input_ids), max_input_len), 0, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_input[i, :len(ids)] = ids
    
    max_target_len = max(len(ids) for ids in target_ids)
    padded_target = torch.full((len(target_ids), max_target_len), 0, dtype=torch.long)
    for i, ids in enumerate(target_ids):
        padded_target[i, :len(ids)] = ids
    
    return {
        'input_ids': padded_input,
        'target_ids': padded_target,
        'input_lengths': torch.tensor([len(ids) for ids in input_ids]),
        'target_lengths': torch.tensor([len(ids) for ids in target_ids]),
        'metadata': batch
    }


# ===============================
# Task 1: 模型创建函数
# ===============================

def create_task1_model(arch_type, vocab_size, param_config):
    """创建指定架构的模型"""
    N = param_config['N']
    d_model = param_config['d_model']
    d_ff = param_config['d_ff']
    h = param_config['h']
    dropout = param_config['dropout']
    
    if arch_type == 'encoder_decoder':
        model = make_model(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        )
    elif arch_type == 'decoder_only':
        model = make_decoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        )
    elif arch_type == 'encoder_only':
        model = make_encoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            task='mlm'
        )
    else:
        raise ValueError(f"未知架构类型: {arch_type}")
    
    return model


# ===============================
# Task 1: 实验类
# ===============================

class Task1Experiment:
    """实验类：管理完整的实验流程"""
    
    def __init__(self, name, config, device='cpu'):
        self.name = name
        self.config = config
        self.device = device

        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'results', 'task1', name
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.dataset = None
        self.processor = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_data(self):
        """设置数据集和处理器"""
        print(f"\n[{self.name}] 设置数据...")
        
        data_path = self.config.get('data_path', KAGGLE_ADDITION_PATH)
        self.dataset = AdditionDataset(data_path)
        
        split_strategy = self.config.get('split_strategy', 'random')
        split_config = TASK1_SPLIT_CONFIGS[split_strategy].copy()
        split_config['seed'] = self.config.get('seed', 42)
        
        self.processor = AdditionProcessor(self.dataset, split_config)
        
        stats = self.processor.get_stats()
        print(f"数据统计: {json.dumps(stats, indent=2)}")
        
        batch_size = TASK1_TRAIN_CONFIG['batch_size']
        
        self.train_loader = self.processor.get_train_loader(
            batch_size, shuffle=True, collate_fn=task1_collate_fn,
            num_workers=0, pin_memory=False
        )
        self.val_loader = self.processor.get_val_loader(
            batch_size, shuffle=False, collate_fn=task1_collate_fn,
            num_workers=0, pin_memory=False
        )
        self.test_loader = self.processor.get_test_loader(
            batch_size, shuffle=False, collate_fn=task1_collate_fn,
            num_workers=0, pin_memory=False
        )
        
        return stats
    
    def setup_model(self):
        """设置模型"""
        print(f"\n[{self.name}] 设置模型...")
        
        arch_type = self.config['arch_type']
        param_scale = self.config['param_scale']
        param_config = TASK1_PARAM_CONFIGS[param_scale]
        vocab_size = self.dataset.get_vocab_size()
        
        print(f"架构: {arch_type}, 参数规模: {param_scale}")
        print(f"词表大小: {vocab_size}")
        
        self.model = create_task1_model(arch_type, vocab_size, param_config)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TASK1_TRAIN_CONFIG['learning_rate']
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.get_pad_id())
        
        model_config = {
            'arch_type': arch_type,
            'param_scale': param_scale,
            'vocab_size': vocab_size,
            'param_config': param_config,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        
        with open(os.path.join(self.save_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        return model_config
    
    def make_src_mask(self, src, pad_idx):
        return padding_mask(src, pad_idx)
    
    def make_tgt_mask(self, tgt, pad_idx):
        tgt_mask = padding_mask(tgt, pad_idx)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).to(tgt.device)
        return tgt_mask
    
    def forward_pass(self, batch, training=True):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        pad_idx = self.dataset.get_pad_id()
        
        if self.config['arch_type'] == 'encoder_decoder':
            src = input_ids
            tgt = target_ids[:, :-1]
            
            src_mask = self.make_src_mask(src, pad_idx)
            tgt_mask = self.make_tgt_mask(tgt, pad_idx)
            
            outputs = self.model(src, tgt, src_mask, tgt_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'decoder_only':
            src = input_ids
            tgt = target_ids[:, :-1]
            
            combined = torch.cat([src, tgt], dim=1)
            mask = self.make_tgt_mask(combined, pad_idx)
            
            outputs = self.model(combined, mask)
            outputs = outputs[:, src.size(1):, :]
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'encoder_only':
            src = input_ids
            src_mask = self.make_src_mask(src, pad_idx)
            
            outputs = self.model(src, src_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, :-1].reshape(-1)
        
        return outputs, targets
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            self.optimizer.zero_grad()
            
            outputs, targets = self.forward_pass(batch, training=True)
            
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TASK1_TRAIN_CONFIG['clip_grad']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            mask = targets != self.dataset.get_pad_id()
            correct += (predicted[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # 检查数据加载器是否为空
        if len(data_loader) == 0:
            print(f"警告: 数据加载器为空，返回默认值")
            return 0.0, 0.0, {}

        digit_acc = {}

        with torch.no_grad():
            for batch in data_loader:
                outputs, targets = self.forward_pass(batch, training=False)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                mask = targets != self.dataset.get_pad_id()
                
                batch_correct = (predicted[mask] == targets[mask]).sum().item()
                batch_total = mask.sum().item()
                
                correct += batch_correct
                total += batch_total
                
                for idx, metadata in enumerate(batch['metadata']):
                    digits1 = metadata['digits1']
                    digits2 = metadata['digits2']
                    key = f"{digits1}+{digits2}"
                    if key not in digit_acc:
                        digit_acc[key] = {'correct': 0, 'total': 0}
                    digit_acc[key]['total'] += 1
                    
                    target_lengths = batch['target_lengths']
                    current_target_len = target_lengths[idx]
                    
                    if idx == 0:
                        start_idx = 0
                    else:
                        start_idx = target_lengths[:idx].sum().item()
                    
                    end_idx = start_idx + current_target_len
                    target_mask = mask[start_idx:end_idx]
                    if target_mask.any():
                        sample_correct = (predicted[start_idx:end_idx][target_mask] == targets[start_idx:end_idx][target_mask]).sum().item()
                        total_sample = target_mask.sum().item()
                        if sample_correct == total_sample:
                            digit_acc[key]['correct'] += 1
        
        avg_loss = total_loss / len(data_loader)
        avg_acc = 100. * correct / total
        
        for key in digit_acc:
            digit_acc[key]['accuracy'] = 100. * digit_acc[key]['correct'] / digit_acc[key]['total']
        
        return avg_loss, avg_acc, digit_acc
    
    def train(self):
        """训练模型"""
        print(f"\n[{self.name}] 开始训练...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(1, TASK1_TRAIN_CONFIG['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            
            val_loss, val_acc, _ = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，停止训练 (epoch {epoch})")
                    break
        
        print(f"\n[{self.name}] 训练完成!")
        
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def test(self):
        """测试模型"""
        print(f"\n[{self.name}] 开始测试...")
        
        self.load_model('best_model.pt')
        
        test_loss, test_acc, digit_acc = self.evaluate(self.test_loader)
        
        print(f"测试结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  准确率: {test_acc:.2f}%")
        print(f"\n按位数准确率:")
        for key, stats in sorted(digit_acc.items()):
            print(f"  {key}: {stats['accuracy']:.2f}%")
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'digit_accuracy': digit_acc,
            'config': self.config
        }
        
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run(self):
        """运行完整实验"""
        print(f"\n{'='*60}")
        print(f"开始实验: {self.name}")
        print(f"{'='*60}")
        
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.setup_data()
        self.setup_model()
        self.train()
        results = self.test()
        
        print(f"\n{'='*60}")
        print(f"实验完成: {self.name}")
        print(f"{'='*60}")
        
        return results


# ===============================
# Task 1: 实验运行函数
# ===============================

def run_task1_exp1_architecture_comparison(device):
    """实验1: 架构对比"""
    print("\n" + "="*60)
    print("Task 1 - 实验 1: 架构对比")
    print("="*60)

    architectures = ['encoder_decoder', 'decoder_only', 'encoder_only']

    results = {}
    for arch in architectures:
        name = f"exp1_arch_{arch}"
        config = {
            'arch_type': arch,
            'split_strategy': 'random',
            'param_scale': 'medium',
            'seed': 42,
            'data_path': KAGGLE_ADDITION_PATH
        }

        experiment = Task1Experiment(name, config, device=device)
        results[arch] = experiment.run()
    
    summary = {
        'experiment': 'exp1_architecture_comparison',
        'results': results
    }
    
    save_path = os.path.join('results', 'task1', 'exp1_architecture_comparison_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_task1_exp2_split_strategies(device):
    """实验2: 数据划分策略泛化"""
    print("\n" + "="*60)
    print("Task 1 - 实验 2: 数据划分策略泛化")
    print("="*60)

    strategies = ['random', 'digit_pair', 'max_digits', 'result_range', 'carry_complexity']

    results = {}
    for strategy in strategies:
        name = f"exp2_split_{strategy}"
        config = {
            'arch_type': 'encoder_decoder',
            'split_strategy': strategy,
            'param_scale': 'medium',
            'seed': 42,
            'data_path': KAGGLE_ADDITION_PATH
        }

        experiment = Task1Experiment(name, config, device=device)
        results[strategy] = experiment.run()
    
    summary = {
        'experiment': 'exp2_split_strategies',
        'results': results
    }
    
    save_path = os.path.join('results', 'task1', 'exp2_split_strategies_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_task1_exp3_parameter_scales(device):
    """实验3: 参数规模影响"""
    print("\n" + "="*60)
    print("Task 1 - 实验 3: 参数规模影响")
    print("="*60)

    scales = ['small', 'medium', 'large']

    results = {}
    for scale in scales:
        name = f"exp3_scale_{scale}"
        config = {
            'arch_type': 'encoder_decoder',
            'split_strategy': 'random',
            'param_scale': scale,
            'seed': 42,
            'data_path': KAGGLE_ADDITION_PATH
        }

        experiment = Task1Experiment(name, config, device=device)
        results[scale] = experiment.run()

    summary = {
        'experiment': 'exp3_parameter_scales',
        'results': results
    }

    save_path = os.path.join('results', 'task1', 'exp3_parameter_scales_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ===============================
# Task 2: 语言建模数据集
# ===============================

class LMDataset(Dataset):
    """语言模型数据集 - 只负责加载原始文本"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.texts = self._load_data()
        print(f"[{split}] 加载了 {len(self.texts)} 条原始文本")
    
    def _load_data(self):
        """加载 WikiText-103 原始文本"""
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=self.split)
        texts = [text for text in dataset['text'] if text.strip()]
        
        num_samples = self.config.get('num_samples', None)
        if num_samples and len(texts) > num_samples:
            random.seed(42)
            texts = random.sample(texts, num_samples)
        
        return texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


# ===============================
# Task 2: 数据处理器
# ===============================

class LMProcessor:
    """语言模型数据处理器 - 负责数据处理"""
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.tokenizer = self._create_tokenizer()
        
        print(f"LMProcessor 初始化完成:")
        print(f"  - Tokenizer: {config.get('tokenizer_type')}")
        print(f"  - 词表大小: {len(self.tokenizer)}")
        print(f"  - 最大长度: {config.get('max_length')}")
    
    def _create_tokenizer(self):
        tokenizer_type = self.config.get('tokenizer_type', 'bert')
        
        model_map = {
            'bert': 'bert-base-uncased',
            'gpt2': 'gpt2',
            'roberta': 'roberta-base',
            'wordpiece': 'bert-base-uncased'
        }
        
        model_name = model_map.get(tokenizer_type)
        if model_name is None:
            raise ValueError(f"未知的 tokenizer_type: {tokenizer_type}. 可选值: {list(model_map.keys())}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"无法从预训练模型加载 tokenizer ({model_name}): {e}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add mask token if not present (needed for MLM tasks in encoder-only models)
        if tokenizer.mask_token is None:
            print(f"Tokenizer 没有 mask token，添加 <mask> token")
            tokenizer.add_tokens(['<mask>'])
            tokenizer.mask_token = '<mask>'
        
        vocab_size = self.config.get('vocab_size', None)
        if vocab_size and vocab_size < len(tokenizer):
            print(f"限制词表大小从 {len(tokenizer)} 到 {vocab_size}")
            tokenizer_vocab = list(tokenizer.get_vocab().keys())
            tokenizer_vocab = tokenizer_vocab[:vocab_size]
            tokenizer.add_tokens([token for token in tokenizer_vocab if token not in tokenizer.get_vocab()])
        
        return tokenizer
    
    def _collate_fn(self, batch_texts):
        encoded = self.tokenizer(
            batch_texts,
            truncation=True,
            max_length=self.config.get('max_length', 512),
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids']
        
        target_ids = torch.cat([
            input_ids[:, 1:],
            torch.full((input_ids.size(0), 1), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
        ], dim=1)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': encoded['attention_mask']
        }

    def get_loader(self, dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def get_vocab_size(self):
        return len(self.tokenizer)


# ===============================
# Task 2: 配置
# ===============================

TASK2_PARAM_CONFIGS = {
    'tiny': {
        'N': 2,
        'd_model': 128,
        'd_ff': 512,
        'h': 4,
        'dropout': 0.1
    },
    'small': {
        'N': 4,
        'd_model': 256,
        'd_ff': 1024,
        'h': 8,
        'dropout': 0.1
    },
    'medium': {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'dropout': 0.1
    },
    'base': {
        'N': 12,
        'd_model': 768,
        'd_ff': 3072,
        'h': 12,
        'dropout': 0.1
    }
}

TASK2_TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.0001,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'clip_grad': 1.0,
    'gradient_accumulation_steps': 1
}

TASK2_TOKENIZER_CONFIGS = {
    'gpt2': {
        'model_name': 'gpt2',
        'vocab_sizes': [50257, 25000, 10000, 5000]
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'vocab_sizes': [30522, 15000, 8000, 4000]
    },
    'roberta': {
        'model_name': 'roberta-base',
        'vocab_sizes': [50265, 25000, 10000, 5000]
    }
}

TASK2_SEQUENCE_LENGTHS = [64, 128, 256, 512]


# ===============================
# Task 2: 模型创建
# ===============================

def create_task2_model(arch_type, vocab_size, param_config):
    N = param_config['N']
    d_model = param_config['d_model']
    d_ff = param_config['d_ff']
    h = param_config['h']
    dropout = param_config['dropout']
    
    if arch_type == 'decoder_only':
        model = make_decoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        )
    elif arch_type == 'encoder_only':
        model = make_encoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            task='mlm'
        )
    elif arch_type == 'encoder_decoder':
        model = make_model(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        )
    else:
        raise ValueError(f"未知架构类型: {arch_type}")
    
    return model


# ===============================
# Task 2: 实验类
# ===============================

class LanguageModelingExperiment:
    """语言建模实验类：管理完整的实验流程"""
    
    def __init__(self, name, config, device='cpu'):
        self.name = name
        self.config = config
        self.device = device

        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'results', 'task2', name
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_data(self):
        print(f"\n[{self.name}] 设置数据...")
        
        tokenizer_type = self.config.get('tokenizer_type', 'gpt2')
        max_length = self.config.get('max_length', 128)
        vocab_size = self.config.get('vocab_size', None)
        num_samples = self.config.get('num_samples', None)
        
        dataset_config = {
            'num_samples': num_samples
        }
        
        self.train_dataset = LMDataset(dataset_config, split='train')
        self.val_dataset = LMDataset(dataset_config, split='validation')
        self.test_dataset = LMDataset(dataset_config, split='test')
        
        if len(self.train_dataset) == 0:
            raise ValueError("训练数据集为空")
        if len(self.val_dataset) == 0:
            raise ValueError("验证数据集为空")
        if len(self.test_dataset) == 0:
            raise ValueError("测试数据集为空")
        
        processor_config = {
            'tokenizer_type': tokenizer_type,
            'max_length': max_length,
            'vocab_size': vocab_size
        }
        
        self.processor = LMProcessor(self.train_dataset, processor_config)
        
        batch_size = TASK2_TRAIN_CONFIG['batch_size']
        
        self.train_loader = self.processor.get_loader(
            self.train_dataset, batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        self.val_loader = self.processor.get_loader(
            self.val_dataset, batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
        self.test_loader = self.processor.get_loader(
            self.test_dataset, batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        stats = {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'vocab_size': len(self.processor.tokenizer),
            'max_length': max_length,
            'tokenizer_type': tokenizer_type
        }
        
        print(f"数据统计: {json.dumps(stats, indent=2)}")
        return stats
    
    def setup_model(self):
        print(f"\n[{self.name}] 设置模型...")
        
        arch_type = self.config['arch_type']
        param_scale = self.config['param_scale']
        param_config = TASK2_PARAM_CONFIGS[param_scale]
        vocab_size = len(self.processor.tokenizer)
        
        print(f"架构: {arch_type}, 参数规模: {param_scale}")
        print(f"词表大小: {vocab_size}")
        
        self.model = create_task2_model(arch_type, vocab_size, param_config)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TASK2_TRAIN_CONFIG['learning_rate'],
            weight_decay=TASK2_TRAIN_CONFIG['weight_decay']
        )
        
        num_training_steps = len(self.train_loader) * TASK2_TRAIN_CONFIG['epochs']
        num_warmup_steps = int(num_training_steps * TASK2_TRAIN_CONFIG['warmup_ratio'])
        
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=num_warmup_steps
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)
        
        model_config = {
            'arch_type': arch_type,
            'param_scale': param_scale,
            'vocab_size': vocab_size,
            'param_config': param_config,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_training_steps': num_training_steps,
            'num_warmup_steps': num_warmup_steps
        }
        
        with open(os.path.join(self.save_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        return model_config
    
    def make_mask(self, seq, pad_idx):
        mask = padding_mask(seq, pad_idx)
        if self.config['arch_type'] == 'decoder_only':
            mask = mask & subsequent_mask(seq.size(1)).to(seq.device)
        return mask
    
    def forward_pass(self, batch, training=True):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        pad_idx = self.processor.tokenizer.pad_token_id
        
        if self.config['arch_type'] == 'decoder_only':
            src = input_ids[:, :-1]
            mask = self.make_mask(src, pad_idx)
            outputs = self.model(src, mask)
            
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = input_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'encoder_only':
            src = input_ids
            mask = self.make_mask(src, pad_idx)
            
            if training:
                mlm_mask = torch.rand_like(src.float()) < 0.15
                mlm_mask = mlm_mask & (src != pad_idx)
                mlm_inputs = src.clone()
                mlm_inputs[mlm_mask] = self.processor.tokenizer.mask_token_id if self.processor.tokenizer.mask_token_id is not None else 103
            else:
                mlm_inputs = src
                mlm_mask = (src != pad_idx)
            
            outputs = self.model(mlm_inputs, mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = src.reshape(-1)
            
            outputs = outputs[mlm_mask.reshape(-1)]
            targets = targets[mlm_mask.reshape(-1)]
        
        elif self.config['arch_type'] == 'encoder_decoder':
            # 对于语言建模任务，使用 input_ids 作为源序列
            # 使用 input_ids 的前缀（去掉最后一个token）作为目标序列
            src = input_ids
            tgt = input_ids[:, :-1]
            
            src_mask = padding_mask(src, pad_idx)
            tgt_mask = padding_mask(tgt, pad_idx)
            tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).to(tgt.device)
            
            outputs = self.model(src, tgt, src_mask, tgt_mask)
            
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = input_ids[:, 1:].reshape(-1)
        
        return outputs, targets
    
    def compute_metrics(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        ppl = torch.exp(loss)
        
        _, predicted = outputs.max(1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        acc = 100. * correct / total
        
        top5_correct = 0
        top5_preds = outputs.topk(5, dim=1).indices
        for i in range(targets.size(0)):
            if targets[i] in top5_preds[i]:
                top5_correct += 1
        top5_acc = 100. * top5_correct / total
        
        return loss, ppl, acc, top5_acc
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_ppl = 0
        total_acc = 0
        total_top5_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            outputs, targets = self.forward_pass(batch, training=True)
            
            loss, ppl, acc, top5_acc = self.compute_metrics(outputs, targets)
            
            loss = loss / TASK2_TRAIN_CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % TASK2_TRAIN_CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), TASK2_TRAIN_CONFIG['clip_grad']
                )
                self.optimizer.step()
                self.scheduler.step()
            
            total_loss += loss.item() * TASK2_TRAIN_CONFIG['gradient_accumulation_steps']
            total_ppl += ppl.item()
            total_acc += acc
            total_top5_acc += top5_acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item()*TASK2_TRAIN_CONFIG["gradient_accumulation_steps"]:.4f}',
                'ppl': f'{ppl.item():.2f}',
                'acc': f'{acc:.2f}%'
            })
        
        avg_loss = total_loss / num_batches
        avg_ppl = total_ppl / num_batches
        avg_acc = total_acc / num_batches
        avg_top5_acc = total_top5_acc / num_batches
        
        return avg_loss, avg_ppl, avg_acc, avg_top5_acc
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_ppl = 0
        total_acc = 0
        total_top5_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                outputs, targets = self.forward_pass(batch, training=False)
                
                loss, ppl, acc, top5_acc = self.compute_metrics(outputs, targets)
                
                total_loss += loss.item()
                total_ppl += ppl.item()
                total_acc += acc
                total_top5_acc += top5_acc
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ppl = total_ppl / num_batches
        avg_acc = total_acc / num_batches
        avg_top5_acc = total_top5_acc / num_batches
        
        return avg_loss, avg_ppl, avg_acc, avg_top5_acc
    
    def train(self):
        print(f"\n[{self.name}] 开始训练...")
        
        best_val_ppl = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(1, TASK2_TRAIN_CONFIG['epochs'] + 1):
            train_loss, train_ppl, train_acc, train_top5_acc = self.train_epoch(epoch)
            
            val_loss, val_ppl, val_acc, val_top5_acc = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, PPL={train_ppl:.2f}, Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Acc={val_acc:.2f}%")
            
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，停止训练 (epoch {epoch})")
                    break
        
        print(f"\n[{self.name}] 训练完成!")
        
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def test(self):
        print(f"\n[{self.name}] 开始测试...")
        
        self.load_model('best_model.pt')
        
        test_loss, test_ppl, test_acc, test_top5_acc = self.evaluate(self.test_loader)
        
        print(f"测试结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  困惑度: {test_ppl:.2f}")
        print(f"  准确率: {test_acc:.2f}%")
        print(f"  Top-5 准确率: {test_top5_acc:.2f}%")
        
        bpc = test_loss / math.log(2)
        print(f"  Bits Per Character: {bpc:.4f}")
        
        results = {
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'test_accuracy': test_acc,
            'test_top5_accuracy': test_top5_acc,
            'test_bpc': bpc,
            'config': self.config
        }
        
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run(self):
        print(f"\n{'='*60}")
        print(f"开始实验: {self.name}")
        print(f"{'='*60}")
        
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.setup_data()
        self.setup_model()
        self.train()
        results = self.test()
        
        print(f"\n{'='*60}")
        print(f"实验完成: {self.name}")
        print(f"{'='*60}")
        
        return results


# ===============================
# Task 2: 实验运行函数
# ===============================

def run_task2_exp1_architecture_comparison(device):
    print("\n" + "="*60)
    print("Task 2 - 实验 1: 架构对比")
    print("="*60)

    architectures = ['decoder_only', 'encoder_only', 'encoder_decoder']

    results = {}
    for arch in architectures:
        name = f"exp1_arch_{arch}"
        config = {
            'arch_type': arch,
            'tokenizer_type': 'gpt2',
            'param_scale': 'small',
            'max_length': 128,
            'num_samples': 50000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device)
        results[arch] = experiment.run()
    
    summary = {
        'experiment': 'exp1_architecture_comparison',
        'results': results
    }
    
    save_path = os.path.join('results', 'task2', 'exp1_architecture_comparison_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_task2_exp2_tokenizer_comparison(device):
    print("\n" + "="*60)
    print("Task 2 - 实验 2: Tokenizer 对比")
    print("="*60)

    tokenizers = ['gpt2', 'bert']

    results = {}
    for tokenizer in tokenizers:
        name = f"exp2_tokenizer_{tokenizer}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': tokenizer,
            'param_scale': 'small',
            'max_length': 128,
            'num_samples': 50000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device)
        results[tokenizer] = experiment.run()
    
    summary = {
        'experiment': 'exp2_tokenizer_comparison',
        'results': results
    }
    
    save_path = os.path.join('results', 'task2', 'exp2_tokenizer_comparison_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_task2_exp3_parameter_scales(device):
    print("\n" + "="*60)
    print("Task 2 - 实验 3: 参数规模影响")
    print("="*60)

    scales = ['tiny', 'small', 'medium']

    results = {}
    for scale in scales:
        name = f"exp3_scale_{scale}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': 'gpt2',
            'param_scale': scale,
            'max_length': 128,
            'num_samples': 50000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device)
        results[scale] = experiment.run()
    
    summary = {
        'experiment': 'exp3_parameter_scales',
        'results': results
    }
    
    save_path = os.path.join('results', 'task2', 'exp3_parameter_scales_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_task2_exp4_sequence_length(device):
    print("\n" + "="*60)
    print("Task 2 - 实验 4: 序列长度影响")
    print("="*60)

    seq_lengths = [64, 128]

    results = {}
    for length in seq_lengths:
        name = f"exp4_seq_{length}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': 'gpt2',
            'param_scale': 'small',
            'max_length': length,
            'num_samples': 50000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device)
        results[length] = experiment.run()

    summary = {
        'experiment': 'exp4_sequence_length',
        'results': results
    }

    save_path = os.path.join('results', 'task2', 'exp4_sequence_length_summary.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ===============================
# 主函数
# ===============================

def main():
    """主函数：运行所有实验或指定任务的实验"""
    import argparse

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    run_task1_exp2_split_strategies(device)


if __name__ == '__main__':
    main()