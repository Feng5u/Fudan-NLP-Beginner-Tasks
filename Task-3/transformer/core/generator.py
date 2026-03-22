import torch.nn as nn
from torch.nn.functional import log_softmax

class Generator(nn.Module):
    """
    生成器

    包括一个线性层和一个 log_softmax 输出
    """
    def __init__(self, d_model, vocab):
        """
        初始化函数

        参数：
            d_model: 词向量的维度
            vocab: 词汇表大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, X):
        """
        前向传播

        参数：
            X: 输入数据，在标准的 Transformer 中，即 Decoder 的输出
        
        返回：
            返回词汇表中每个词在当前位置出现的概率
        """
        return log_softmax(self.proj(X), dim=-1)