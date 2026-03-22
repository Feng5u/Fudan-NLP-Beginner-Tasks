import math
import torch.nn as nn

class Embedding(nn.Module):
    """
    词嵌入层
    """
    def __init__(self, d_model, vocab):
        """
        初始化函数

        参数：
            d_model: 模型维度
            vocab: 词汇表大小
        """
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, X):
        """
        前向传播过程

        参数：
            X: 输入数据

        返回：
            缩放后的词嵌入向量
        """
        return self.lut(X) * math.sqrt(self.d_model)