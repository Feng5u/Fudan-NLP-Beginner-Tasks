import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    归一化层

    对最后一个维度归一化，学习两个可训练参数：
        - self.a_2，初始化为全一
        - self.b_2，初始化为全零
    """
    def __init__(self, features, eps=1e-6):
        """
        初始化函数

        参数：
            features: 特征维度大小，通常为 d_model
            eps: 防止出现除以零错误
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, X):
        """
        前向传播

        参数：
            X: 输入数据

        返回：
            返回归一化结果
        """
        mean = X.mean(-1, keepdim=True)
        std = X.std(-1, keepdim=True)
        return self.a_2 * (X - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    子层连接

    包括一个残差连接和一个归一化层
    """
    def __init__(self, size, dropout):
        """
        初始化函数

        参数：
            size: 数据特征维度大小
            dropout: dropout 比例
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, sublayer):
        """
        前向传播过程

        参数：
            X: 输入数据
            sublayer: 子层实例

        返回：
            将输入数据归一化后经过子层，dropout 后与输入数据进行残差连接
        """
        return X + self.dropout(sublayer(self.norm(X)))
