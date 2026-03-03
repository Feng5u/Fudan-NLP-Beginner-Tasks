import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    FFN 层
    """
    def __init__(self, d_model, d_ff, dropout):
        """
        初始化函数

        参数：
            d_model: 模型维度
            d_ff: 隐藏层维度
            dropout: dropout 比例
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        前向传播过程

        参数：
            X: 输入数据

        返回：
            FFN 层输出
        """
        return self.w_2(self.dropout(self.w_1(X).relu()))
