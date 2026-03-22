import torch.nn as nn

class BaseModeL(nn.Module):
    """
    所有模型的基类
    """

    def __init__(self):
        super(BaseModeL, self).__init__()

    def count_parameters(self):
        """
        统计参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad())

    def forward(self, *args, **kwargs):
        """
        强制子类实现 forward 接口
        """
        raise NotImplementedError