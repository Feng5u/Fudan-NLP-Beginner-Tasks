from .encoder_decoder import make_model as make_encoder_decoder_model
from .decoder_only import make_decoder_only_model
from .encoder_only import make_encoder_only_model

def create_model(arch='encoder_decoder', **kwargs):
    """
    统一模型创建接口

    参数：
        arch: 模型类型
        **kwargs: 模型参数
    """
    if arch == 'encoder_decoder':
        return make_encoder_decoder_model(**kwargs)
    elif arch == 'decoder_only':
        return make_decoder_only_model(**kwargs)
    elif arch == 'encoder_only':
        return make_encoder_only_model(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}.")

__all__ = [
    'create_model',
    'make_encoder_decoder_model',
    'make_decoder_only_model',
    'make_encoder_only_model'
]