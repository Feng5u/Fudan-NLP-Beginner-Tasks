"""
Transformer 模型库
"""
from .models import create_model
from .models.encoder_decoder import make_model as make_encoder_decoder
from .models.decoder_only import make_decoder_only_model
from .models.encoder_only import make_encoder_only_model

__all__ = [
    'create_model',
    'make_encoder_decoder',
    'make_decoder_only_model',
    'make_encoder_only_model'
]

__version__ = '0.1.0'