import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class LMProcessor:
    """
    语言模型数据处理器 - 负责数据处理
    """
    
    def __init__(self, dataset, config):
        """
        参数：
            dataset: LMDataset 实例（已加载原始数据）
            config: 配置字典，包含：
                - tokenizer_type: tokenizer类型
                - vocab_size: 词表大小（可选）
                - max_length: 最大序列长度
        """
        self.dataset = dataset
        self.config = config
        
        self.tokenizer = self._create_tokenizer()
        
        print(f"LMProcessor 初始化完成:")
        print(f"  - Tokenizer: {config.get('tokenizer_type')}")
        print(f"  - 词表大小: {len(self.tokenizer)}")
        print(f"  - 最大长度: {config.get('max_length')}")
    
    def _create_tokenizer(self):
        """
        创建 tokenizer
        """
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
        
        vocab_size = self.config.get('vocab_size', None)
        if vocab_size and vocab_size < len(tokenizer):
            print(f"注意: 限制词表大小从 {len(tokenizer)} 到 {vocab_size}")
        
        return tokenizer
    
    def _collate_fn(self, batch_texts):
        """
        批处理函数：编码和padding
        """
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
            torch.full((input_ids.size(0), 1), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        ], dim=1)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': encoded['attention_mask']
        }

    def get_loader(self, dataset, batch_size, shuffle=False):
        """
        获取给定 dataset 的数据加载器（通用接口）

        用法示例：
            train_ds = LMDataset(config, split='train')
            train_loader = lm_processor.get_loader(train_ds, batch_size=32, shuffle=True)
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def get_vocab_size(self):
        return len(self.tokenizer)