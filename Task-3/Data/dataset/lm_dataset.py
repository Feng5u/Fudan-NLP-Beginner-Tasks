from torch.utils.data import Dataset
from datasets import load_dataset

class LMDataset(Dataset):
    """
    语言模型数据集 - 只负责加载原始文本
    """
    
    def __init__(self, config, split='train'):
        """
        参数：
            config: 配置字典
            split: 'train', 'validation', 'test'
        """
        self.config = config
        self.split = split
        
        self.texts = self._load_data()
        
        print(f"[{split}] 加载了 {len(self.texts)} 条原始文本")
    
    def _load_data(self):
        """
        加载 WikiText-103 原始文本
        """
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=self.split)
        
        texts = [text for text in dataset['text'] if text.strip()]
        
        num_samples = self.config.get('num_samples', None)
        if num_samples and len(texts) > num_samples:
            import random
            random.seed(42)
            texts = random.sample(texts, num_samples)
        
        return texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]