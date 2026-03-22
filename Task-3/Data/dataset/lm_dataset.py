import os
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

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
        直接从缓存的 arrow 文件加载 WikiText-103 原始文本
        """
        # 缓存目录路径 - 使用绝对路径避免 ~ 在不同用户下指向不同位置
        cache_dir = '/home/feng5u/.cache/huggingface/datasets/wikitext/wikitext-103-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3'

        # 根据 split 确定要加载的文件
        if self.split == 'train':
            # 训练集有两个文件
            file1 = os.path.join(cache_dir, 'wikitext-train-00000-of-00002.arrow')
            file2 = os.path.join(cache_dir, 'wikitext-train-00001-of-00002.arrow')
            if os.path.exists(file1) and os.path.exists(file2):
                ds1 = HFDataset.from_file(file1)
                ds2 = HFDataset.from_file(file2)
                dataset = concatenate_datasets([ds1, ds2])
            else:
                raise FileNotFoundError(f"训练集文件不存在: {file1} 或 {file2}")
        elif self.split == 'validation':
            file_path = os.path.join(cache_dir, 'wikitext-validation.arrow')
            if os.path.exists(file_path):
                dataset = HFDataset.from_file(file_path)
            else:
                raise FileNotFoundError(f"验证集文件不存在: {file_path}")
        elif self.split == 'test':
            file_path = os.path.join(cache_dir, 'wikitext-test.arrow')
            if os.path.exists(file_path):
                dataset = HFDataset.from_file(file_path)
            else:
                raise FileNotFoundError(f"测试集文件不存在: {file_path}")
        else:
            raise ValueError(f"未知的 split: {self.split}")

        # 提取文本
        texts = [text for text in dataset['text'] if text.strip()]

        # 限制样本数量
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