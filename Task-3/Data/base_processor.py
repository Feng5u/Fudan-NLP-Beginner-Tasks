from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Subset

class BaseProcessor(ABC):
    """
    处理器基类
    """
    
    def __init__(self, dataset, config):
        """
        初始化函数

        参数：
            dataset: 原始数据集
            config: 配置字典
        """
        self.dataset = dataset
        self.config = config
        self.name = "base_processor"
    
    @abstractmethod
    def split(self):
        """
        划分数据集
        
        返回：
            train_indices, val_indices, test_indices
        """
        pass
    
    def get_train_dataset(self):
        train_idx, _, _ = self.split()
        return Subset(self.dataset, train_idx)
    
    def get_val_dataset(self):
        _, val_idx, _ = self.split()
        return Subset(self.dataset, val_idx)
    
    def get_test_dataset(self):
        _, _, test_idx = self.split()
        return Subset(self.dataset, test_idx)
    
    def get_train_loader(self, batch_size, shuffle=True, collate_fn=None, num_workers=4, pin_memory=True):
        dataset = self.get_train_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    def get_val_loader(self, batch_size, shuffle=False, collate_fn=None, num_workers=4, pin_memory=True):
        dataset = self.get_val_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    def get_test_loader(self, batch_size, shuffle=False, collate_fn=None, num_workers=4, pin_memory=True):
        dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    def get_vocab_size(self):
        return self.dataset.get_vocab_size()
    
    def get_stats(self):
        train_idx, val_idx, test_idx = self.split()
        return {
            'processor_name': self.name,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_size': len(self.dataset)
        }