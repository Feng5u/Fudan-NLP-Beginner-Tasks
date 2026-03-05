import random
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from ..base_processor import BaseProcessor

class AdditionProcessor(BaseProcessor):
    """
    加法数据处理器
    
    支持的划分策略：
    1. 'random' - 随机划分
    2. 'digit_pair' - 按数字组合划分
    3. 'max_digits' - 按最大位数划分
    4. 'result_range' - 按结果范围划分
    5. 'carry_complexity' - 按进位复杂度划分
    """
    
    VALID_STRATEGIES = ['random', 'digit_pair', 'max_digits', 'result_range', 'carry_complexity']
    
    def __init__(self, dataset, config):
        """
        参数：
            dataset: 原始数据集
            config: 配置字典，必须包含：
                - split_strategy: 划分策略名称
                - 以及其他策略所需的参数
        """
        super().__init__(dataset, config)
        
        self.split_strategy = config.get('split_strategy', 'random')
        if self.split_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"无效的划分策略: {self.split_strategy}，可选: {self.VALID_STRATEGIES}")
        
        self.name = f"{self.split_strategy}"
        self.seed = config.get('seed', 42)
        
        self._cached_split = None
        
        print(f"创建处理器: {self.name}")
    
    def split(self):
        """
        根据选择的策略进行划分
        """
        if self._cached_split is not None:
            return self._cached_split
        
        if self.split_strategy == 'random':
            result = self._random_split()
        elif self.split_strategy == 'digit_pair':
            result = self._digit_pair_split()
        elif self.split_strategy == 'max_digits':
            result = self._max_digits_split()
        elif self.split_strategy == 'result_range':
            result = self._result_range_split()
        elif self.split_strategy == 'carry_complexity':
            result = self._carry_complexity_split()
        else:
            raise ValueError(f"未知策略: {self.split_strategy}")
        
        self._cached_split = result
        return result
    
    def _random_split(self):
        """
        随机划分
        """
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        total_size = len(self.dataset)
        indices = list(range(total_size))
        
        random.seed(self.seed)
        random.shuffle(indices)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"随机划分: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def _digit_pair_split(self):
        """
        按数字组合划分
        """
        train_pairs = self.config.get('train_pairs', 
                                     [(3,3), (3,4), (4,3)])
        val_pairs = self.config.get('val_pairs',
                                   [(3,5), (5,3)])
        test_pairs = self.config.get('test_pairs',
                                    [(4,4)])
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        train_set = set(train_pairs)
        val_set = set(val_pairs)
        test_set = set(test_pairs)
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            pair = (item['digits1'], item['digits2'])
            
            if pair in train_set:
                train_indices.append(idx)
            elif pair in val_set:
                val_indices.append(idx)
            elif pair in test_set:
                test_indices.append(idx)
        
        print(f"数字组合划分:")
        print(f"  训练组合 {train_pairs}: {len(train_indices)} 样本")
        print(f"  验证组合 {val_pairs}: {len(val_indices)} 样本")
        print(f"  测试组合 {test_pairs}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _max_digits_split(self):
        """
        按最大位数划分
        """
        train_digits = self.config.get('train_max_digits', [3])
        val_digits = self.config.get('val_max_digits', [4])
        test_digits = self.config.get('test_max_digits', [5])
        
        train_set = set(train_digits)
        val_set = set(val_digits)
        test_set = set(test_digits)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            max_d = max(item['digits1'], item['digits2'])
            
            if max_d in train_set:
                train_indices.append(idx)
            elif max_d in val_set:
                val_indices.append(idx)
            elif max_d in test_set:
                test_indices.append(idx)
        
        print(f"最大位数划分:")
        print(f"  训练位数 {train_digits}: {len(train_indices)} 样本")
        print(f"  验证位数 {val_digits}: {len(val_indices)} 样本")
        print(f"  测试位数 {test_digits}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _result_range_split(self):
        """
        按结果范围划分
        """
        train_range = self.config.get('train_range', (0, 1000))
        val_range = self.config.get('val_range', (1001, 5000))
        test_range = self.config.get('test_range', (5001, 20000))
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            result = item['result']
            
            if train_range[0] <= result <= train_range[1]:
                train_indices.append(idx)
            elif val_range[0] <= result <= val_range[1]:
                val_indices.append(idx)
            elif test_range[0] <= result <= test_range[1]:
                test_indices.append(idx)
        
        print(f"结果范围划分:")
        print(f"  训练范围 {train_range}: {len(train_indices)} 样本")
        print(f"  验证范围 {val_range}: {len(val_indices)} 样本")
        print(f"  测试范围 {test_range}: {len(test_indices)} 样本")
        
        return train_indices, val_indices, test_indices
    
    def _carry_complexity_split(self):
        """
        按进位复杂度划分
        """
        simple = []      # 无进位
        medium = []      # 1-2次进位
        complex_carry = []  # 3次以上进位
        
        for idx in range(len(self.dataset)):
            item = self.dataset.data[idx]
            num1, num2 = item['num1'], item['num2']
            
            carry_count = self._count_carries(num1, num2)
            
            if carry_count == 0:
                simple.append(idx)
            elif carry_count <= 2:
                medium.append(idx)
            else:
                complex_carry.append(idx)
        
        random.seed(self.seed)
        random.shuffle(simple)
        random.shuffle(medium)
        random.shuffle(complex_carry)
        
        def split_list(lst, train_ratio=0.7, val_ratio=0.15):
            total = len(lst)
            train_size = int(total * train_ratio)
            val_size = int(total * val_ratio)
            return (lst[:train_size], 
                    lst[train_size:train_size+val_size], 
                    lst[train_size+val_size:])
        
        simple_train, simple_val, simple_test = split_list(simple)
        medium_train, medium_val, medium_test = split_list(medium)
        complex_train, complex_val, complex_test = split_list(complex_carry)
        
        train_indices = simple_train + medium_train + complex_train
        val_indices = simple_val + medium_val + complex_val
        test_indices = simple_test + medium_test + complex_test
        
        print(f"进位复杂度划分:")
        print(f"  无进位样本: {len(simple)}")
        print(f"  中等进位样本: {len(medium)}")
        print(f"  复杂进位样本: {len(complex_carry)}")
        print(f"  最终: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
    
    def _count_carries(self, num1, num2):
        """
        计算加法中的进位次数
        """
        carry_count = 0
        n1, n2 = str(num1)[::-1], str(num2)[::-1]
        max_len = max(len(n1), len(n2))
        n1 = n1.ljust(max_len, '0')
        n2 = n2.ljust(max_len, '0')
        
        carry = 0
        for d1, d2 in zip(n1, n2):
            s = int(d1) + int(d2) + carry
            if s >= 10:
                carry_count += 1
                carry = 1
            else:
                carry = 0
        
        return carry_count
    
    def get_stats(self):
        """
        获取详细的统计信息
        """
        train_idx, val_idx, test_idx = self.split()
        
        stats = {
            'processor_name': self.name,
            'split_strategy': self.split_strategy,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'total_size': len(self.dataset)
        }
        
        if self.split_strategy == 'random':
            stats.update({
                'train_ratio': self.config.get('train_ratio', 0.7),
                'val_ratio': self.config.get('val_ratio', 0.15),
                'seed': self.seed
            })
        
        elif self.split_strategy == 'digit_pair':
            stats.update({
                'train_pairs': self.config.get('train_pairs'),
                'val_pairs': self.config.get('val_pairs'),
                'test_pairs': self.config.get('test_pairs')
            })
        
        elif self.split_strategy == 'max_digits':
            stats.update({
                'train_digits': self.config.get('train_max_digits'),
                'val_digits': self.config.get('val_max_digits'),
                'test_digits': self.config.get('test_max_digits')
            })
        
        elif self.split_strategy == 'result_range':
            stats.update({
                'train_range': self.config.get('train_range'),
                'val_range': self.config.get('val_range'),
                'test_range': self.config.get('test_range')
            })
        
        return stats
