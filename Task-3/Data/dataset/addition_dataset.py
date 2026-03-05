import os
import json
import torch
import random

from torch.utils.data import Dataset

random.seed(42)

class AdditionDataset(Dataset):
    """
    多位数加法数据集
    """
    
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    
    DIGITS = [str(i) for i in range(10)]
    OPERATORS = ['+', '=']
    
    VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + DIGITS + OPERATORS
    
    TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
    ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
    
    PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
    SOS_ID = TOKEN_TO_ID[SOS_TOKEN]
    EOS_ID = TOKEN_TO_ID[EOS_TOKEN]
    UNK_ID = TOKEN_TO_ID[UNK_TOKEN]
    
    def __init__(self, file_path):
        """
        初始化加法数据集
        
        参数：
            file_path: 文件路径
            split: 'train', 'val', 或 'test'
        """
        self.file_path = file_path
        
        self.data = self._load_data()
        self.data = self._validate_data()
        
        self._print_stats()
    
    def _load_data(self):
        """
        从文本文件加载数据
        """
        data = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if '+' in line and '=' in line:
                        num1, rest = line.split('+')
                        num2, result = rest.split('=')
                        
                        if num1.isdigit() and num2.isdigit() and result.isdigit():
                            data.append({
                                'id': line_num,
                                'expr': line,
                                'input': f"{num1}+{num2}=",
                                'target': result,
                                'num1': int(num1),
                                'num2': int(num2),
                                'result': int(result),
                                'digits1': len(num1),
                                'digits2': len(num2)
                            })
                except Exception as e:
                    print(f"警告: 第{line_num}行解析失败: {line}, 错误: {e}")
        
        return data
    
    def _validate_data(self):
        """
        验证数据有效性
        """
        valid_data = []
        
        for item in self.data:
            if len(str(item['num1'])) != item['digits1']:
                continue
            if len(str(item['num2'])) != item['digits2']:
                continue
            
            if item['num1'] + item['num2'] != item['result']:
                continue
            
            valid_data.append(item)
        
        if len(valid_data) < len(self.data):
            print(f"数据验证: 移除了 {len(self.data) - len(valid_data)} 条无效数据")
        
        return valid_data
    
    def _print_stats(self):
        """
        打印数据统计信息
        """
        if not self.data:
            return

        digits1 = [item['digits1'] for item in self.data]
        digits2 = [item['digits2'] for item in self.data]
        results = [item['result'] for item in self.data]
        
        pair_counts = {}
        for item in self.data:
            pair = (item['digits1'], item['digits2'])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {len(self.data)}")
        print(f"  数字位数范围: {min(digits1)}-{max(digits1)} + {min(digits2)}-{max(digits2)}")
        print(f"  结果范围: {min(results)} - {max(results)}")
        print(f"  组合分布:")
        for pair, count in sorted(pair_counts.items()):
            print(f"    {pair[0]}+{pair[1]}: {count} 条 ({count/len(self.data)*100:.1f}%)")
    
    def encode(self, text):
        """
        将文本编码为 ID 序列
        使用类级别的固定词汇表
        """
        ids = [self.SOS_ID]
        
        for char in text:
            if char in self.TOKEN_TO_ID:
                ids.append(self.TOKEN_TO_ID[char])
            else:
                ids.append(self.UNK_ID)
        
        ids.append(self.EOS_ID)
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        将ID序列解码为文本
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        chars = []
        for id_ in ids:
            token = self.ID_TO_TOKEN[id_]
            if skip_special_tokens and token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
                continue
            chars.append(token)
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本，格式化为模型输入
        
        返回的字典包含：
            - input_ids: 输入序列ID (sos + 输入 + eos)
            - target_ids: 目标序列ID (sos + 目标 + eos)
            - expr: 原始表达式
            - num1, num2, result: 数字值
            - digits1, digits2: 位数
        """
        item = self.data[idx]
        
        input_ids = self.encode(item['input'])
        target_ids = self.encode(item['target'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'expr': item['expr'],
            'num1': item['num1'],
            'num2': item['num2'],
            'result': item['result'],
            'digits1': item['digits1'],
            'digits2': item['digits2']
        }
    
    @classmethod
    def get_vocab_size(cls):
        return len(cls.VOCAB)
    
    @classmethod
    def get_pad_id(cls):
        return cls.PAD_ID