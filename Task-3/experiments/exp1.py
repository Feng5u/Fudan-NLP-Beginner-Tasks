import os
import json
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 导入数据相关模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Data.dataset.addition_dataset import AdditionDataset
from Data.processors.addition_processor import AdditionProcessor

# 导入模型相关模块
from transformer.models.encoder_decoder import make_model
from transformer.models.decoder_only import make_decoder_only_model
from transformer.models.encoder_only import make_encoder_only_model
from transformer.utils.masking import padding_mask, subsequent_mask


# ===============================
# 配置定义
# ===============================

PARAM_CONFIGS = {
    'small': {
        'N': 2,
        'd_model': 128,
        'd_ff': 512,
        'h': 4,
        'dropout': 0.1
    },
    'medium': {
        'N': 4,
        'd_model': 256,
        'd_ff': 1024,
        'h': 8,
        'dropout': 0.1
    },
    'large': {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'dropout': 0.1
    }
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'warmup_steps': 2000,
    'clip_grad': 1.0
}

SPLIT_CONFIGS = {
    'random': {
        'split_strategy': 'random',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'seed': 42
    },
    'digit_pair': {
        'split_strategy': 'digit_pair',
        'train_pairs': [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)],
        'val_pairs': [(3, 4), (4, 3)],
        'test_pairs': [(4, 4), (4, 5), (5, 4), (5, 5)]
    },
    'max_digits': {
        'split_strategy': 'max_digits',
        'train_max_digits': [3],
        'val_max_digits': [4],
        'test_max_digits': [5]
    },
    'result_range': {
        'split_strategy': 'result_range',
        'train_range': (0, 1000),
        'val_range': (1001, 5000),
        'test_range': (5001, 20000)
    },
    'carry_complexity': {
        'split_strategy': 'carry_complexity',
        'seed': 42
    }
}


# ===============================
# 工具函数
# ===============================

def set_seed(seed=42):
    """
    设置随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_device():
    """
    自动检测并设置设备
    
    返回：
        device: torch.device
        use_multi_gpu: bool, 是否使用多 GPU
        gpu_ids: list[int], GPU ID 列表
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"检测到 {num_gpus} 个 GPU，将使用多 GPU 训练")
            print(f"GPU 设备: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
            device = torch.device('cuda:0')
            use_multi_gpu = True
            gpu_ids = list(range(num_gpus))
        else:
            print(f"检测到 1 个 GPU，将使用单 GPU 训练")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda:0')
            use_multi_gpu = False
            gpu_ids = [0]
    else:
        print("未检测到 GPU，将使用 CPU 训练")
        device = torch.device('cpu')
        use_multi_gpu = False
        gpu_ids = []
    
    return device, use_multi_gpu, gpu_ids


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    max_input_len = max(len(ids) for ids in input_ids)
    padded_input = torch.full((len(input_ids), max_input_len), 0, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_input[i, :len(ids)] = ids
    
    max_target_len = max(len(ids) for ids in target_ids)
    padded_target = torch.full((len(target_ids), max_target_len), 0, dtype=torch.long)
    for i, ids in enumerate(target_ids):
        padded_target[i, :len(ids)] = ids
    
    return {
        'input_ids': padded_input,
        'target_ids': padded_target,
        'input_lengths': torch.tensor([len(ids) for ids in input_ids]),
        'target_lengths': torch.tensor([len(ids) for ids in target_ids]),
        'metadata': batch
    }


# ===============================
# 模型创建
# ===============================

def create_model(arch_type, vocab_size, param_config, use_relative_position=False, max_relative_position=127):
    """
    创建指定架构的模型
    
    参数：
        arch_type: 'encoder_decoder', 'decoder_only', 'encoder_only'
        vocab_size: 词表大小
        param_config: 参数配置字典
        use_relative_position: 是否使用相对位置编码
        max_relative_position: 最大相对距离
    """
    N = param_config['N']
    d_model = param_config['d_model']
    d_ff = param_config['d_ff']
    h = param_config['h']
    dropout = param_config['dropout']
    
    if arch_type == 'encoder_decoder':
        model = make_model(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            use_relative_position=use_relative_position,
            max_relative_position=max_relative_position
        )
    elif arch_type == 'decoder_only':
        model = make_decoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            use_relative_position=use_relative_position,
            max_relative_position=max_relative_position
        )
    elif arch_type == 'encoder_only':
        model = make_encoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            task='mlm',
            use_relative_position=use_relative_position,
            max_relative_position=max_relative_position
        )
    else:
        raise ValueError(f"未知架构类型: {arch_type}")
    
    return model


# ===============================
# 训练和评估类
# ===============================

class Experiment:
    """
    实验类：管理完整的实验流程
    """
    def __init__(self, name, config, device='cpu', use_multi_gpu=False, gpu_ids=None):
        """
        参数：
            name: 实验名称
            config: 实验配置字典，包含：
                - arch_type: 架构类型
                - split_strategy: 数据划分策略
                - param_scale: 参数规模 ('small', 'medium', 'large')
                - data_path: 数据文件路径
            device: 设备
            use_multi_gpu: 是否使用多 GPU
            gpu_ids: GPU ID 列表
        """
        self.name = name
        self.config = config
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = gpu_ids if gpu_ids is not None else []
        
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'results', 'task1', name
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.dataset = None
        self.processor = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_data(self):
        """
        设置数据集和处理器
        """
        print(f"\n[{self.name}] 设置数据...")
        
        data_path = self.config.get('data_path', 'Data/dataset/addition.txt')
        if not os.path.isabs(data_path):
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), data_path
            )
        
        self.dataset = AdditionDataset(data_path)
        
        split_strategy = self.config.get('split_strategy', 'random')
        split_config = SPLIT_CONFIGS[split_strategy].copy()
        split_config['seed'] = self.config.get('seed', 42)
        
        self.processor = AdditionProcessor(self.dataset, split_config)
        
        stats = self.processor.get_stats()
        print(f"数据统计: {json.dumps(stats, indent=2)}")
        
        batch_size = TRAIN_CONFIG['batch_size']
        
        train_indices, val_indices, test_indices = self.processor.split()
        
        if len(train_indices) == 0:
            raise ValueError(f"训练集为空！请检查数据划分策略 '{split_strategy}' 的配置。")
        if len(val_indices) == 0:
            raise ValueError(f"验证集为空！请检查数据划分策略 '{split_strategy}' 的配置。")
        if len(test_indices) == 0:
            raise ValueError(f"测试集为空！请检查数据划分策略 '{split_strategy}' 的配置。")
        
        self.train_loader = self.processor.get_train_loader(
            batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        self.val_loader = self.processor.get_val_loader(
            batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        self.test_loader = self.processor.get_test_loader(
            batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        
        return stats
    
    def setup_model(self):
        """
        设置模型
        """
        print(f"\n[{self.name}] 设置模型...")
        
        arch_type = self.config['arch_type']
        param_scale = self.config['param_scale']
        param_config = PARAM_CONFIGS[param_scale]
        vocab_size = self.dataset.get_vocab_size()
        
        # 获取相对位置编码配置
        use_relative_position = self.config.get('use_relative_position', False)
        max_relative_position = self.config.get('max_relative_position', 127)
        
        print(f"架构: {arch_type}, 参数规模: {param_scale}")
        print(f"词表大小: {vocab_size}")
        print(f"位置编码: {'相对位置编码' if use_relative_position else '绝对位置编码'}")
        if use_relative_position:
            print(f"最大相对距离: {max_relative_position}")
        
        self.model = create_model(arch_type, vocab_size, param_config, use_relative_position, max_relative_position)
        self.model = self.model.to(self.device)
        
        # 如果使用多 GPU，使用 DataParallel 包装模型
        if self.use_multi_gpu and len(self.gpu_ids) > 1:
            print(f"使用多 GPU 训练: {self.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            print(f"模型已包装为 DataParallel")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAIN_CONFIG['learning_rate']
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.get_pad_id())
        
        model_config = {
            'arch_type': arch_type,
            'param_scale': param_scale,
            'vocab_size': vocab_size,
            'param_config': param_config,
            'use_relative_position': use_relative_position,
            'max_relative_position': max_relative_position if use_relative_position else None,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device),
            'use_multi_gpu': self.use_multi_gpu,
            'gpu_ids': self.gpu_ids
        }
        
        with open(os.path.join(self.save_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        return model_config
    
    def make_src_mask(self, src, pad_idx):
        return padding_mask(src, pad_idx)
    
    def make_tgt_mask(self, tgt, pad_idx):
        tgt_mask = padding_mask(tgt, pad_idx)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1)).to(tgt.device)
        return tgt_mask
    
    def forward_pass(self, batch, training=True):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        pad_idx = self.dataset.get_pad_id()
        
        if self.config['arch_type'] == 'encoder_decoder':
            src = input_ids
            tgt = target_ids[:, :-1]
            
            src_mask = self.make_src_mask(src, pad_idx)
            tgt_mask = self.make_tgt_mask(tgt, pad_idx)
            
            outputs = self.model(src, tgt, src_mask, tgt_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'decoder_only':
            src = input_ids
            tgt = target_ids[:, :-1]
            
            combined = torch.cat([src, tgt], dim=1)
            mask = self.make_tgt_mask(combined, pad_idx)
            
            outputs = self.model(combined, mask)
            outputs = outputs[:, src.size(1):, :]
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'encoder_only':
            src = input_ids
            src_mask = self.make_src_mask(src, pad_idx)

            outputs = self.model(src, src_mask)

            # 对于 encoder-only 架构，我们需要重新设计输入输出格式
            # 输入：[SOS, num1, +, num2, =, EOS]
            # 输出：预测结果部分（从=之后开始）
            # 目标：[result_digits, EOS]

            # 由于不同样本的输入长度不同，我们需要动态计算每个样本的结果起始位置
            input_lengths = batch['input_lengths']

            # 找到每个样本中=的位置
            # 输入格式：[SOS, num1, +, num2, =, EOS]
            # =的位置是 input_lengths[i] - 2
            batch_size = input_ids.size(0)
            outputs_list = []
            targets_list = []

            for i in range(batch_size):
                # 计算=的位置
                eq_pos = input_lengths[i] - 2
                result_start = eq_pos + 1

                # 获取该样本的输入长度和目标长度
                src_len = input_lengths[i]
                tgt_len = batch['target_lengths'][i]

                # 从outputs中提取结果部分的预测
                # outputs[i, result_start:src_len-1] 是结果部分的预测（排除EOS）
                # 我们需要预测 tgt_len-1 个token（排除SOS）
                # 但由于padding，我们需要截取到实际的输出长度

                # 简化方案：使用 target_ids 的长度（排除SOS）
                result_len = tgt_len - 1

                # 从 outputs 中提取对应位置的预测
                # outputs[i, src_len-1-result_len:src_len-1] 对应结果部分
                output_start = max(0, src_len - 1 - result_len)
                output_end = src_len - 1

                sample_output = outputs[i, output_start:output_end, :]
                # 重要：只取前 result_len 个 token，排除SOS和padding
                sample_target = target_ids[i, 1:result_len+1]

                outputs_list.append(sample_output)
                targets_list.append(sample_target)

            # 将所有样本的输出和目标拼接起来
            outputs = torch.cat(outputs_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
        
        return outputs, targets
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            self.optimizer.zero_grad()
            
            outputs, targets = self.forward_pass(batch, training=True)
            
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TRAIN_CONFIG['clip_grad']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            mask = targets != self.dataset.get_pad_id()
            correct += (predicted[mask] == targets[mask]).sum().item()
            total += mask.sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, data_loader):
        if len(data_loader) == 0:
            return float('nan'), 0.0, {}
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        digit_acc = {}
        
        with torch.no_grad():
            for batch in data_loader:
                outputs, targets = self.forward_pass(batch, training=False)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                mask = targets != self.dataset.get_pad_id()
                
                batch_correct = (predicted[mask] == targets[mask]).sum().item()
                batch_total = mask.sum().item()
                
                correct += batch_correct
                total += batch_total
                
                # 根据架构类型计算每个样本的 targets 长度
                arch_type = self.config['arch_type']
                if arch_type == 'encoder_only':
                    # encoder_only: targets 是逐个样本拼接的，长度 = sum(target_lengths - 1)
                    target_lengths_adjusted = batch['target_lengths'] - 1
                else:
                    # encoder_decoder 和 decoder_only: targets = target_ids[:, 1:].reshape(-1), 去掉 SOS
                    target_lengths_adjusted = batch['target_lengths'] - 1

                # 对于 encoder-only，需要逐个样本处理
                if arch_type == 'encoder_only':
                    current_start = 0
                    for idx, metadata in enumerate(batch['metadata']):
                        digits1 = metadata['digits1']
                        digits2 = metadata['digits2']
                        key = f"{digits1}+{digits2}"
                        if key not in digit_acc:
                            digit_acc[key] = {'correct': 0, 'total': 0}
                        digit_acc[key]['total'] += 1

                        current_target_len = target_lengths_adjusted[idx]
                        end_idx = current_start + current_target_len

                        target_mask = mask[current_start:end_idx]
                        if target_mask.any():
                            sample_correct = (predicted[current_start:end_idx][target_mask] == targets[current_start:end_idx][target_mask]).sum().item()
                            total_sample = target_mask.sum().item()
                            if sample_correct == total_sample:
                                digit_acc[key]['correct'] += 1

                        current_start = end_idx
                else:
                    # encoder_decoder 和 decoder_only 使用相同的索引计算方式
                    for idx, metadata in enumerate(batch['metadata']):
                        digits1 = metadata['digits1']
                        digits2 = metadata['digits2']
                        key = f"{digits1}+{digits2}"
                        if key not in digit_acc:
                            digit_acc[key] = {'correct': 0, 'total': 0}
                        digit_acc[key]['total'] += 1

                        current_target_len = target_lengths_adjusted[idx]

                        if idx == 0:
                            start_idx = 0
                        else:
                            start_idx = target_lengths_adjusted[:idx].sum().item()

                        end_idx = start_idx + current_target_len
                        target_mask = mask[start_idx:end_idx]
                        if target_mask.any():
                            sample_correct = (predicted[start_idx:end_idx][target_mask] == targets[start_idx:end_idx][target_mask]).sum().item()
                            total_sample = target_mask.sum().item()
                            if sample_correct == total_sample:
                                digit_acc[key]['correct'] += 1
        
        avg_loss = total_loss / len(data_loader)
        avg_acc = 100. * correct / total
        
        for key in digit_acc:
            digit_acc[key]['accuracy'] = 100. * digit_acc[key]['correct'] / digit_acc[key]['total']
        
        return avg_loss, avg_acc, digit_acc
    
    def train(self):
        """
        训练模型
        """
        print(f"\n[{self.name}] 开始训练...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            
            val_loss, val_acc, _ = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if np.isnan(val_loss):
                print(f"Epoch {epoch}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                      f"Val Loss=N/A (empty), Val Acc=N/A")
                continue
            
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，停止训练 (epoch {epoch})")
                    break
        
        print(f"\n[{self.name}] 训练完成!")
        
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def test(self):
        """
        测试模型
        """
        print(f"\n[{self.name}] 开始测试...")
        
        self.load_model('best_model.pt')
        
        if len(self.test_loader) == 0:
            print(f"警告: 测试集为空，跳过测试")
            return {
                'test_loss': None,
                'test_accuracy': None,
                'digit_accuracy': {},
                'config': self.config
            }
        
        test_loss, test_acc, digit_acc = self.evaluate(self.test_loader)
        
        print(f"测试结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  准确率: {test_acc:.2f}%")
        print(f"\n按位数准确率:")
        for key, stats in sorted(digit_acc.items()):
            print(f"  {key}: {stats['accuracy']:.2f}%")
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'digit_accuracy': digit_acc,
            'config': self.config
        }
        
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run(self):
        """
        运行完整实验
        """
        print(f"\n{'='*60}")
        print(f"开始实验: {self.name}")
        print(f"{'='*60}")
        
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.setup_data()
        
        self.setup_model()
        
        self.train()
        
        results = self.test()
        
        print(f"\n{'='*60}")
        print(f"实验完成: {self.name}")
        print(f"{'='*60}")
        
        return results


# ===============================
# 实验运行函数
# ===============================

def run_exp1_architecture_comparison(device, use_multi_gpu=False, gpu_ids=None):
    """
    实验1: 架构对比
    """
    print("\n" + "="*60)
    print("实验 1: 架构对比")
    print("="*60)

    architectures = ['encoder_decoder', 'decoder_only', 'encoder_only']

    results = {}
    for arch in architectures:
        name = f"exp1_arch_{arch}"
        config = {
            'arch_type': arch,
            'split_strategy': 'random',
            'param_scale': 'medium',
            'seed': 42
        }

        experiment = Experiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[arch] = experiment.run()
    
    summary = {
        'experiment': 'exp1_architecture_comparison',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task1', 'exp1_architecture_comparison_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp2_split_strategies(device, use_multi_gpu=False, gpu_ids=None):
    """
    实验2: 数据划分策略泛化
    """
    print("\n" + "="*60)
    print("实验 2: 数据划分策略泛化")
    print("="*60)

    strategies = ['random', 'digit_pair', 'max_digits', 'result_range', 'carry_complexity']

    results = {}
    for strategy in strategies:
        name = f"exp2_split_{strategy}"
        config = {
            'arch_type': 'encoder_only',
            'split_strategy': strategy,
            'param_scale': 'medium',
            'seed': 42
        }

        experiment = Experiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[strategy] = experiment.run()
    
    summary = {
        'experiment': 'exp2_split_strategies',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task1', 'exp2_split_strategies_summary.json'
    )
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp3_parameter_scales(device, use_multi_gpu=False, gpu_ids=None):
    """
    实验3: 参数规模影响
    """
    print("\n" + "="*60)
    print("实验 3: 参数规模影响")
    print("="*60)

    scales = ['small', 'medium', 'large']

    results = {}
    for scale in scales:
        name = f"exp3_scale_{scale}"
        config = {
            'arch_type': 'encoder_only',
            'split_strategy': 'random',
            'param_scale': scale,
            'seed': 42
        }

        experiment = Experiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[scale] = experiment.run()

    summary = {
        'experiment': 'exp3_parameter_scales',
        'results': results
    }

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task1', 'exp3_parameter_scales_summary.json'
    )
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def run_single_experiment(name, config, device='cpu', use_multi_gpu=False, gpu_ids=None):
    experiment = Experiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
    return experiment.run()


# ===============================
# 主函数
# ===============================

def main():
    """
    主函数：运行所有实验
    """
    set_seed(42)

    # 自动检测和设置设备
    device, use_multi_gpu, gpu_ids = setup_device()
    print(f"使用设备: {device}")
    if use_multi_gpu:
        print(f"多 GPU 训练: 是 (GPU IDs: {gpu_ids})")
    else:
        print(f"多 GPU 训练: 否")

    import sys
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

        if exp_name == 'exp1':
            run_exp1_architecture_comparison(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp2':
            run_exp2_split_strategies(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp3':
            run_exp3_parameter_scales(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        else:
            print(f"未知实验: {exp_name}")
            print("可用实验: exp1, exp2, exp3")
    else:
        print("运行所有实验...")

        print("\n" + "="*60)
        print("开始完整实验流程")
        print("="*60)

        run_exp1_architecture_comparison(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)

        run_exp2_split_strategies(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)

        run_exp3_parameter_scales(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)

        print("\n" + "="*60)
        print("所有实验完成!")
        print("="*60)


if __name__ == '__main__':
    main()