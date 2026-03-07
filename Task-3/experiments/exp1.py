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
        'train_pairs': [(3, 3), (3, 4), (4, 3)],
        'val_pairs': [(3, 5), (5, 3)],
        'test_pairs': [(4, 4)]
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


def collate_fn(batch):
    """
    批处理函数：padding 到最大长度
    """
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    max_input_len = max(len(ids) for ids in input_ids)
    padded_input = torch.zeros(len(input_ids), max_input_len, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_input[i, :len(ids)] = ids
    
    max_target_len = max(len(ids) for ids in target_ids)
    padded_target = torch.zeros(len(target_ids), max_target_len, dtype=torch.long)
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

def create_model(arch_type, vocab_size, param_config):
    """
    创建指定架构的模型
    
    参数：
        arch_type: 'encoder_decoder', 'decoder_only', 'encoder_only'
        vocab_size: 词表大小
        param_config: 参数配置字典
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
            dropout=dropout
        )
    elif arch_type == 'decoder_only':
        model = make_decoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout
        )
    elif arch_type == 'encoder_only':
        model = make_encoder_only_model(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            dropout=dropout,
            task='mlm'
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
    def __init__(self, name, config, device='cpu'):
        """
        参数：
            name: 实验名称
            config: 实验配置字典，包含：
                - arch_type: 架构类型
                - split_strategy: 数据划分策略
                - param_scale: 参数规模 ('small', 'medium', 'large')
                - data_path: 数据文件路径
        """
        self.name = name
        self.config = config
        self.device = device
        
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
        
        self.train_loader = self.processor.get_train_loader(
            batch_size, shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = self.processor.get_val_loader(
            batch_size, shuffle=False, collate_fn=collate_fn
        )
        self.test_loader = self.processor.get_test_loader(
            batch_size, shuffle=False, collate_fn=collate_fn
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
        
        print(f"架构: {arch_type}, 参数规模: {param_scale}")
        print(f"词表大小: {vocab_size}")
        
        self.model = create_model(arch_type, vocab_size, param_config)
        self.model = self.model.to(self.device)
        
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
            'total_params': total_params,
            'trainable_params': trainable_params
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
        """
        前向传播
        """
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
            
            combined = torch.cat([src, tgt[:, 1:]], dim=1)
            mask = self.make_tgt_mask(combined, pad_idx)
            
            outputs = self.model(combined, mask)
            outputs = outputs[:, src.size(1)-1:-1, :]
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, 1:].reshape(-1)
            
        elif self.config['arch_type'] == 'encoder_only':
            src = input_ids
            src_mask = self.make_src_mask(src, pad_idx)
            
            outputs = self.model(src, src_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, :-1].reshape(-1)
        
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
        """
        评估模型
        """
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
                
                for idx, metadata in enumerate(batch['metadata']):
                    digits1 = metadata['digits1']
                    digits2 = metadata['digits2']
                    key = f"{digits1}+{digits2}"
                    if key not in digit_acc:
                        digit_acc[key] = {'correct': 0, 'total': 0}
                    digit_acc[key]['total'] += 1
                    
                    target_lengths = batch['target_lengths'].to(self.device)
                    current_target_len = target_lengths[idx]
                    
                    start_idx = 0
                    for i in range(idx):
                        start_idx += target_lengths[i]
                    
                    end_idx = start_idx + current_target_len
                    target_mask = mask[start_idx:end_idx]
                    if target_mask.any():
                        sample_correct = (predicted[start_idx:end_idx][target_mask] == targets[start_idx:end_idx][target_mask]).sum().item()
                        digit_acc[key]['correct'] += sample_correct
        
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

def run_exp1_architecture_comparison():
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
        
        experiment = Experiment(name, config)
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


def run_exp2_split_strategies():
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
            'arch_type': 'encoder_decoder',
            'split_strategy': strategy,
            'param_scale': 'medium',
            'seed': 42
        }
        
        experiment = Experiment(name, config)
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


def run_exp3_parameter_scales():
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
            'arch_type': 'encoder_decoder',
            'split_strategy': 'random',
            'param_scale': scale,
            'seed': 42
        }
        
        experiment = Experiment(name, config)
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


def run_single_experiment(name, config):
    experiment = Experiment(name, config)
    return experiment.run()


# ===============================
# 主函数
# ===============================

def main():
    """
    主函数：运行所有实验
    """
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    import sys
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        
        if exp_name == 'exp1':
            run_exp1_architecture_comparison()
        elif exp_name == 'exp2':
            run_exp2_split_strategies()
        elif exp_name == 'exp3':
            run_exp3_parameter_scales()
        else:
            print(f"未知实验: {exp_name}")
            print("可用实验: exp1, exp2, exp3")
    else:
        print("运行所有实验...")
        
        print("\n" + "="*60)
        print("开始完整实验流程")
        print("="*60)
        
        run_exp1_architecture_comparison()
        
        run_exp2_split_strategies()
        
        run_exp3_parameter_scales()
        
        print("\n" + "="*60)
        print("所有实验完成!")
        print("="*60)


if __name__ == '__main__':
    main()