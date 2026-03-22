import os
import json
import copy
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Force HuggingFace to use offline mode to avoid SSL/connection errors
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Data.dataset.lm_dataset import LMDataset
from Data.processors.lm_processor import LMProcessor
from transformer.models.decoder_only import make_decoder_only_model
from transformer.models.encoder_only import make_encoder_only_model
from transformer.utils.masking import padding_mask, subsequent_mask

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

PARAM_CONFIGS = {
    'tiny': {
        'N': 2,
        'd_model': 128,
        'd_ff': 512,
        'h': 4,
        'dropout': 0.1
    },
    'small': {
        'N': 4,
        'd_model': 256,
        'd_ff': 1024,
        'h': 8,
        'dropout': 0.1
    },
    'medium': {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'dropout': 0.1
    },
    'base': {
        'N': 12,
        'd_model': 768,
        'd_ff': 3072,
        'h': 12,
        'dropout': 0.1
    }
}

TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.0001,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'clip_grad': 1.0,
    'gradient_accumulation_steps': 1
}

TOKENIZER_CONFIGS = {
    'gpt2': {
        'model_name': 'gpt2',
        'vocab_sizes': [50257, 25000, 10000, 5000]
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'vocab_sizes': [30522, 15000, 8000, 4000]
    },
    'roberta': {
        'model_name': 'roberta-base',
        'vocab_sizes': [50265, 25000, 10000, 5000]
    }
}

SEQUENCE_LENGTHS = [64, 128, 256, 512]


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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(arch_type, vocab_size, param_config, use_relative_position=False, max_relative_position=127):
    """
    创建指定架构的模型
    
    参数：
        arch_type: 'decoder_only', 'encoder_only'
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
    
    if arch_type == 'decoder_only':
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


class LanguageModelingExperiment:
    """
    语言建模实验类：管理完整的实验流程
    """
    
    def __init__(self, name, config, device='cpu', use_multi_gpu=False, gpu_ids=None):
        """
        参数：
            name: 实验名称
            config: 实验配置字典
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
            'results', 'task2', name
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def setup_data(self):
        print(f"\n[{self.name}] 设置数据...")
        
        tokenizer_type = self.config.get('tokenizer_type', 'gpt2')
        max_length = self.config.get('max_length', 128)
        vocab_size = self.config.get('vocab_size', None)
        num_samples = self.config.get('num_samples', None)
        
        dataset_config = {
            'num_samples': num_samples
        }
        
        self.train_dataset = LMDataset(dataset_config, split='train')
        self.val_dataset = LMDataset(dataset_config, split='validation')
        self.test_dataset = LMDataset(dataset_config, split='test')
        
        if len(self.train_dataset) == 0:
            raise ValueError("训练数据集为空")
        if len(self.val_dataset) == 0:
            raise ValueError("验证数据集为空")
        if len(self.test_dataset) == 0:
            raise ValueError("测试数据集为空")
        
        processor_config = {
            'tokenizer_type': tokenizer_type,
            'max_length': max_length,
            'vocab_size': vocab_size
        }
        
        self.processor = LMProcessor(self.train_dataset, processor_config)
        
        batch_size = TRAIN_CONFIG['batch_size']
        
        self.train_loader = self.processor.get_loader(
            self.train_dataset, batch_size, shuffle=True,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        self.val_loader = self.processor.get_loader(
            self.val_dataset, batch_size, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        self.test_loader = self.processor.get_loader(
            self.test_dataset, batch_size, shuffle=False,
            num_workers=4, pin_memory=torch.cuda.is_available()
        )
        
        stats = {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'vocab_size': len(self.processor.tokenizer),
            'max_length': max_length,
            'tokenizer_type': tokenizer_type
        }
        
        print(f"数据统计: {json.dumps(stats, indent=2)}")
        return stats
    
    def setup_model(self):
        print(f"\n[{self.name}] 设置模型...")
        
        arch_type = self.config['arch_type']
        param_scale = self.config['param_scale']
        param_config = PARAM_CONFIGS[param_scale]
        vocab_size = len(self.processor.tokenizer)
        
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
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        num_training_steps = len(self.train_loader) * TRAIN_CONFIG['epochs']
        num_warmup_steps = int(num_training_steps * TRAIN_CONFIG['warmup_ratio'])
        
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=num_warmup_steps
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)
        
        model_config = {
            'arch_type': arch_type,
            'param_scale': param_scale,
            'vocab_size': vocab_size,
            'param_config': param_config,
            'use_relative_position': use_relative_position,
            'max_relative_position': max_relative_position if use_relative_position else None,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_training_steps': num_training_steps,
            'num_warmup_steps': num_warmup_steps,
            'device': str(self.device),
            'use_multi_gpu': self.use_multi_gpu,
            'gpu_ids': self.gpu_ids
        }
        
        with open(os.path.join(self.save_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)
        
        return model_config
    
    def make_mask(self, seq, pad_idx):
        mask = padding_mask(seq, pad_idx)
        if self.config['arch_type'] == 'decoder_only':
            mask = mask & subsequent_mask(seq.size(1)).to(seq.device)
        return mask
    
    def forward_pass(self, batch, training=True):
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        pad_idx = self.processor.tokenizer.pad_token_id
        
        if self.config['arch_type'] == 'decoder_only':
            src = input_ids[:, :-1]
            mask = self.make_mask(src, pad_idx)
            outputs = self.model(src, mask)
            
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = target_ids[:, :-1].reshape(-1)
            
        elif self.config['arch_type'] == 'encoder_only':
            src = input_ids
            mask = self.make_mask(src, pad_idx)
            
            if training:
                mlm_mask = torch.rand_like(src.float()) < 0.15
                mlm_mask = mlm_mask & (src != pad_idx)
                mlm_inputs = src.clone()
                mlm_inputs[mlm_mask] = self.processor.tokenizer.mask_token_id if self.processor.tokenizer.mask_token_id is not None else 103
            else:
                mlm_inputs = src
                mlm_mask = (src != pad_idx)
            
            outputs = self.model(mlm_inputs, mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = src.reshape(-1)
            
            outputs = outputs[mlm_mask.reshape(-1)]
            targets = targets[mlm_mask.reshape(-1)]
        
        return outputs, targets
    
    def compute_metrics(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        
        ppl = torch.exp(loss)
        
        _, predicted = outputs.max(1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        acc = 100. * correct / total
        
        top5_correct = 0
        top5_preds = outputs.topk(5, dim=1).indices
        for i in range(targets.size(0)):
            if targets[i] in top5_preds[i]:
                top5_correct += 1
        top5_acc = 100. * top5_correct / total
        
        return loss, ppl, acc, top5_acc
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_ppl = 0
        total_acc = 0
        total_top5_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            outputs, targets = self.forward_pass(batch, training=True)
            
            loss, ppl, acc, top5_acc = self.compute_metrics(outputs, targets)
            
            loss = loss / TRAIN_CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % TRAIN_CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), TRAIN_CONFIG['clip_grad']
                )
                self.optimizer.step()
                self.scheduler.step()
            
            total_loss += loss.item() * TRAIN_CONFIG['gradient_accumulation_steps']
            total_ppl += ppl.item()
            total_acc += acc
            total_top5_acc += top5_acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item()*TRAIN_CONFIG["gradient_accumulation_steps"]:.4f}',
                'ppl': f'{ppl.item():.2f}',
                'acc': f'{acc:.2f}%'
            })
        
        avg_loss = total_loss / num_batches
        avg_ppl = total_ppl / num_batches
        avg_acc = total_acc / num_batches
        avg_top5_acc = total_top5_acc / num_batches
        
        return avg_loss, avg_ppl, avg_acc, avg_top5_acc
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_ppl = 0
        total_acc = 0
        total_top5_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                outputs, targets = self.forward_pass(batch, training=False)
                
                loss, ppl, acc, top5_acc = self.compute_metrics(outputs, targets)
                
                total_loss += loss.item()
                total_ppl += ppl.item()
                total_acc += acc
                total_top5_acc += top5_acc
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ppl = total_ppl / num_batches
        avg_acc = total_acc / num_batches
        avg_top5_acc = total_top5_acc / num_batches
        
        return avg_loss, avg_ppl, avg_acc, avg_top5_acc
    
    def train(self):
        print(f"\n[{self.name}] 开始训练...")
        
        best_val_ppl = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
            train_loss, train_ppl, train_acc, train_top5_acc = self.train_epoch(epoch)
            
            val_loss, val_ppl, val_acc, val_top5_acc = self.evaluate(self.val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, PPL={train_ppl:.2f}, Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Acc={val_acc:.2f}%")
            
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
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
        print(f"\n[{self.name}] 开始测试...")
        
        self.load_model('best_model.pt')
        
        test_loss, test_ppl, test_acc, test_top5_acc = self.evaluate(self.test_loader)
        
        print(f"测试结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  困惑度: {test_ppl:.2f}")
        print(f"  准确率: {test_acc:.2f}%")
        print(f"  Top-5 准确率: {test_top5_acc:.2f}%")
        
        bpc = test_loss / math.log(2)
        print(f"  Bits Per Character: {bpc:.4f}")
        
        results = {
            'test_loss': test_loss,
            'test_ppl': test_ppl,
            'test_accuracy': test_acc,
            'test_top5_accuracy': test_top5_acc,
            'test_bpc': bpc,
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def run(self):
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


def run_exp1_architecture_comparison(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 1: 架构对比")
    print("="*60)

    architectures = ['decoder_only', 'encoder_only']

    results = {}
    for arch in architectures:
        name = f"exp1_arch_{arch}"
        config = {
            'arch_type': arch,
            'tokenizer_type': 'gpt2',
            'param_scale': 'small',
            'max_length': 128,
            'num_samples': 500000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[arch] = experiment.run()
    
    summary = {
        'experiment': 'exp1_architecture_comparison',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp1_architecture_comparison_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp2_tokenizer_comparison(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 2: Tokenizer 对比")
    print("="*60)

    tokenizers = ['gpt2', 'bert', 'roberta']

    results = {}
    for tokenizer in tokenizers:
        name = f"exp2_tokenizer_{tokenizer}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': tokenizer,
            'param_scale': 'small',
            'max_length': 128,
            'num_samples': 500000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[tokenizer] = experiment.run()
    
    summary = {
        'experiment': 'exp2_tokenizer_comparison',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp2_tokenizer_comparison_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp3_vocab_size(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 3: 词表大小影响")
    print("="*60)

    vocab_sizes = [50257, 25000, 10000, 5000]

    results = {}
    for vocab_size in vocab_sizes:
        name = f"exp3_vocab_{vocab_size}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': 'gpt2',
            'vocab_size': vocab_size,
            'param_scale': 'small',
            'max_length': 128,
            'num_samples': 500000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[vocab_size] = experiment.run()
    
    summary = {
        'experiment': 'exp3_vocab_size',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp3_vocab_size_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp4_parameter_scales(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 4: 参数规模影响")
    print("="*60)

    scales = ['tiny', 'small', 'medium', 'base']

    results = {}
    for scale in scales:
        name = f"exp4_scale_{scale}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': 'gpt2',
            'param_scale': scale,
            'max_length': 128,
            'num_samples': 500000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[scale] = experiment.run()
    
    summary = {
        'experiment': 'exp4_parameter_scales',
        'results': results
    }
    
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp4_parameter_scales_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def run_exp5_sequence_length(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 5: 序列长度影响")
    print("="*60)

    seq_lengths = [64, 128, 256]

    results = {}
    for length in seq_lengths:
        name = f"exp5_seq_{length}"
        config = {
            'arch_type': 'decoder_only',
            'tokenizer_type': 'gpt2',
            'param_scale': 'small',
            'max_length': length,
            'num_samples': 500000,
            'seed': 42
        }

        experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        results[length] = experiment.run()

    summary = {
        'experiment': 'exp5_sequence_length',
        'results': results
    }

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp5_sequence_length_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def run_exp6_best_model(device, use_multi_gpu=False, gpu_ids=None):
    print("\n" + "="*60)
    print("实验 6: 最佳配置组合")
    print("="*60)

    name = "exp6_best_model"
    config = {
        'arch_type': 'decoder_only',
        'tokenizer_type': 'gpt2',
        'param_scale': 'medium',
        'max_length': 512,
        'num_samples': 1000000,
        'seed': 42
    }

    experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
    results = experiment.run()

    summary = {
        'experiment': 'exp6_best_model',
        'results': results
    }

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'results', 'task2', 'exp6_best_model_summary.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def run_single_experiment(name, config, device='cpu', use_multi_gpu=False, gpu_ids=None):
    experiment = LanguageModelingExperiment(name, config, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
    return experiment.run()


def main():
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
            run_exp2_tokenizer_comparison(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp3':
            run_exp3_vocab_size(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp4':
            run_exp4_parameter_scales(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp5':
            run_exp5_sequence_length(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        elif exp_name == 'exp6':
            run_exp6_best_model(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        else:
            print(f"未知实验: {exp_name}")
            print("可用实验: exp1, exp2, exp3, exp4, exp5, exp6")
    else:
        print("运行所有实验...")

        print("\n" + "="*60)
        print("开始完整实验流程")
        print("="*60)

        run_exp1_architecture_comparison(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        run_exp2_tokenizer_comparison(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        run_exp3_vocab_size(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        run_exp4_parameter_scales(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        run_exp5_sequence_length(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)
        run_exp6_best_model(device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)

        print("\n" + "="*60)
        print("所有实验完成!")
        print("="*60)


if __name__ == '__main__':
    main()