#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
用于生成Task-1和Task-2的实验结果图表
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set font to English
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

# 基础路径
BASE_PATH = Path('/home/feng5u/桌面/Notes/Fudan NLP/Task-3')
RESULTS_PATH = BASE_PATH / 'results'
OUTPUT_PATH = BASE_PATH / 'docs' / 'img'

# 创建输出目录
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_plot(figsize=(10, 6), title=''):
    """设置图表"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    return fig, ax


def task1_exp1_architecture_comparison():
    """Task-1 Experiment 1: Architecture Comparison"""
    print("Generating Task-1 Experiment 1: Architecture Comparison...")

    # Define experiment directories
    exp_dirs = {
        'decoder_only': RESULTS_PATH / 'task1' / 'exp1_arch_decoder_only',
        'encoder_decoder': RESULTS_PATH / 'task1' / 'exp1_arch_encoder_decoder',
        'encoder_only': RESULTS_PATH / 'task1' / 'exp1_arch_encoder_only'
    }

    # Extract data from each experiment directory
    arch_types = []
    test_accuracies = []
    test_losses = []

    for arch, exp_dir in exp_dirs.items():
        test_results_path = exp_dir / 'test_results.json'
        if test_results_path.exists():
            data = load_json(test_results_path)
            arch_types.append(arch)
            test_accuracies.append(data['test_accuracy'])
            test_losses.append(data['test_loss'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Task-1 Experiment 1: Architecture Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Accuracy comparison
    colors1 = ['#3498db', '#2ecc71', '#e74c3c']
    bars1 = ax1.bar(arch_types, test_accuracies, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Architecture Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    colors2 = ['#e67e22', '#9b59b6', '#1abc9c']
    bars2 = ax2.bar(arch_types, test_losses, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Architecture Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task1_exp1_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task1_exp1_architecture_comparison.png")


def task1_exp2_split_strategies():
    """Task-1 Experiment 2: Split Strategies Comparison"""
    print("Generating Task-1 Experiment 2: Split Strategies Comparison...")

    data = load_json(RESULTS_PATH / 'task1' / 'exp2_split_strategies_summary.json')
    results = data['results']

    # Extract data
    strategies = []
    test_accuracies = []
    test_losses = []

    for strategy, metrics in results.items():
        strategies.append(strategy)
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Task-1 Experiment 2: Split Strategies Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Accuracy comparison
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    bars1 = ax1.bar(strategies, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Split Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticklabels(strategies, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Loss comparison
    bars2 = ax2.bar(strategies, test_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Split Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticklabels(strategies, rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task1_exp2_split_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task1_exp2_split_strategies.png")


def task1_exp3_parameter_scales():
    """Task-1 Experiment 3: Parameter Scales Comparison"""
    print("Generating Task-1 Experiment 3: Parameter Scales Comparison...")

    data = load_json(RESULTS_PATH / 'task1' / 'exp3_parameter_scales_summary.json')
    results = data['results']

    # Extract data
    scales = []
    test_accuracies = []
    test_losses = []

    for scale, metrics in results.items():
        scales.append(scale)
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Task-1 Experiment 3: Parameter Scales Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Accuracy comparison
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars1 = ax1.bar(scales, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    bars2 = ax2.bar(scales, test_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task1_exp3_parameter_scales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task1_exp3_parameter_scales.png")


def task1_exp3_digit_accuracy():
    """Task-1 Experiment 3: Digit-wise Accuracy Distribution"""
    print("Generating Task-1 Experiment 3: Digit-wise Accuracy Distribution...")

    data = load_json(RESULTS_PATH / 'task1' / 'exp3_parameter_scales_summary.json')
    results = data['results']

    # Extract digit-wise accuracy for different scales
    scales = ['small', 'medium', 'large']
    digit_pairs = None

    # Get all digit pairs
    for scale in scales:
        digit_acc = results[scale]['digit_accuracy']
        if digit_pairs is None:
            digit_pairs = sorted(digit_acc.keys())
        break

    # Create data matrix
    accuracy_matrix = []
    for scale in scales:
        row = []
        for dp in digit_pairs:
            acc = results[scale]['digit_accuracy'][dp]['accuracy']
            row.append(acc)
        accuracy_matrix.append(row)

    accuracy_matrix = np.array(accuracy_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set axes
    ax.set_xticks(np.arange(len(digit_pairs)))
    ax.set_yticks(np.arange(len(scales)))
    ax.set_xticklabels(digit_pairs, rotation=45, ha='right')
    ax.set_yticklabels(scales)

    # Add values on heatmap
    for i in range(len(scales)):
        for j in range(len(digit_pairs)):
            text = ax.text(j, i, f'{accuracy_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    ax.set_xlabel('Digit Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax.set_title('Task-1 Experiment 3: Digit-wise Accuracy Distribution (%)', fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task1_exp3_digit_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task1_exp3_digit_accuracy_heatmap.png")


def task2_exp1_architecture_comparison():
    """Task-2 Experiment 1: Architecture Comparison"""
    print("Generating Task-2 Experiment 1: Architecture Comparison...")

    data = load_json(RESULTS_PATH / 'task2' / 'exp1_architecture_comparison_summary.json')
    results = data['results']

    # Extract data
    arch_types = []
    test_ppls = []
    test_accuracies = []
    test_losses = []

    for arch, metrics in results.items():
        arch_types.append(arch)
        test_ppls.append(metrics['test_ppl'])
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task-2 Experiment 1: Architecture Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Perplexity comparison
    colors1 = ['#3498db', '#2ecc71']
    bars1 = ax1.bar(arch_types, test_ppls, color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Architecture Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Accuracy comparison
    colors2 = ['#e74c3c', '#9b59b6']
    bars2 = ax2.bar(arch_types, test_accuracies, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Architecture Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    colors3 = ['#f39c12', '#1abc9c']
    bars3 = ax3.bar(arch_types, test_losses, color=colors3, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Architecture Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task2_exp1_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task2_exp1_architecture_comparison.png")


def task2_exp2_tokenizer_comparison():
    """Task-2 Experiment 2: Tokenizer Comparison"""
    print("Generating Task-2 Experiment 2: Tokenizer Comparison...")

    data = load_json(RESULTS_PATH / 'task2' / 'exp2_tokenizer_comparison_summary.json')
    results = data['results']

    # Extract data
    tokenizers = []
    test_ppls = []
    test_accuracies = []
    test_losses = []

    for tokenizer, metrics in results.items():
        tokenizers.append(tokenizer)
        test_ppls.append(metrics['test_ppl'])
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task-2 Experiment 2: Tokenizer Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Perplexity comparison
    colors = ['#3498db', '#2ecc71']
    bars1 = ax1.bar(tokenizers, test_ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Tokenizer Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Accuracy comparison
    bars2 = ax2.bar(tokenizers, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Tokenizer Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    bars3 = ax3.bar(tokenizers, test_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Tokenizer Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task2_exp2_tokenizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task2_exp2_tokenizer_comparison.png")


def task2_exp3_parameter_scales():
    """Task-2 Experiment 3: Parameter Scales Comparison"""
    print("Generating Task-2 Experiment 3: Parameter Scales Comparison...")

    data = load_json(RESULTS_PATH / 'task2' / 'exp3_parameter_scales_summary.json')
    results = data['results']

    # Extract data
    scales = []
    test_ppls = []
    test_accuracies = []
    test_losses = []

    for scale, metrics in results.items():
        scales.append(scale)
        test_ppls.append(metrics['test_ppl'])
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task-2 Experiment 3: Parameter Scales Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Perplexity comparison
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars1 = ax1.bar(scales, test_ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Accuracy comparison
    bars2 = ax2.bar(scales, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    bars3 = ax3.bar(scales, test_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Parameter Scale', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task2_exp3_parameter_scales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task2_exp3_parameter_scales.png")


def task2_exp4_sequence_length():
    """Task-2 Experiment 4: Sequence Length Comparison"""
    print("Generating Task-2 Experiment 4: Sequence Length Comparison...")

    data = load_json(RESULTS_PATH / 'task2' / 'exp4_sequence_length_summary.json')
    results = data['results']

    # Extract data
    seq_lengths = []
    test_ppls = []
    test_accuracies = []
    test_losses = []

    for seq_len, metrics in results.items():
        seq_lengths.append(int(seq_len))
        test_ppls.append(metrics['test_ppl'])
        test_accuracies.append(metrics['test_accuracy'])
        test_losses.append(metrics['test_loss'])

    # Sort by sequence length
    indices = np.argsort(seq_lengths)
    seq_lengths = [seq_lengths[i] for i in indices]
    test_ppls = [test_ppls[i] for i in indices]
    test_accuracies = [test_accuracies[i] for i in indices]
    test_losses = [test_losses[i] for i in indices]

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task-2 Experiment 4: Sequence Length Comparison', fontsize=16, fontweight='bold', y=1.02)

    # Perplexity comparison
    colors = ['#3498db', '#2ecc71']
    bars1 = ax1.bar(range(len(seq_lengths)), test_ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(seq_lengths)))
    ax1.set_xticklabels(seq_lengths)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Accuracy comparison
    bars2 = ax2.bar(range(len(seq_lengths)), test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(seq_lengths)))
    ax2.set_xticklabels(seq_lengths)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss comparison
    bars3 = ax3.bar(range(len(seq_lengths)), test_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Loss Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(seq_lengths)))
    ax3.set_xticklabels(seq_lengths)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task2_exp4_sequence_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task2_exp4_sequence_length.png")


def task2_comprehensive_summary():
    """Task-2 Comprehensive Summary: All Experiments Metrics"""
    print("Generating Task-2 Comprehensive Summary...")

    # Collect data from all experiments
    experiments = {}

    # Exp2: Tokenizer Comparison
    try:
        data = load_json(RESULTS_PATH / 'task2' / 'exp2_tokenizer_comparison_summary.json')
        experiments['Tokenizer\n(Exp2)'] = {
            config: metrics['test_ppl']
            for config, metrics in data['results'].items()
        }
    except:
        print("Warning: Could not load exp2_tokenizer_comparison_summary.json")

    # Exp3: Parameter Scales Comparison
    try:
        data = load_json(RESULTS_PATH / 'task2' / 'exp3_parameter_scales_summary.json')
        experiments['Scale\n(Exp3)'] = {
            config: metrics['test_ppl']
            for config, metrics in data['results'].items()
        }
    except:
        print("Warning: Could not load exp3_parameter_scales_summary.json")

    # Exp4: Sequence Length Comparison
    try:
        data = load_json(RESULTS_PATH / 'task2' / 'exp4_sequence_length_summary.json')
        experiments['SeqLen\n(Exp4)'] = {
            config: metrics['test_ppl']
            for config, metrics in data['results'].items()
        }
    except:
        print("Warning: Could not load exp4_sequence_length_summary.json")

    # Extract all configs and perplexities
    all_configs = []
    all_ppls = []

    for exp_name, configs in experiments.items():
        for config, ppl in configs.items():
            all_configs.append(f'{exp_name}\n{config}')
            all_ppls.append(ppl)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Set colors based on perplexity (lower is better, use green)
    max_ppl = max(all_ppls)
    min_ppl = min(all_ppls)
    colors = []
    for ppl in all_ppls:
        # Normalize to 0-1
        norm = (ppl - min_ppl) / (max_ppl - min_ppl + 1e-6)
        # Lower perplexity = more green, higher = more red
        colors.append(plt.cm.RdYlGn_r(norm))

    bars = ax.bar(range(len(all_configs)), all_ppls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Experiment Config', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (PPL)', fontsize=12, fontweight='bold')
    ax.set_title('Task-2 Comprehensive Summary: Perplexity of All Experiments', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(all_configs)))
    ax.set_xticklabels(all_configs, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'task2_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: task2_comprehensive_summary.png")


def relative_position_architecture_comparison():
    """相对位置编码架构对比：decoder_only（绝对位置vs相对位置）和 encoder_decoder（相对位置）"""
    print("Generating Relative Position Architecture Comparison...")

    # 加载数据
    abs_pos_decoder_only = load_json(RESULTS_PATH / 'task1' / 'exp2_split_strategies_summary.json')
    rel_pos_decoder_only = load_json(Path(str(RESULTS_PATH).replace('results', 'results（相对位置编码_decoder_only）')) / 'exp2_split_strategies_summary.json')
    rel_pos_encoder_decoder = load_json(Path(str(RESULTS_PATH).replace('results', 'results（相对位置编码_encoder_decoder）')) / 'exp2_split_strategies_summary.json')

    # 提取数据（使用carry_complexity策略，因为这是最成功的策略）
    strategies = ['carry_complexity', 'digit_pair', 'max_digits', 'result_range']
    strategy_labels = ['Carry\nComplexity', 'Digit\nPair', 'Max\nDigits', 'Result\nRange']

    # 准备数据
    abs_acc = []
    abs_loss = []
    rel_dec_acc = []
    rel_dec_loss = []
    rel_enc_dec_acc = []
    rel_enc_dec_loss = []

    for strategy in strategies:
        if strategy in abs_pos_decoder_only['results']:
            abs_acc.append(abs_pos_decoder_only['results'][strategy]['test_accuracy'])
            abs_loss.append(abs_pos_decoder_only['results'][strategy]['test_loss'])
        else:
            abs_acc.append(0)
            abs_loss.append(0)

        if strategy in rel_pos_decoder_only['results']:
            rel_dec_acc.append(rel_pos_decoder_only['results'][strategy]['test_accuracy'])
            rel_dec_loss.append(rel_pos_decoder_only['results'][strategy]['test_loss'])
        else:
            rel_dec_acc.append(0)
            rel_dec_loss.append(0)

        if strategy in rel_pos_encoder_decoder['results']:
            rel_enc_dec_acc.append(rel_pos_encoder_decoder['results'][strategy]['test_accuracy'])
            rel_enc_dec_loss.append(rel_pos_encoder_decoder['results'][strategy]['test_loss'])
        else:
            rel_enc_dec_acc.append(0)
            rel_enc_dec_loss.append(0)

    # 创建图表 - 增加垂直间距
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, top=0.85)  # 增加子图间距，为图例留出空间
    fig.suptitle('Relative Position Encoding: Architecture Comparison', fontsize=16, fontweight='bold', y=0.98)

    # 设置条形位置
    x = np.arange(len(strategies))
    width = 0.25

    # 统一使用单一颜色方案，避免混淆
    color_abs = '#3498db'      # 蓝色 - 绝对位置编码
    color_rel_dec = '#2ecc71'  # 绿色 - 相对位置编码 (decoder_only)
    color_rel_enc = '#e67e22'  # 橙色 - 相对位置编码 (encoder_decoder)

    # 准确率对比
    bars1 = ax1.bar(x - width, abs_acc, width, label='Decoder-Only\n(Absolute Pos)', color=color_abs, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, rel_dec_acc, width, label='Decoder-Only\n(Relative Pos)', color=color_rel_dec, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, rel_enc_dec_acc, width, label='Encoder-Decoder\n(Relative Pos)', color=color_rel_enc, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Split Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_labels, fontsize=11)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=10, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 105)

    # 添加数值标签 - 调整位置避免重叠
    for bars in [bars1, bars2, bars3]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # 根据柱子高度动态调整标签位置
                if height < 30:
                    va = 'top'
                    y_offset = 5
                else:
                    va = 'bottom'
                    y_offset = 2
                ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                        f'{height:.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    # 损失对比
    bars4 = ax2.bar(x - width, abs_loss, width, color=color_abs, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars5 = ax2.bar(x, rel_dec_loss, width, color=color_rel_dec, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars6 = ax2.bar(x + width, rel_enc_dec_loss, width, color=color_rel_enc, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Split Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_labels, fontsize=11)
    # 移除子图图例，因为已经在上方统一显示了
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签 - 调整位置避免重叠
    for bar_idx, bars in enumerate([bars4, bars5, bars6]):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # 根据柱子索引调整水平位置，避免标签重叠
                # bar_idx: 0=绝对位置, 1=相对位置(decoder), 2=相对位置(encoder-decoder)
                if bar_idx == 0:
                    ha = 'left'  # 第一个柱子标签左对齐
                elif bar_idx == 1:
                    ha = 'center'  # 中间柱子标签居中
                else:
                    ha = 'right'  # 最后一个柱子标签右对齐

                # 添加微小的垂直偏移
                y_offset = 0.1 if height > 1 else 0.05

                ax2.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                        f'{height:.3f}', ha=ha, va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.savefig(OUTPUT_PATH / 'relative_position_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: relative_position_architecture_comparison.png")


def relative_position_random_split_comparison():
    """Random Split 策略下的位置编码对比：绝对位置编码(exp1) vs 相对位置编码(exp2)，架构均为 encoder_decoder"""
    print("Generating Relative Position Random Split Comparison...")

    # 加载数据
    # 绝对位置编码下的 exp1，架构为 encoder_decoder
    abs_pos_exp1 = load_json(RESULTS_PATH / 'task1' / 'exp1_arch_encoder_decoder' / 'test_results.json')
    # 相对位置编码下的 exp2，架构为 encoder_decoder
    rel_pos_exp2 = load_json(Path(str(RESULTS_PATH).replace('results', 'results（相对位置编码_encoder_decoder）')) / 'exp2_split_random' / 'test_results.json')

    # 提取数据
    configs = ['Encoder-Decoder\n(Absolute Pos, Exp1)', 'Encoder-Decoder\n(Relative Pos, Exp2)']
    accuracies = [abs_pos_exp1['test_accuracy'], rel_pos_exp2['test_accuracy']]
    losses = [abs_pos_exp1['test_loss'], rel_pos_exp2['test_loss']]

    # 提取数字级别的准确率
    digit_pairs = sorted(abs_pos_exp1['digit_accuracy'].keys())
    abs_digit_acc = []
    rel_digit_acc = []

    for dp in digit_pairs:
        abs_digit_acc.append(abs_pos_exp1['digit_accuracy'][dp]['accuracy'])
        rel_digit_acc.append(rel_pos_exp2['digit_accuracy'][dp]['accuracy'])

    # 创建图表 - 增加水平间距
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.3, top=0.85, bottom=0.2)  # 增加子图间距，为标题和x轴标签留出空间
    fig.suptitle('Random Split Strategy: Position Encoding Comparison (Encoder-Decoder Architecture)', fontsize=16, fontweight='bold', y=0.95)

    # 统一颜色方案
    color_abs = '#3498db'      # 蓝色 - 绝对位置编码
    color_rel = '#2ecc71'      # 绿色 - 相对位置编码

    # 整体准确率对比
    bars1 = ax1.bar(configs, accuracies, color=[color_abs, color_rel], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Test Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='x', labelsize=10)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 损失对比
    bars2 = ax2.bar(configs, losses, color=[color_abs, color_rel], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', labelsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 数字级别准确率对比（折线图）
    x = np.arange(len(digit_pairs))
    line1 = ax3.plot(x, abs_digit_acc, marker='o', linewidth=2.5, markersize=7, label='Absolute Position (Exp1)', color=color_abs, zorder=5)[0]
    line2 = ax3.plot(x, rel_digit_acc, marker='s', linewidth=2.5, markersize=7, label='Relative Position (Exp2)', color=color_rel, zorder=5)[0]

    ax3.set_xlabel('Digit Pairs', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Digit Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Digit-wise Accuracy', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(digit_pairs, rotation=45, ha='right', fontsize=9)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.grid(alpha=0.3, linestyle='--', zorder=1)
    ax3.set_ylim(0, 105)

    # 添加数据标签，避免重叠
    for i, (abs_val, rel_val) in enumerate(zip(abs_digit_acc, rel_digit_acc)):
        # 只在关键点添加标签，避免 overcrowding
        if i % 3 == 0 or i == len(digit_pairs) - 1:  # 每3个点显示一次
            offset = 3
            ax3.text(x[i], abs_val + offset, f'{abs_val:.0f}', ha='center', va='bottom',
                    fontsize=7, color=color_abs, fontweight='bold', zorder=6)
            ax3.text(x[i], rel_val + offset, f'{rel_val:.0f}', ha='center', va='bottom',
                    fontsize=7, color=color_rel, fontweight='bold', zorder=6)

    plt.savefig(OUTPUT_PATH / 'relative_position_random_split_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: relative_position_random_split_comparison.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Starting Visualization Generation")
    print("=" * 60)
    print()

    # Task-1 plots
    print("Task-1: Multi-digit Addition Task")
    print("-" * 60)
    task1_exp1_architecture_comparison()
    task1_exp2_split_strategies()
    task1_exp3_parameter_scales()
    task1_exp3_digit_accuracy()
    print()

    # Task-2 plots (excluding architecture comparison)
    print("Task-2: Language Modeling Task")
    print("-" * 60)
    # Skip architecture comparison: task2_exp1_architecture_comparison()
    task2_exp2_tokenizer_comparison()
    task2_exp3_parameter_scales()
    task2_exp4_sequence_length()
    task2_comprehensive_summary()
    print()

    # Relative position encoding comparison plots
    print("Relative Position Encoding Comparison")
    print("-" * 60)
    relative_position_architecture_comparison()
    relative_position_random_split_comparison()
    print()

    print("=" * 60)
    print(f"All plots saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()