import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 准备数据
configurations = [
    "(1,1)-1000", "(1,1)-5000", "(1,1)-10000", "(1,1)-20000",
    "(1,2)-1000", "(1,2)-5000", "(1,2)-10000", "(1,2)-20000",
    "(1,3)-1000", "(1,3)-5000", "(1,3)-10000", "(1,3)-20000",
    "(2,2)-1000", "(2,2)-5000", "(2,2)-10000", "(2,2)-20000",
    "(2,3)-1000", "(2,3)-5000", "(2,3)-10000", "(2,3)-20000",
    "(3,3)-1000", "(3,3)-5000", "(3,3)-10000", "(3,3)-20000"
]

test_accuracies = [
    0.4548, 0.4748, 0.4781, 0.4784,  # (1,1)
    0.4421, 0.4772, 0.4817, 0.4811,  # (1,2)
    0.4400, 0.4723, 0.4811, 0.4814,  # (1,3)
    0.3838, 0.4143, 0.4152, 0.4146,  # (2,2)
    0.3793, 0.4056, 0.4086, 0.4068,  # (2,3)
    0.3270, 0.3358, 0.3382, 0.3409   # (3,3)
]

time_costs = [
    83.83, 131.19, 132.00, 134.00,  # (1,1)
    70.72, 102.22, 106.96, 119.25,  # (1,2)
    69.81, 95.64, 103.31, 106.30,   # (1,3)
    97.54, 130.28, 131.65, 135.58,  # (2,2)
    92.70, 115.70, 128.33, 135.87,  # (2,3)
    129.05, 130.57, 132.29, 135.07  # (3,3)
]

early_stop_epochs = [
    644, 1000, 1000, 1000,   # (1,1)
    553, 792, 814, 882,      # (1,2)
    547, 738, 779, 783,      # (1,3)
    763, 1000, 1000, 1000,   # (2,2)
    719, 877, 967, 1000,     # (2,3)
    1000, 1000, 1000, 1000   # (3,3)
]

# 创建DataFrame
df = pd.DataFrame({
    'config': configurations,
    'test_accuracy': test_accuracies,
    'time_cost': time_costs,
    'early_stop_epoch': early_stop_epochs
})

# 提取配置信息
df['n_gram'] = df['config'].apply(lambda x: x.split('-')[0])
df['features'] = df['config'].apply(lambda x: int(x.split('-')[1]))
df['n_min'] = df['n_gram'].apply(lambda x: int(x[1]))
df['n_max'] = df['n_gram'].apply(lambda x: int(x[3]))
df['n_sum'] = df['n_min'] + df['n_max']  # 提前计算n_sum

# 图表1：模型性能对比条形图
plt.figure(figsize=(14, 8))
df_sorted = df.sort_values('test_accuracy', ascending=True)
bars = plt.barh(range(len(df_sorted)), df_sorted['test_accuracy'], 
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted))))
plt.yticks(range(len(df_sorted)), df_sorted['config'])
plt.xlabel('Test Accuracy')
plt.title('Model Performance Comparison by Configuration')
plt.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (bar, acc) in enumerate(zip(bars, df_sorted['test_accuracy'])):
    plt.text(acc + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# 图表2：特征数量对性能的影响
plt.figure(figsize=(12, 7))
feature_counts = [1000, 5000, 10000, 20000]
n_gram_configs = ['(1,1)', '(1,2)', '(1,3)', '(2,2)', '(2,3)', '(3,3)']
markers = ['o', 's', '^', 'D', 'v', 'p']

for idx, n_gram in enumerate(n_gram_configs):
    subset = df[df['n_gram'] == n_gram].sort_values('features')
    plt.plot(subset['features'], subset['test_accuracy'], 
             marker=markers[idx], markersize=8, linewidth=2,
             label=f'n={n_gram}')

plt.xlabel('Max Features')
plt.ylabel('Test Accuracy')
plt.title('Impact of Feature Count on Model Performance')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks([1000, 5000, 10000, 20000])
plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.tight_layout()
plt.show()

# 图表3：n-gram复杂度与性能关系
plt.figure(figsize=(12, 7))

# 按n-gram配置分组并排序
n_gram_order = ['(1,1)', '(1,2)', '(1,3)', '(2,2)', '(2,3)', '(3,3)']
colors = plt.cm.Set2(np.linspace(0, 1, len(feature_counts)))

for idx, feature_count in enumerate(feature_counts):
    subset = df[df['features'] == feature_count]
    # 确保按正确的顺序排列
    subset = subset.set_index('n_gram').loc[n_gram_order].reset_index()
    plt.plot(range(len(n_gram_order)), subset['test_accuracy'], 
             marker='o', markersize=8, linewidth=2,
             color=colors[idx], label=f'{feature_count} features')

plt.xlabel('N-gram Configuration')
plt.ylabel('Test Accuracy')
plt.title('N-gram Configuration vs Performance')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.xticks(range(len(n_gram_order)), n_gram_order)
plt.tight_layout()
plt.show()

# 图表4：训练时间与特征数量关系
plt.figure(figsize=(12, 7))

for idx, n_gram in enumerate(n_gram_configs):
    subset = df[df['n_gram'] == n_gram].sort_values('features')
    plt.plot(subset['features'], subset['time_cost'], 
             marker=markers[idx], markersize=8, linewidth=2,
             label=f'n={n_gram}')

plt.xlabel('Max Features')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Feature Count')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks([1000, 5000, 10000, 20000])
plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
plt.tight_layout()
plt.show()

# 图表5：最优配置与数据分布散点图
plt.figure(figsize=(12, 8))

# 使用n_sum作为x坐标
scatter = plt.scatter(df['n_sum'], df['features'], 
                     c=df['test_accuracy'], s=df['test_accuracy']*300, 
                     alpha=0.7, cmap='plasma', edgecolors='black', linewidth=0.5)

plt.xlabel('N-gram Sum (n_min + n_max)')
plt.ylabel('Max Features')
plt.title('Optimal Configurations\n(Size and Color = Test Accuracy)')
plt.grid(True, alpha=0.3)

# 设置坐标轴刻度
n_sums = sorted(df['n_sum'].unique())
plt.xticks(n_sums)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Test Accuracy')

# 标注前5个最佳配置
top_5 = df.nlargest(5, 'test_accuracy')
for _, row in top_5.iterrows():
    plt.annotate(f"{row['n_gram']}\n{row['test_accuracy']:.4f}", 
                (row['n_sum'], row['features']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, ha='left', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# 图表6：验证集早停轮次分析图
plt.figure(figsize=(14, 8))

x = np.arange(len(configurations))
width = 0.35

fig, ax1 = plt.subplots(figsize=(14, 8))

# 条形图：早停轮次
bars1 = ax1.bar(x - width/2, df['early_stop_epoch'], width, 
               alpha=0.6, color='skyblue', label='Early Stop Epoch')

# 折线图：测试准确率
ax2 = ax1.twinx()
line = ax2.plot(x + width/2, df['test_accuracy'], 
               color='coral', marker='o', linewidth=2, 
               label='Test Accuracy')

ax1.set_xlabel('Configuration')
ax1.set_ylabel('Early Stop Epoch', color='skyblue')
ax2.set_ylabel('Test Accuracy', color='coral')

# 设置x轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(configurations, rotation=45, ha='right', fontsize=9)

plt.title('Early Stopping Analysis')
ax1.grid(True, alpha=0.3, axis='y')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()

# results
# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (1, 1)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.3041, Train Acc: 0.4736
#     Val Loss: 1.3781, Val Acc: 0.4162

# Epoch 200 / 1000:
#     Train Loss: 1.2313, Train Acc: 0.5128
#     Val Loss: 1.3467, Val Acc: 0.4326

# Epoch 300 / 1000:
#     Train Loss: 1.1841, Train Acc: 0.5319
#     Val Loss: 1.3295, Val Acc: 0.4385

# Epoch 400 / 1000:
#     Train Loss: 1.1493, Train Acc: 0.5496
#     Val Loss: 1.3192, Val Acc: 0.4402

# Epoch 500 / 1000:
#     Train Loss: 1.1214, Train Acc: 0.5606
#     Val Loss: 1.3131, Val Acc: 0.4426

# Epoch 600 / 1000:
#     Train Loss: 1.0988, Train Acc: 0.5682
#     Val Loss: 1.3095, Val Acc: 0.4467

# Early stopping triggered at epoch 644

# Training completed!
# Final Test Loss: 1.2982
# Final Test Accuracy: 0.4548
# Time Cost: 83.83213233947754 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (1, 1)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2630, Train Acc: 0.5063
#     Val Loss: 1.3664, Val Acc: 0.4279

# Epoch 200 / 1000:
#     Train Loss: 1.1590, Train Acc: 0.5664
#     Val Loss: 1.3266, Val Acc: 0.4525

# Epoch 300 / 1000:
#     Train Loss: 1.0842, Train Acc: 0.6056
#     Val Loss: 1.3027, Val Acc: 0.4619

# Epoch 400 / 1000:
#     Train Loss: 1.0250, Train Acc: 0.6407
#     Val Loss: 1.2870, Val Acc: 0.4642

# Epoch 500 / 1000:
#     Train Loss: 0.9760, Train Acc: 0.6638
#     Val Loss: 1.2762, Val Acc: 0.4631

# Epoch 600 / 1000:
#     Train Loss: 0.9352, Train Acc: 0.6848
#     Val Loss: 1.2687, Val Acc: 0.4619

# Epoch 700 / 1000:
#     Train Loss: 0.8990, Train Acc: 0.7072
#     Val Loss: 1.2635, Val Acc: 0.4625

# Epoch 800 / 1000:
#     Train Loss: 0.8666, Train Acc: 0.7234
#     Val Loss: 1.2599, Val Acc: 0.4654

# Epoch 900 / 1000:
#     Train Loss: 0.8376, Train Acc: 0.7373
#     Val Loss: 1.2577, Val Acc: 0.4666

# Epoch 1000 / 1000:
#     Train Loss: 0.8115, Train Acc: 0.7510
#     Val Loss: 1.2563, Val Acc: 0.4672


# Training completed!
# Final Test Loss: 1.2617
# Final Test Accuracy: 0.4748
# Time Cost: 131.18973588943481 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (1, 1)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2526, Train Acc: 0.5148
#     Val Loss: 1.3657, Val Acc: 0.4250

# Epoch 200 / 1000:
#     Train Loss: 1.1393, Train Acc: 0.5819
#     Val Loss: 1.3255, Val Acc: 0.4537

# Epoch 300 / 1000:
#     Train Loss: 1.0564, Train Acc: 0.6313
#     Val Loss: 1.3013, Val Acc: 0.4683

# Epoch 400 / 1000:
#     Train Loss: 0.9920, Train Acc: 0.6682
#     Val Loss: 1.2853, Val Acc: 0.4660

# Epoch 500 / 1000:
#     Train Loss: 0.9369, Train Acc: 0.6969
#     Val Loss: 1.2743, Val Acc: 0.4683

# Epoch 600 / 1000:
#     Train Loss: 0.8907, Train Acc: 0.7231
#     Val Loss: 1.2665, Val Acc: 0.4660

# Epoch 700 / 1000:
#     Train Loss: 0.8506, Train Acc: 0.7451
#     Val Loss: 1.2610, Val Acc: 0.4637

# Epoch 800 / 1000:
#     Train Loss: 0.8149, Train Acc: 0.7630
#     Val Loss: 1.2573, Val Acc: 0.4648

# Epoch 900 / 1000:
#     Train Loss: 0.7825, Train Acc: 0.7804
#     Val Loss: 1.2548, Val Acc: 0.4707

# Epoch 1000 / 1000:
#     Train Loss: 0.7533, Train Acc: 0.7955
#     Val Loss: 1.2533, Val Acc: 0.4760


# Training completed!
# Final Test Loss: 1.2568
# Final Test Accuracy: 0.4781
# Time Cost: 131.99994778633118 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (1, 1)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  13645
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2482, Train Acc: 0.5199
#     Val Loss: 1.3655, Val Acc: 0.4267

# Epoch 200 / 1000:
#     Train Loss: 1.1300, Train Acc: 0.5891
#     Val Loss: 1.3252, Val Acc: 0.4531

# Epoch 300 / 1000:
#     Train Loss: 1.0438, Train Acc: 0.6432
#     Val Loss: 1.3010, Val Acc: 0.4654

# Epoch 400 / 1000:
#     Train Loss: 0.9760, Train Acc: 0.6794
#     Val Loss: 1.2849, Val Acc: 0.4660

# Epoch 500 / 1000:
#     Train Loss: 0.9187, Train Acc: 0.7092
#     Val Loss: 1.2737, Val Acc: 0.4678

# Epoch 600 / 1000:
#     Train Loss: 0.8703, Train Acc: 0.7381
#     Val Loss: 1.2659, Val Acc: 0.4683

# Epoch 700 / 1000:
#     Train Loss: 0.8279, Train Acc: 0.7615
#     Val Loss: 1.2603, Val Acc: 0.4648

# Epoch 800 / 1000:
#     Train Loss: 0.7903, Train Acc: 0.7778
#     Val Loss: 1.2565, Val Acc: 0.4654

# Epoch 900 / 1000:
#     Train Loss: 0.7567, Train Acc: 0.7967
#     Val Loss: 1.2538, Val Acc: 0.4672

# Epoch 1000 / 1000:
#     Train Loss: 0.7268, Train Acc: 0.8117
#     Val Loss: 1.2522, Val Acc: 0.4707


# Training completed!
# Final Test Loss: 1.2560
# Final Test Accuracy: 0.4784
# Time Cost: 133.9961724281311 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (2, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4405, Train Acc: 0.4055
#     Val Loss: 1.4734, Val Acc: 0.3646

# Epoch 200 / 1000:
#     Train Loss: 1.4011, Train Acc: 0.4289
#     Val Loss: 1.4584, Val Acc: 0.3728

# Epoch 300 / 1000:
#     Train Loss: 1.3700, Train Acc: 0.4426
#     Val Loss: 1.4490, Val Acc: 0.3798

# Epoch 400 / 1000:
#     Train Loss: 1.3459, Train Acc: 0.4541
#     Val Loss: 1.4430, Val Acc: 0.3851

# Epoch 500 / 1000:
#     Train Loss: 1.3250, Train Acc: 0.4664
#     Val Loss: 1.4390, Val Acc: 0.3798

# Epoch 600 / 1000:
#     Train Loss: 1.3065, Train Acc: 0.4766
#     Val Loss: 1.4366, Val Acc: 0.3816

# Epoch 700 / 1000:
#     Train Loss: 1.2917, Train Acc: 0.4803
#     Val Loss: 1.4353, Val Acc: 0.3792

# Early stopping triggered at epoch 763

# Training completed!
# Final Test Loss: 1.4321
# Final Test Accuracy: 0.3838
# Time Cost: 97.53811264038086 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (2, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4079, Train Acc: 0.4528
#     Val Loss: 1.4669, Val Acc: 0.3775

# Epoch 200 / 1000:
#     Train Loss: 1.3400, Train Acc: 0.5065
#     Val Loss: 1.4468, Val Acc: 0.3957

# Epoch 300 / 1000:
#     Train Loss: 1.2847, Train Acc: 0.5404
#     Val Loss: 1.4333, Val Acc: 0.3986

# Epoch 400 / 1000:
#     Train Loss: 1.2370, Train Acc: 0.5679
#     Val Loss: 1.4237, Val Acc: 0.3998

# Epoch 500 / 1000:
#     Train Loss: 1.1961, Train Acc: 0.5887
#     Val Loss: 1.4169, Val Acc: 0.3974

# Epoch 600 / 1000:
#     Train Loss: 1.1599, Train Acc: 0.6056
#     Val Loss: 1.4119, Val Acc: 0.3921

# Epoch 700 / 1000:
#     Train Loss: 1.1268, Train Acc: 0.6242
#     Val Loss: 1.4084, Val Acc: 0.3962

# Epoch 800 / 1000:
#     Train Loss: 1.0976, Train Acc: 0.6401
#     Val Loss: 1.4061, Val Acc: 0.4004

# Epoch 900 / 1000:
#     Train Loss: 1.0708, Train Acc: 0.6498
#     Val Loss: 1.4047, Val Acc: 0.4027

# Epoch 1000 / 1000:
#     Train Loss: 1.0453, Train Acc: 0.6629
#     Val Loss: 1.4039, Val Acc: 0.4033


# Training completed!
# Final Test Loss: 1.4062
# Final Test Accuracy: 0.4143
# Time Cost: 130.27643871307373 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (2, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.3910, Train Acc: 0.4815
#     Val Loss: 1.4656, Val Acc: 0.3845

# Epoch 200 / 1000:
#     Train Loss: 1.3070, Train Acc: 0.5420
#     Val Loss: 1.4445, Val Acc: 0.4004

# Epoch 300 / 1000:
#     Train Loss: 1.2382, Train Acc: 0.5868
#     Val Loss: 1.4300, Val Acc: 0.4015

# Epoch 400 / 1000:
#     Train Loss: 1.1796, Train Acc: 0.6178
#     Val Loss: 1.4197, Val Acc: 0.3998

# Epoch 500 / 1000:
#     Train Loss: 1.1281, Train Acc: 0.6452
#     Val Loss: 1.4121, Val Acc: 0.4009

# Epoch 600 / 1000:
#     Train Loss: 1.0826, Train Acc: 0.6734
#     Val Loss: 1.4064, Val Acc: 0.3968

# Epoch 700 / 1000:
#     Train Loss: 1.0419, Train Acc: 0.6938
#     Val Loss: 1.4023, Val Acc: 0.4009

# Epoch 800 / 1000:
#     Train Loss: 1.0054, Train Acc: 0.7108
#     Val Loss: 1.3993, Val Acc: 0.4033

# Epoch 900 / 1000:
#     Train Loss: 0.9714, Train Acc: 0.7308
#     Val Loss: 1.3972, Val Acc: 0.4062

# Epoch 1000 / 1000:
#     Train Loss: 0.9403, Train Acc: 0.7456
#     Val Loss: 1.3958, Val Acc: 0.4045


# Training completed!
# Final Test Loss: 1.4014
# Final Test Accuracy: 0.4152
# Time Cost: 131.64598536491394 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (2, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  20000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.3730, Train Acc: 0.5074
#     Val Loss: 1.4652, Val Acc: 0.3857

# Epoch 200 / 1000:
#     Train Loss: 1.2747, Train Acc: 0.5771
#     Val Loss: 1.4439, Val Acc: 0.4009

# Epoch 300 / 1000:
#     Train Loss: 1.1934, Train Acc: 0.6282
#     Val Loss: 1.4293, Val Acc: 0.4033

# Epoch 400 / 1000:
#     Train Loss: 1.1240, Train Acc: 0.6707
#     Val Loss: 1.4188, Val Acc: 0.4015

# Epoch 500 / 1000:
#     Train Loss: 1.0632, Train Acc: 0.7000
#     Val Loss: 1.4110, Val Acc: 0.4033

# Epoch 600 / 1000:
#     Train Loss: 1.0093, Train Acc: 0.7333
#     Val Loss: 1.4052, Val Acc: 0.4027

# Epoch 700 / 1000:
#     Train Loss: 0.9618, Train Acc: 0.7587
#     Val Loss: 1.4008, Val Acc: 0.4015

# Epoch 800 / 1000:
#     Train Loss: 0.9195, Train Acc: 0.7814
#     Val Loss: 1.3975, Val Acc: 0.4027

# Epoch 900 / 1000:
#     Train Loss: 0.8804, Train Acc: 0.8001
#     Val Loss: 1.3951, Val Acc: 0.4039

# Epoch 1000 / 1000:
#     Train Loss: 0.8448, Train Acc: 0.8179
#     Val Loss: 1.3934, Val Acc: 0.4050


# Training completed!
# Final Test Loss: 1.3992
# Final Test Accuracy: 0.4146
# Time Cost: 135.57528853416443 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (3, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4879, Train Acc: 0.3680
#     Val Loss: 1.4978, Val Acc: 0.3142

# Epoch 200 / 1000:
#     Train Loss: 1.4749, Train Acc: 0.3801
#     Val Loss: 1.4935, Val Acc: 0.3224

# Epoch 300 / 1000:
#     Train Loss: 1.4639, Train Acc: 0.3801
#     Val Loss: 1.4905, Val Acc: 0.3218

# Epoch 400 / 1000:
#     Train Loss: 1.4536, Train Acc: 0.3837
#     Val Loss: 1.4883, Val Acc: 0.3224

# Epoch 500 / 1000:
#     Train Loss: 1.4446, Train Acc: 0.3863
#     Val Loss: 1.4865, Val Acc: 0.3224

# Epoch 600 / 1000:
#     Train Loss: 1.4363, Train Acc: 0.3891
#     Val Loss: 1.4850, Val Acc: 0.3212

# Epoch 700 / 1000:
#     Train Loss: 1.4283, Train Acc: 0.3934
#     Val Loss: 1.4838, Val Acc: 0.3206

# Epoch 800 / 1000:
#     Train Loss: 1.4212, Train Acc: 0.3963
#     Val Loss: 1.4827, Val Acc: 0.3206

# Epoch 900 / 1000:
#     Train Loss: 1.4141, Train Acc: 0.4000
#     Val Loss: 1.4819, Val Acc: 0.3224

# Epoch 1000 / 1000:
#     Train Loss: 1.4082, Train Acc: 0.4019
#     Val Loss: 1.4812, Val Acc: 0.3230


# Training completed!
# Final Test Loss: 1.4796
# Final Test Accuracy: 0.3270
# Time Cost: 129.05377388000488 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (3, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4735, Train Acc: 0.4575
#     Val Loss: 1.4968, Val Acc: 0.3259

# Epoch 200 / 1000:
#     Train Loss: 1.4469, Train Acc: 0.4745
#     Val Loss: 1.4916, Val Acc: 0.3370

# Epoch 300 / 1000:
#     Train Loss: 1.4232, Train Acc: 0.4795
#     Val Loss: 1.4880, Val Acc: 0.3347

# Epoch 400 / 1000:
#     Train Loss: 1.4008, Train Acc: 0.4858
#     Val Loss: 1.4852, Val Acc: 0.3347

# Epoch 500 / 1000:
#     Train Loss: 1.3802, Train Acc: 0.4954
#     Val Loss: 1.4828, Val Acc: 0.3341

# Epoch 600 / 1000:
#     Train Loss: 1.3608, Train Acc: 0.5069
#     Val Loss: 1.4809, Val Acc: 0.3329

# Epoch 700 / 1000:
#     Train Loss: 1.3424, Train Acc: 0.5148
#     Val Loss: 1.4793, Val Acc: 0.3312

# Epoch 800 / 1000:
#     Train Loss: 1.3249, Train Acc: 0.5256
#     Val Loss: 1.4779, Val Acc: 0.3318

# Epoch 900 / 1000:
#     Train Loss: 1.3088, Train Acc: 0.5331
#     Val Loss: 1.4767, Val Acc: 0.3341

# Epoch 1000 / 1000:
#     Train Loss: 1.2932, Train Acc: 0.5390
#     Val Loss: 1.4757, Val Acc: 0.3318


# Training completed!
# Final Test Loss: 1.4757
# Final Test Accuracy: 0.3358
# Time Cost: 130.57136487960815 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (3, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4653, Train Acc: 0.4769
#     Val Loss: 1.4967, Val Acc: 0.3247

# Epoch 200 / 1000:
#     Train Loss: 1.4306, Train Acc: 0.5204
#     Val Loss: 1.4915, Val Acc: 0.3359

# Epoch 300 / 1000:
#     Train Loss: 1.3992, Train Acc: 0.5285
#     Val Loss: 1.4878, Val Acc: 0.3341

# Epoch 400 / 1000:
#     Train Loss: 1.3700, Train Acc: 0.5384
#     Val Loss: 1.4849, Val Acc: 0.3335

# Epoch 500 / 1000:
#     Train Loss: 1.3421, Train Acc: 0.5524
#     Val Loss: 1.4826, Val Acc: 0.3329

# Epoch 600 / 1000:
#     Train Loss: 1.3161, Train Acc: 0.5663
#     Val Loss: 1.4806, Val Acc: 0.3324

# Epoch 700 / 1000:
#     Train Loss: 1.2914, Train Acc: 0.5792
#     Val Loss: 1.4789, Val Acc: 0.3306

# Epoch 800 / 1000:
#     Train Loss: 1.2681, Train Acc: 0.5922
#     Val Loss: 1.4774, Val Acc: 0.3306

# Epoch 900 / 1000:
#     Train Loss: 1.2465, Train Acc: 0.6041
#     Val Loss: 1.4762, Val Acc: 0.3312

# Epoch 1000 / 1000:
#     Train Loss: 1.2251, Train Acc: 0.6149
#     Val Loss: 1.4752, Val Acc: 0.3312


# Training completed!
# Final Test Loss: 1.4754
# Final Test Accuracy: 0.3382
# Time Cost: 132.28557586669922 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (3, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  20000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4489, Train Acc: 0.5276
#     Val Loss: 1.4966, Val Acc: 0.3236

# Epoch 200 / 1000:
#     Train Loss: 1.3982, Train Acc: 0.5619
#     Val Loss: 1.4915, Val Acc: 0.3376

# Epoch 300 / 1000:
#     Train Loss: 1.3525, Train Acc: 0.5797
#     Val Loss: 1.4877, Val Acc: 0.3359

# Epoch 400 / 1000:
#     Train Loss: 1.3099, Train Acc: 0.6007
#     Val Loss: 1.4848, Val Acc: 0.3353

# Epoch 500 / 1000:
#     Train Loss: 1.2698, Train Acc: 0.6279
#     Val Loss: 1.4825, Val Acc: 0.3353

# Epoch 600 / 1000:
#     Train Loss: 1.2327, Train Acc: 0.6513
#     Val Loss: 1.4805, Val Acc: 0.3347

# Epoch 700 / 1000:
#     Train Loss: 1.1979, Train Acc: 0.6724
#     Val Loss: 1.4788, Val Acc: 0.3324

# Epoch 800 / 1000:
#     Train Loss: 1.1649, Train Acc: 0.6890
#     Val Loss: 1.4774, Val Acc: 0.3324

# Epoch 900 / 1000:
#     Train Loss: 1.1333, Train Acc: 0.7105
#     Val Loss: 1.4761, Val Acc: 0.3324

# Epoch 1000 / 1000:
#     Train Loss: 1.1038, Train Acc: 0.7231
#     Val Loss: 1.4751, Val Acc: 0.3324


# Training completed!
# Final Test Loss: 1.4745
# Final Test Accuracy: 0.3409
# Time Cost: 135.0682303905487 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (1, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2955, Train Acc: 0.4762
#     Val Loss: 1.3759, Val Acc: 0.4132

# Epoch 200 / 1000:
#     Train Loss: 1.2236, Train Acc: 0.5166
#     Val Loss: 1.3478, Val Acc: 0.4367

# Epoch 300 / 1000:
#     Train Loss: 1.1775, Train Acc: 0.5394
#     Val Loss: 1.3340, Val Acc: 0.4402

# Epoch 400 / 1000:
#     Train Loss: 1.1434, Train Acc: 0.5535
#     Val Loss: 1.3271, Val Acc: 0.4431

# Epoch 500 / 1000:
#     Train Loss: 1.1178, Train Acc: 0.5658
#     Val Loss: 1.3239, Val Acc: 0.4420

# Early stopping triggered at epoch 553

# Training completed!
# Final Test Loss: 1.3153
# Final Test Accuracy: 0.4421
# Time Cost: 70.71546840667725 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (1, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2312, Train Acc: 0.5289
#     Val Loss: 1.3555, Val Acc: 0.4332

# Epoch 200 / 1000:
#     Train Loss: 1.1134, Train Acc: 0.5956
#     Val Loss: 1.3151, Val Acc: 0.4478

# Epoch 300 / 1000:
#     Train Loss: 1.0312, Train Acc: 0.6413
#     Val Loss: 1.2926, Val Acc: 0.4666

# Epoch 400 / 1000:
#     Train Loss: 0.9664, Train Acc: 0.6746
#     Val Loss: 1.2790, Val Acc: 0.4637

# Epoch 500 / 1000:
#     Train Loss: 0.9138, Train Acc: 0.6995
#     Val Loss: 1.2706, Val Acc: 0.4689

# Epoch 600 / 1000:
#     Train Loss: 0.8691, Train Acc: 0.7253
#     Val Loss: 1.2655, Val Acc: 0.4760

# Epoch 700 / 1000:
#     Train Loss: 0.8311, Train Acc: 0.7412
#     Val Loss: 1.2627, Val Acc: 0.4783

# Early stopping triggered at epoch 792

# Training completed!
# Final Test Loss: 1.2730
# Final Test Accuracy: 0.4772
# Time Cost: 102.21839427947998 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (1, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2047, Train Acc: 0.5520
#     Val Loss: 1.3517, Val Acc: 0.4355

# Epoch 200 / 1000:
#     Train Loss: 1.0683, Train Acc: 0.6357
#     Val Loss: 1.3092, Val Acc: 0.4566

# Epoch 300 / 1000:
#     Train Loss: 0.9714, Train Acc: 0.6898
#     Val Loss: 1.2854, Val Acc: 0.4701

# Epoch 400 / 1000:
#     Train Loss: 0.8959, Train Acc: 0.7280
#     Val Loss: 1.2708, Val Acc: 0.4701

# Epoch 500 / 1000:
#     Train Loss: 0.8334, Train Acc: 0.7577
#     Val Loss: 1.2618, Val Acc: 0.4713

# Epoch 600 / 1000:
#     Train Loss: 0.7814, Train Acc: 0.7860
#     Val Loss: 1.2560, Val Acc: 0.4766

# Epoch 700 / 1000:
#     Train Loss: 0.7368, Train Acc: 0.8071
#     Val Loss: 1.2528, Val Acc: 0.4789

# Epoch 800 / 1000:
#     Train Loss: 0.6979, Train Acc: 0.8253
#     Val Loss: 1.2513, Val Acc: 0.4807

# Early stopping triggered at epoch 814

# Training completed!
# Final Test Loss: 1.2628
# Final Test Accuracy: 0.4817
# Time Cost: 106.96104764938354 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (1, 2)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  20000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.1791, Train Acc: 0.5756
#     Val Loss: 1.3504, Val Acc: 0.4361

# Epoch 200 / 1000:
#     Train Loss: 1.0252, Train Acc: 0.6721
#     Val Loss: 1.3071, Val Acc: 0.4549

# Epoch 300 / 1000:
#     Train Loss: 0.9148, Train Acc: 0.7321
#     Val Loss: 1.2828, Val Acc: 0.4713

# Epoch 400 / 1000:
#     Train Loss: 0.8299, Train Acc: 0.7780
#     Val Loss: 1.2678, Val Acc: 0.4736

# Epoch 500 / 1000:
#     Train Loss: 0.7606, Train Acc: 0.8123
#     Val Loss: 1.2582, Val Acc: 0.4771

# Epoch 600 / 1000:
#     Train Loss: 0.7029, Train Acc: 0.8381
#     Val Loss: 1.2521, Val Acc: 0.4795

# Epoch 700 / 1000:
#     Train Loss: 0.6541, Train Acc: 0.8609
#     Val Loss: 1.2484, Val Acc: 0.4801

# Epoch 800 / 1000:
#     Train Loss: 0.6125, Train Acc: 0.8783
#     Val Loss: 1.2464, Val Acc: 0.4801

# Early stopping triggered at epoch 882

# Training completed!
# Final Test Loss: 1.2593
# Final Test Accuracy: 0.4811
# Time Cost: 119.24528694152832 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (1, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2948, Train Acc: 0.4756
#     Val Loss: 1.3757, Val Acc: 0.4138

# Epoch 200 / 1000:
#     Train Loss: 1.2231, Train Acc: 0.5172
#     Val Loss: 1.3484, Val Acc: 0.4367

# Epoch 300 / 1000:
#     Train Loss: 1.1766, Train Acc: 0.5388
#     Val Loss: 1.3353, Val Acc: 0.4379

# Epoch 400 / 1000:
#     Train Loss: 1.1435, Train Acc: 0.5518
#     Val Loss: 1.3290, Val Acc: 0.4390

# Epoch 500 / 1000:
#     Train Loss: 1.1176, Train Acc: 0.5619
#     Val Loss: 1.3263, Val Acc: 0.4373

# Early stopping triggered at epoch 547

# Training completed!
# Final Test Loss: 1.3129
# Final Test Accuracy: 0.4400
# Time Cost: 69.81087303161621 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (1, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2300, Train Acc: 0.5316
#     Val Loss: 1.3549, Val Acc: 0.4343

# Epoch 200 / 1000:
#     Train Loss: 1.1119, Train Acc: 0.5959
#     Val Loss: 1.3152, Val Acc: 0.4496

# Epoch 300 / 1000:
#     Train Loss: 1.0293, Train Acc: 0.6402
#     Val Loss: 1.2933, Val Acc: 0.4648

# Epoch 400 / 1000:
#     Train Loss: 0.9655, Train Acc: 0.6754
#     Val Loss: 1.2803, Val Acc: 0.4648

# Epoch 500 / 1000:
#     Train Loss: 0.9138, Train Acc: 0.7018
#     Val Loss: 1.2724, Val Acc: 0.4683

# Epoch 600 / 1000:
#     Train Loss: 0.8697, Train Acc: 0.7235
#     Val Loss: 1.2677, Val Acc: 0.4742

# Epoch 700 / 1000:
#     Train Loss: 0.8319, Train Acc: 0.7399
#     Val Loss: 1.2654, Val Acc: 0.4783

# Early stopping triggered at epoch 738

# Training completed!
# Final Test Loss: 1.2731
# Final Test Accuracy: 0.4723
# Time Cost: 95.6448495388031 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (1, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2013, Train Acc: 0.5525
#     Val Loss: 1.3505, Val Acc: 0.4338

# Epoch 200 / 1000:
#     Train Loss: 1.0646, Train Acc: 0.6366
#     Val Loss: 1.3085, Val Acc: 0.4560

# Epoch 300 / 1000:
#     Train Loss: 0.9673, Train Acc: 0.6879
#     Val Loss: 1.2852, Val Acc: 0.4695

# Epoch 400 / 1000:
#     Train Loss: 0.8924, Train Acc: 0.7283
#     Val Loss: 1.2712, Val Acc: 0.4601

# Epoch 500 / 1000:
#     Train Loss: 0.8301, Train Acc: 0.7593
#     Val Loss: 1.2626, Val Acc: 0.4654

# Epoch 600 / 1000:
#     Train Loss: 0.7784, Train Acc: 0.7841
#     Val Loss: 1.2573, Val Acc: 0.4713

# Epoch 700 / 1000:
#     Train Loss: 0.7342, Train Acc: 0.8073
#     Val Loss: 1.2546, Val Acc: 0.4736

# Early stopping triggered at epoch 779

# Training completed!
# Final Test Loss: 1.2630
# Final Test Accuracy: 0.4811
# Time Cost: 103.31167507171631 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (1, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  20000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.1706, Train Acc: 0.5787
#     Val Loss: 1.3487, Val Acc: 0.4355

# Epoch 200 / 1000:
#     Train Loss: 1.0135, Train Acc: 0.6773
#     Val Loss: 1.3060, Val Acc: 0.4572

# Epoch 300 / 1000:
#     Train Loss: 0.9025, Train Acc: 0.7386
#     Val Loss: 1.2824, Val Acc: 0.4672

# Epoch 400 / 1000:
#     Train Loss: 0.8174, Train Acc: 0.7848
#     Val Loss: 1.2681, Val Acc: 0.4736

# Epoch 500 / 1000:
#     Train Loss: 0.7480, Train Acc: 0.8158
#     Val Loss: 1.2593, Val Acc: 0.4725

# Epoch 600 / 1000:
#     Train Loss: 0.6907, Train Acc: 0.8442
#     Val Loss: 1.2539, Val Acc: 0.4701

# Epoch 700 / 1000:
#     Train Loss: 0.6424, Train Acc: 0.8673
#     Val Loss: 1.2509, Val Acc: 0.4725

# Early stopping triggered at epoch 783

# Training completed!
# Final Test Loss: 1.2596
# Final Test Accuracy: 0.4814
# Time Cost: 106.30422949790955 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 1000   n: (2, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  1000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4391, Train Acc: 0.4020
#     Val Loss: 1.4724, Val Acc: 0.3634

# Epoch 200 / 1000:
#     Train Loss: 1.3995, Train Acc: 0.4254
#     Val Loss: 1.4581, Val Acc: 0.3693

# Epoch 300 / 1000:
#     Train Loss: 1.3699, Train Acc: 0.4400
#     Val Loss: 1.4495, Val Acc: 0.3705

# Epoch 400 / 1000:
#     Train Loss: 1.3459, Train Acc: 0.4514
#     Val Loss: 1.4441, Val Acc: 0.3728

# Epoch 500 / 1000:
#     Train Loss: 1.3257, Train Acc: 0.4637
#     Val Loss: 1.4408, Val Acc: 0.3710

# Epoch 600 / 1000:
#     Train Loss: 1.3080, Train Acc: 0.4727
#     Val Loss: 1.4389, Val Acc: 0.3746

# Epoch 700 / 1000:
#     Train Loss: 1.2932, Train Acc: 0.4798
#     Val Loss: 1.4381, Val Acc: 0.3722

# Early stopping triggered at epoch 719

# Training completed!
# Final Test Loss: 1.4347
# Final Test Accuracy: 0.3793
# Time Cost: 92.70239210128784 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 5000   n: (2, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  5000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4032, Train Acc: 0.4509
#     Val Loss: 1.4649, Val Acc: 0.3781

# Epoch 200 / 1000:
#     Train Loss: 1.3343, Train Acc: 0.5013
#     Val Loss: 1.4453, Val Acc: 0.3910

# Epoch 300 / 1000:
#     Train Loss: 1.2785, Train Acc: 0.5375
#     Val Loss: 1.4326, Val Acc: 0.3933

# Epoch 400 / 1000:
#     Train Loss: 1.2317, Train Acc: 0.5619
#     Val Loss: 1.4239, Val Acc: 0.4009

# Epoch 500 / 1000:
#     Train Loss: 1.1920, Train Acc: 0.5841
#     Val Loss: 1.4180, Val Acc: 0.4015

# Epoch 600 / 1000:
#     Train Loss: 1.1561, Train Acc: 0.6003
#     Val Loss: 1.4140, Val Acc: 0.3957

# Epoch 700 / 1000:
#     Train Loss: 1.1253, Train Acc: 0.6164
#     Val Loss: 1.4115, Val Acc: 0.3980

# Epoch 800 / 1000:
#     Train Loss: 1.0966, Train Acc: 0.6296
#     Val Loss: 1.4100, Val Acc: 0.4033

# Early stopping triggered at epoch 877

# Training completed!
# Final Test Loss: 1.4102
# Final Test Accuracy: 0.4056
# Time Cost: 115.69732403755188 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 10000   n: (2, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  10000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.3828, Train Acc: 0.4797
#     Val Loss: 1.4631, Val Acc: 0.3851

# Epoch 200 / 1000:
#     Train Loss: 1.2964, Train Acc: 0.5419
#     Val Loss: 1.4422, Val Acc: 0.4009

# Epoch 300 / 1000:
#     Train Loss: 1.2264, Train Acc: 0.5890
#     Val Loss: 1.4286, Val Acc: 0.3992

# Epoch 400 / 1000:
#     Train Loss: 1.1681, Train Acc: 0.6194
#     Val Loss: 1.4192, Val Acc: 0.4021

# Epoch 500 / 1000:
#     Train Loss: 1.1172, Train Acc: 0.6458
#     Val Loss: 1.4127, Val Acc: 0.4009

# Epoch 600 / 1000:
#     Train Loss: 1.0727, Train Acc: 0.6681
#     Val Loss: 1.4082, Val Acc: 0.3968

# Epoch 700 / 1000:
#     Train Loss: 1.0329, Train Acc: 0.6895
#     Val Loss: 1.4052, Val Acc: 0.4004

# Epoch 800 / 1000:
#     Train Loss: 0.9970, Train Acc: 0.7068
#     Val Loss: 1.4033, Val Acc: 0.4056

# Epoch 900 / 1000:
#     Train Loss: 0.9645, Train Acc: 0.7218
#     Val Loss: 1.4023, Val Acc: 0.4050

# Early stopping triggered at epoch 967

# Training completed!
# Final Test Loss: 1.4043
# Final Test Accuracy: 0.4086
# Time Cost: 128.3252830505371 s

# ==================================================
# Experiment: Changing parameter max_feature & n
# max_feature: 20000   n: (2, 3)
# ==================================================
# ==================================================
# 0. Data Preprocessing
# ==================================================
# Train Data Shape: (8528, 2)
# Test Data Shape: (3309, 2)
# After Split - Train Data Shape: (6822, 2)
# After Split - Validation Data Shape: (1706, 2)
# Feature dimension:  20000
# Data ready for training.

# ==================================================
# 1. Raw Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.3579, Train Acc: 0.5143
#     Val Loss: 1.4619, Val Acc: 0.3880

# Epoch 200 / 1000:
#     Train Loss: 1.2526, Train Acc: 0.5877
#     Val Loss: 1.4403, Val Acc: 0.4009

# Epoch 300 / 1000:
#     Train Loss: 1.1666, Train Acc: 0.6402
#     Val Loss: 1.4260, Val Acc: 0.4004

# Epoch 400 / 1000:
#     Train Loss: 1.0945, Train Acc: 0.6833
#     Val Loss: 1.4161, Val Acc: 0.4021

# Epoch 500 / 1000:
#     Train Loss: 1.0329, Train Acc: 0.7129
#     Val Loss: 1.4090, Val Acc: 0.3998

# Epoch 600 / 1000:
#     Train Loss: 0.9795, Train Acc: 0.7403
#     Val Loss: 1.4040, Val Acc: 0.3957

# Epoch 700 / 1000:
#     Train Loss: 0.9327, Train Acc: 0.7622
#     Val Loss: 1.4005, Val Acc: 0.4015

# Epoch 800 / 1000:
#     Train Loss: 0.8901, Train Acc: 0.7841
#     Val Loss: 1.3980, Val Acc: 0.4050

# Epoch 900 / 1000:
#     Train Loss: 0.8523, Train Acc: 0.8015
#     Val Loss: 1.3965, Val Acc: 0.4062

# Epoch 1000 / 1000:
#     Train Loss: 0.8184, Train Acc: 0.8157
#     Val Loss: 1.3956, Val Acc: 0.4039


# Training completed!
# Final Test Loss: 1.4013
# Final Test Accuracy: 0.4068
# Time Cost: 135.87224411964417 s