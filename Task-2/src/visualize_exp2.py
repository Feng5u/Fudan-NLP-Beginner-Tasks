import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    # kernel sizes, num_filters, optimizer, accuracy
    [[2,3,4], 100, 'adam', 0.63155],
    [[3,4,5], 100, 'adam', 0.62692],
    [[4,5,6], 100, 'adam', 0.62767],
    [[5,6,7], 100, 'adam', 0.62693],
    [[3,4,5], 50, 'adam', 0.62778],
    [[3,4,5], 150, 'adam', 0.62716],
    [[3,4,5], 200, 'adam', 0.62639],
    [[3,4,5], 150, 'sgd', 0.59065],
    [[3,4,5], 150, 'rmsprop', 0.62954]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['kernel_sizes', 'num_filters', 'optimizer', 'accuracy'])
df['kernel_sizes_str'] = df['kernel_sizes'].astype(str)

# Chart 1: Effect of kernel sizes (fixed num_filters=100, optimizer=adam)
plt.figure(figsize=(10, 5))
kernel_data = df[(df['num_filters']==100) & (df['optimizer']=='adam')].sort_values('accuracy', ascending=False)
bars = plt.bar(range(len(kernel_data)), kernel_data['accuracy'], color='skyblue', edgecolor='navy')
plt.xlabel('Kernel Sizes')
plt.ylabel('Test Accuracy')
plt.title('Effect of Kernel Sizes (filters=100, optimizer=adam)')
plt.xticks(range(len(kernel_data)), kernel_data['kernel_sizes_str'])
plt.ylim(0.62, 0.635)

for bar, acc in zip(bars, kernel_data['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{acc:.5f}', ha='center', fontsize=9)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Chart 2: Effect of number of filters (fixed kernel=[3,4,5], optimizer=adam) - MODIFIED SIZE
plt.figure(figsize=(10, 5))  # Changed from (10,5) to (14,6)
filter_data = df[(df['kernel_sizes_str']=='[3, 4, 5]') & (df['optimizer']=='adam')].sort_values('num_filters')
plt.plot(filter_data['num_filters'], filter_data['accuracy'], marker='o', linewidth=2, markersize=10, color='coral')
plt.xlabel('Number of Filters')
plt.ylabel('Test Accuracy')
plt.title('Effect of Number of Filters (kernel=[3,4,5], optimizer=adam)')
plt.grid(True, alpha=0.3)

for _, row in filter_data.iterrows():
    plt.text(row['num_filters'], row['accuracy'] + 0.00005, f'{row["accuracy"]:.5f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Chart 3: Effect of optimizers (fixed kernel=[3,4,5], filters=150)
plt.figure(figsize=(8, 5))
opt_data = df[(df['kernel_sizes_str']=='[3, 4, 5]') & (df['num_filters']==150)]
colors = ['#ff9999' if opt == 'sgd' else '#66b3ff' if opt == 'adam' else '#99ff99' for opt in opt_data['optimizer']]
bars = plt.bar(opt_data['optimizer'], opt_data['accuracy'], color=colors, edgecolor='black')
plt.xlabel('Optimizer')
plt.ylabel('Test Accuracy')
plt.title('Effect of Optimizers (kernel=[3,4,5], filters=150)')
plt.ylim(0.58, 0.635)

for bar, acc in zip(bars, opt_data['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{acc:.5f}', ha='center', fontsize=10)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Print analysis
print("="*60)
print("EXPERIMENT ANALYSIS")
print("="*60)

best = df.loc[df['accuracy'].idxmax()]
print(f"\nBest configuration: {best['kernel_sizes']}, filters={best['num_filters']}, {best['optimizer']} - Accuracy: {best['accuracy']:.5f}")

print("\n1. Effect of Kernel Sizes:")
kernel_range = df[(df['num_filters']==100) & (df['optimizer']=='adam')]['accuracy']
print(f"   Best: {kernel_data.iloc[0]['kernel_sizes']} ({kernel_data.iloc[0]['accuracy']:.5f})")
print(f"   Worst: {kernel_data.iloc[-1]['kernel_sizes']} ({kernel_data.iloc[-1]['accuracy']:.5f})")
print(f"   Difference: {(kernel_data.iloc[0]['accuracy'] - kernel_data.iloc[-1]['accuracy'])*10000:.2f}*10^-4")

print("\n2. Effect of Number of Filters:")
filter_acc = filter_data['accuracy'].values
print(f"   Best: {filter_data.iloc[filter_acc.argmax()]['num_filters']} filters ({filter_data.iloc[filter_acc.argmax()]['accuracy']:.5f})")
print(f"   Variation: {filter_data['accuracy'].std():.6f} (very small - filters don't affect much)")

print("\n3. Effect of Optimizers:")
opt_acc = opt_data.set_index('optimizer')['accuracy']
print(f"   RMSprop: {opt_acc['rmsprop']:.5f}")
print(f"   Adam:    {opt_acc['adam']:.5f}")
print(f"   SGD:     {opt_acc['sgd']:.5f}")
print(f"   RMSprop performs best, SGD significantly worse")

print("\n4. Key Findings:")
print("   - Kernel size has minimal impact (variation within 0.005)")
print("   - Number of filters has almost no impact (variation < 0.0015)")
print("   - Optimizer choice matters: RMSprop ≈ Adam >> SGD")
print(f"   - RMSprop improves over Adam by {(opt_acc['rmsprop']-opt_acc['adam'])*10000:.1f}*10^-4")
print(f"   - SGD is {(opt_acc['adam']-opt_acc['sgd'])*10000:.1f}*10^-4 worse than Adam")