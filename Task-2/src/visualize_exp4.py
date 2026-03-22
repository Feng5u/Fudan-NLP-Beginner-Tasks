import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    ['CNN', 0.62716],
    ['RNN', 0.62422],
    ['Transformer', 0.61023]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['model', 'accuracy'])

# Sort by accuracy for better visualization
df = df.sort_values('accuracy', ascending=False)

# Create bar chart
plt.figure(figsize=(9, 6))
colors = ['#66b3ff', '#ff9999', '#99ff99']
bars = plt.bar(df['model'], df['accuracy'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels
for bar, acc in zip(bars, df['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{acc:.5f}', ha='center', fontsize=12, fontweight='bold')

# Customize chart
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Model Architecture Comparison', fontsize=14, fontweight='bold')
plt.ylim(0.60, 0.635)
plt.grid(True, alpha=0.3, axis='y')

# Add baseline and annotations
best_acc = df.iloc[0]['accuracy']
plt.axhline(y=best_acc, color='green', linestyle='--', alpha=0.5, linewidth=1)

# Add comparison annotations
cnn_rnn_diff = (0.62716 - 0.62422) * 10000
cnn_trans_diff = (0.62716 - 0.61023) * 10000


plt.tight_layout()
plt.show()

# Print analysis
print("="*50)
print("MODEL ARCHITECTURE COMPARISON")
print("="*50)
print(f"\nCNN:         {df[df['model']=='CNN']['accuracy'].values[0]:.5f}")
print(f"RNN:         {df[df['model']=='RNN']['accuracy'].values[0]:.5f}")
print(f"Transformer: {df[df['model']=='Transformer']['accuracy'].values[0]:.5f}")

print(f"\nRanking:")
for i, row in df.iterrows():
    print(f"  {i+1}. {row['model']}: {row['accuracy']:.5f}")

print(f"\nDifferences:")
print(f"  CNN - RNN:         {0.62716 - 0.62422:.5f} ({(0.62716 - 0.62422)*10000:.1f}*10⁻⁴)")
print(f"  CNN - Transformer: {0.62716 - 0.61023:.5f} ({(0.62716 - 0.61023)*10000:.1f}*10⁻⁴)")
print(f"  RNN - Transformer: {0.62422 - 0.61023:.5f} ({(0.62422 - 0.61023)*10000:.1f}*10⁻⁴)")

print(f"\nKey Finding: CNN performs best, followed closely by RNN, with Transformer lagging behind")