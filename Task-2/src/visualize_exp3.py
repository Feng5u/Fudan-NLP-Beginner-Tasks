import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = [
    ['with GloVe', 0.64128],
    ['without GloVe', 0.62716]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['condition', 'accuracy'])

# Create bar chart
plt.figure(figsize=(8, 6))
colors = ['#66b3ff', '#ff9999']
bars = plt.bar(df['condition'], df['accuracy'], color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, acc in zip(bars, df['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{acc:.5f}', ha='center', fontsize=12, fontweight='bold')

# Customize chart
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Effect of GloVe Pretrained Vectors', fontsize=14, fontweight='bold')
plt.ylim(0.62, 0.65)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print analysis
print("="*50)
print("GLOVE PRETRAINED VECTORS ANALYSIS")
print("="*50)
print(f"\nConfiguration: kernel=[3,4,5], filters=150, optimizer=adam")
print(f"\nWith GloVe:    {df.iloc[0]['accuracy']:.5f}")
print(f"Without GloVe: {df.iloc[1]['accuracy']:.5f}")
print(f"Absolute gain: {df.iloc[0]['accuracy'] - df.iloc[1]['accuracy']:.5f}")