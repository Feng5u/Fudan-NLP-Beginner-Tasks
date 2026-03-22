import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = """
cross_entropy 0.0001 0.62622
cross_entropy 0.001 0.63149
cross_entropy 0.01 0.60281
cross_entropy 0.1 0.51775
hinge 0.0001 0.62153
hinge 0.001 0.62822
hinge 0.01 0.59427
hinge 0.1 0.51767
perceptron 0.0001 0.52718
perceptron 0.001 0.51780
perceptron 0.01 0.51792
perceptron 0.1 0.51789
MSE 0.0001 0.60788
MSE 0.001 0.61992
MSE 0.01 0.51709
MSE 0.1 0.51642
"""

lines = data.strip().split('\n')
data_list = [line.split() for line in lines]
df = pd.DataFrame(data_list, columns=['loss', 'lr', 'accuracy'])
df['lr'] = df['lr'].astype(float)
df['accuracy'] = df['accuracy'].astype(float)

# Chart 1: Bar chart for all configurations
plt.figure(figsize=(12, 5))
df_sorted = df.sort_values('accuracy', ascending=False)
x_labels = [f"{row['loss']}\n{row['lr']}" for _, row in df_sorted.iterrows()]
plt.bar(range(len(df_sorted)), df_sorted['accuracy'], color='steelblue')
plt.xlabel('Configuration (loss function / learning rate)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for All Configurations')
plt.xticks(range(len(df_sorted)), x_labels, rotation=45, ha='right')
plt.ylim(0.5, 0.65)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Chart 2: Grouped bar chart by loss function
plt.figure(figsize=(10, 5))
loss_functions = df['loss'].unique()
lr_values = [0.0001, 0.001, 0.01, 0.1]
x = np.arange(len(lr_values))
width = 0.2
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

for i, loss in enumerate(loss_functions):
    accuracies = []
    for lr in lr_values:
        acc = df[(df['loss']==loss) & (df['lr']==lr)]['accuracy'].values[0]
        accuracies.append(acc)
    plt.bar(x + i*width, accuracies, width, label=loss, color=colors[i])

plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy by Loss Function')
plt.xticks(x + width*1.5, lr_values)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("Best configuration:", df.loc[df['accuracy'].idxmax(), ['loss', 'lr', 'accuracy']].to_string(index=False))