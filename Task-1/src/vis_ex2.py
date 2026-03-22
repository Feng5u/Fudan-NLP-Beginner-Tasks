import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Organize experimental data
learning_rates = [0.001, 0.01, 0.1, 0.5]
loss_functions = ['cross_entropy', 'mse', 'hinge', 'perceptron']

# Test accuracy data
test_accuracies = {
    'cross_entropy': [0.4524, 0.4805, 0.4796, 0.4817],
    'mse': [0.4403, 0.4763, 0.4711, 0.4717],
    'hinge': [0.4615, 0.4663, 0.4636, 0.4606],
    'perceptron': [0.3953, 0.4294, 0.4427, 0.3741]
}

# Convergence epochs data (extracted from experiment logs)
convergence_epochs = {
    'cross_entropy': [1000, 809, 85, 23],  # 1000 means no early stopping
    'mse': [1000, 1000, 126, 32],          # 1000 means no early stopping
    'hinge': [11, 11, 10, 12],
    'perceptron': [9, 18, 17, 6]
}

# Training time data (in seconds)
training_times = {
    'cross_entropy': [146.24, 116.91, 12.18, 3.27],
    'mse': [143.15, 142.91, 17.93, 4.60],
    'hinge': [2.00, 1.90, 1.71, 2.07],
    'perceptron': [1.50, 3.11, 2.88, 1.04]
}

# Create figure
fig = plt.figure(figsize=(18, 12))

# 1. Heatmap - Test Accuracy
plt.subplot(2, 2, 1)
accuracy_matrix = np.array([test_accuracies[loss] for loss in loss_functions])

# Create heatmap
im = plt.imshow(accuracy_matrix, cmap='YlGnBu', aspect='auto', vmin=0.35, vmax=0.5)

# Add value labels
for i in range(len(loss_functions)):
    for j in range(len(learning_rates)):
        plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}', 
                 ha='center', va='center', color='black', fontsize=10)

plt.colorbar(im, label='Test Accuracy')
plt.title('Test Accuracy Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Loss Function', fontsize=12)
plt.xticks(range(len(learning_rates)), learning_rates)
plt.yticks(range(len(loss_functions)), loss_functions)

# 2. Bubble Chart - Convergence Epochs
plt.subplot(2, 2, 2)

# Prepare bubble chart data
x_positions = []
y_positions = []
bubble_sizes = []
colors = []

for i, loss_func in enumerate(loss_functions):
    for j, lr in enumerate(learning_rates):
        x_positions.append(j)
        y_positions.append(i)
        
        # Bubble size based on convergence epochs (log scale for better visualization)
        epochs = convergence_epochs[loss_func][j]
        if epochs == 1000:  # Special handling for non-convergence
            bubble_size = 3000  # Fixed large value
        else:
            bubble_size = epochs * 30  # Scaling factor
        
        bubble_sizes.append(bubble_size)
        
        # Color based on accuracy
        accuracy = test_accuracies[loss_func][j]
        colors.append(accuracy)

# Plot bubble chart
scatter = plt.scatter(x_positions, y_positions, 
                     s=bubble_sizes, 
                     c=colors, 
                     cmap='YlGnBu', 
                     alpha=0.7, 
                     edgecolors='black')

# Add epoch labels
for i, loss_func in enumerate(loss_functions):
    for j, lr in enumerate(learning_rates):
        epochs = convergence_epochs[loss_func][j]
        if epochs == 1000:
            label = '>1000'
        else:
            label = str(epochs)
        
        plt.text(j, i, label, 
                ha='center', va='center', 
                fontsize=9, fontweight='bold')

plt.colorbar(scatter, label='Test Accuracy')
plt.title('Convergence Epochs Bubble Chart\n(Bubble size = convergence epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Loss Function', fontsize=12)
plt.xticks(range(len(learning_rates)), learning_rates)
plt.yticks(range(len(loss_functions)), loss_functions)
plt.grid(True, alpha=0.3)

# 3. Bar Chart - Best Accuracy Comparison
plt.subplot(2, 2, 3)

# Find best accuracy for each loss function and corresponding learning rate
best_accuracies = []
best_lrs = []
loss_func_names = []

for loss_func in loss_functions:
    accuracies = test_accuracies[loss_func]
    best_idx = np.argmax(accuracies)
    best_accuracies.append(accuracies[best_idx])
    best_lrs.append(learning_rates[best_idx])
    loss_func_names.append(loss_func)

x = np.arange(len(loss_functions))
bars = plt.bar(x, best_accuracies, color=plt.cm.Set2(range(len(loss_functions))))

# Add value labels
for i, (bar, acc, lr) in enumerate(zip(bars, best_accuracies, best_lrs)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.3f}\n(lr={lr})', 
             ha='center', va='bottom', fontsize=10)

plt.title('Best Test Accuracy for Each Loss Function', fontsize=14, fontweight='bold')
plt.xlabel('Loss Function', fontsize=12)
plt.ylabel('Best Test Accuracy', fontsize=12)
plt.xticks(x, loss_func_names, rotation=45)
plt.ylim(0.35, 0.52)
plt.grid(True, alpha=0.3, axis='y')

# 4. Line Chart - Training Time Comparison
plt.subplot(2, 2, 4)

for loss_func in loss_functions:
    times = training_times[loss_func]
    plt.plot(learning_rates, times, marker='o', linewidth=2, markersize=8, 
            label=loss_func, alpha=0.8)

plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Learning Rate', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.xticks(learning_rates, learning_rates)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.show()

# Results
# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.001    loss function type: cross_entropy
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
# 1. Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.4382, Train Acc: 0.4043
#     Val Loss: 1.4568, Val Acc: 0.3693

# Epoch 200 / 1000:
#     Train Loss: 1.3912, Train Acc: 0.4366
#     Val Loss: 1.4304, Val Acc: 0.3927

# Epoch 300 / 1000:
#     Train Loss: 1.3570, Train Acc: 0.4604
#     Val Loss: 1.4132, Val Acc: 0.4015

# Epoch 400 / 1000:
#     Train Loss: 1.3282, Train Acc: 0.4766
#     Val Loss: 1.4002, Val Acc: 0.4062

# Epoch 500 / 1000:
#     Train Loss: 1.3025, Train Acc: 0.4913
#     Val Loss: 1.3894, Val Acc: 0.4185

# Epoch 600 / 1000:
#     Train Loss: 1.2795, Train Acc: 0.5060
#     Val Loss: 1.3801, Val Acc: 0.4191

# Epoch 700 / 1000:
#     Train Loss: 1.2580, Train Acc: 0.5187
#     Val Loss: 1.3719, Val Acc: 0.4238

# Epoch 800 / 1000:
#     Train Loss: 1.2397, Train Acc: 0.5298
#     Val Loss: 1.3645, Val Acc: 0.4291

# Epoch 900 / 1000:
#     Train Loss: 1.2204, Train Acc: 0.5413
#     Val Loss: 1.3578, Val Acc: 0.4326

# Epoch 1000 / 1000:
#     Train Loss: 1.2028, Train Acc: 0.5517
#     Val Loss: 1.3517, Val Acc: 0.4343


# Training completed!
# Final Test Loss: 1.3412
# Final Test Accuracy: 0.4524
# Time cost: 146.243572473526 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.01    loss function type: cross_entropy
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
# 1. Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 1.2049, Train Acc: 0.5518
#     Val Loss: 1.3517, Val Acc: 0.4349

# Epoch 200 / 1000:
#     Train Loss: 1.0685, Train Acc: 0.6348
#     Val Loss: 1.3093, Val Acc: 0.4560

# Epoch 300 / 1000:
#     Train Loss: 0.9719, Train Acc: 0.6887
#     Val Loss: 1.2855, Val Acc: 0.4695

# Epoch 400 / 1000:
#     Train Loss: 0.8955, Train Acc: 0.7281
#     Val Loss: 1.2708, Val Acc: 0.4695

# Epoch 500 / 1000:
#     Train Loss: 0.8334, Train Acc: 0.7586
#     Val Loss: 1.2617, Val Acc: 0.4719

# Epoch 600 / 1000:
#     Train Loss: 0.7814, Train Acc: 0.7855
#     Val Loss: 1.2560, Val Acc: 0.4760

# Epoch 700 / 1000:
#     Train Loss: 0.7371, Train Acc: 0.8078
#     Val Loss: 1.2527, Val Acc: 0.4783

# Epoch 800 / 1000:
#     Train Loss: 0.6974, Train Acc: 0.8238
#     Val Loss: 1.2513, Val Acc: 0.4801

# Early stopping triggered at epoch 809

# Training completed!
# Final Test Loss: 1.2627
# Final Test Accuracy: 0.4805
# Time cost: 116.9098572731018 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.1    loss function type: cross_entropy
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 85

# Training completed!
# Final Test Loss: 1.2626
# Final Test Accuracy: 0.4796
# Time cost: 12.176857471466064 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.5    loss function type: cross_entropy
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 23

# Training completed!
# Final Test Loss: 1.2673
# Final Test Accuracy: 0.4817
# Time cost: 3.2740602493286133 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.001    loss function type: mse
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
# 1. Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 0.7500, Train Acc: 0.3851
#     Val Loss: 0.7541, Val Acc: 0.3617

# Epoch 200 / 1000:
#     Train Loss: 0.7349, Train Acc: 0.4056
#     Val Loss: 0.7442, Val Acc: 0.3716

# Epoch 300 / 1000:
#     Train Loss: 0.7232, Train Acc: 0.4272
#     Val Loss: 0.7372, Val Acc: 0.3886

# Epoch 400 / 1000:
#     Train Loss: 0.7132, Train Acc: 0.4423
#     Val Loss: 0.7317, Val Acc: 0.3945

# Epoch 500 / 1000:
#     Train Loss: 0.7046, Train Acc: 0.4541
#     Val Loss: 0.7272, Val Acc: 0.3992

# Epoch 600 / 1000:
#     Train Loss: 0.6966, Train Acc: 0.4643
#     Val Loss: 0.7234, Val Acc: 0.4033

# Epoch 700 / 1000:
#     Train Loss: 0.6891, Train Acc: 0.4756
#     Val Loss: 0.7202, Val Acc: 0.4080

# Epoch 800 / 1000:
#     Train Loss: 0.6821, Train Acc: 0.4855
#     Val Loss: 0.7173, Val Acc: 0.4097

# Epoch 900 / 1000:
#     Train Loss: 0.6756, Train Acc: 0.4962
#     Val Loss: 0.7148, Val Acc: 0.4115

# Epoch 1000 / 1000:
#     Train Loss: 0.6696, Train Acc: 0.5031
#     Val Loss: 0.7125, Val Acc: 0.4179


# Training completed!
# Final Test Loss: 0.7094
# Final Test Accuracy: 0.4403
# Time cost: 143.1469268798828 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.01    loss function type: mse
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
# 1. Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 0.6699, Train Acc: 0.5043
#     Val Loss: 0.7125, Val Acc: 0.4174

# Epoch 200 / 1000:
#     Train Loss: 0.6195, Train Acc: 0.5676
#     Val Loss: 0.6968, Val Acc: 0.4402

# Epoch 300 / 1000:
#     Train Loss: 0.5803, Train Acc: 0.6142
#     Val Loss: 0.6874, Val Acc: 0.4490

# Epoch 400 / 1000:
#     Train Loss: 0.5476, Train Acc: 0.6513
#     Val Loss: 0.6808, Val Acc: 0.4549

# Epoch 500 / 1000:
#     Train Loss: 0.5189, Train Acc: 0.6811
#     Val Loss: 0.6758, Val Acc: 0.4596

# Epoch 600 / 1000:
#     Train Loss: 0.4935, Train Acc: 0.7057
#     Val Loss: 0.6720, Val Acc: 0.4683

# Epoch 700 / 1000:
#     Train Loss: 0.4711, Train Acc: 0.7257
#     Val Loss: 0.6690, Val Acc: 0.4713

# Epoch 800 / 1000:
#     Train Loss: 0.4505, Train Acc: 0.7451
#     Val Loss: 0.6667, Val Acc: 0.4725

# Epoch 900 / 1000:
#     Train Loss: 0.4321, Train Acc: 0.7629
#     Val Loss: 0.6649, Val Acc: 0.4725

# Epoch 1000 / 1000:
#     Train Loss: 0.4147, Train Acc: 0.7783
#     Val Loss: 0.6635, Val Acc: 0.4730


# Training completed!
# Final Test Loss: 0.6660
# Final Test Accuracy: 0.4763
# Time cost: 142.91210532188416 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.1    loss function type: mse
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
# 1. Experiment Start
# ==================================================
# Epoch 100 / 1000:
#     Train Loss: 0.4172, Train Acc: 0.7799
#     Val Loss: 0.6638, Val Acc: 0.4760

# Early stopping triggered at epoch 126

# Training completed!
# Final Test Loss: 0.6657
# Final Test Accuracy: 0.4711
# Time cost: 17.932445287704468 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.5    loss function type: mse
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 32

# Training completed!
# Final Test Loss: 0.6669
# Final Test Accuracy: 0.4717
# Time cost: 4.595304727554321 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.001    loss function type: hinge
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 11

# Training completed!
# Final Test Loss: 1.0000
# Final Test Accuracy: 0.4615
# Time cost: 2.003605604171753 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.01    loss function type: hinge
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 11

# Training completed!
# Final Test Loss: 1.0001
# Final Test Accuracy: 0.4663
# Time cost: 1.8981616497039795 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.1    loss function type: hinge
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 10

# Training completed!
# Final Test Loss: 1.0005
# Final Test Accuracy: 0.4636
# Time cost: 1.709465503692627 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.5    loss function type: hinge
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 12

# Training completed!
# Final Test Loss: 1.0029
# Final Test Accuracy: 0.4606
# Time cost: 2.06514048576355 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.001    loss function type: perceptron
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 9

# Training completed!
# Final Test Loss: 0.0000
# Final Test Accuracy: 0.3953
# Time cost: 1.500333547592163 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.01    loss function type: perceptron
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 18

# Training completed!
# Final Test Loss: 0.0001
# Final Test Accuracy: 0.4294
# Time cost: 3.1110339164733887 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.1    loss function type: perceptron
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 17

# Training completed!
# Final Test Loss: 0.0009
# Final Test Accuracy: 0.4427
# Time cost: 2.8808436393737793 s

# ==================================================
# Experiment: Changing parameter learning_rate & loss function type
# learning_rate: 0.5    loss function type: perceptron
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
# 1. Experiment Start
# ==================================================
# Early stopping triggered at epoch 6

# Training completed!
# Final Test Loss: 0.0061
# Final Test Accuracy: 0.3741
# Time cost: 1.042724609375 s