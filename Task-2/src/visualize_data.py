import pandas as pd
import matplotlib.pyplot as plt

def plot_tsv_distribution(file_path, label_column='Sentiment'):
    df = pd.read_csv(file_path, delimiter='\t')
    label_counts = df[label_column].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(label_counts)), label_counts.values, color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(len(label_counts)), label_counts.index)
    
    for i, v in enumerate(label_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.show()
    
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {len(label_counts)}")
    for label, count in label_counts.items():
        print(f"Class {label}: {count}")

plot_tsv_distribution('/home/feng5u/桌面/Notes/2025-2026学年_寒假/Fudan NLP/data/raw/train.tsv', 'Sentiment')