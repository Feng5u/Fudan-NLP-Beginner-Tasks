import random

def generate_addition_data(filename, num_samples_per_combo=1000):
    """
    生成多位数加法数据集
    
    参数:
    filename: 输出文件名
    num_samples_per_combo: 每种位数组合的样本数量
    """
    
    digit_combinations = [
        (1, 1), (1, 2), (2, 1), (2, 2),
        (2, 3), (3, 2), (3, 3),
        (3, 4), (4, 3), (4, 4),
        (4, 5), (5, 4), (5, 5)
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for digits1, digits2 in digit_combinations:
            for _ in range(num_samples_per_combo):
                num1 = random.randint(10**(digits1-1), 10**digits1 - 1) if digits1 > 1 else random.randint(0, 9)
                num2 = random.randint(10**(digits2-1), 10**digits2 - 1) if digits2 > 1 else random.randint(0, 9)
                
                result = num1 + num2
                
                f.write(f"{num1}+{num2}={result}\n")
    
    print(f"数据集已生成到 {filename}")
    print(f"共生成 {len(digit_combinations) * num_samples_per_combo} 个样本")

if __name__ == "__main__":
    generate_addition_data("subtask-1.txt", num_samples_per_combo=7700)
    
    print("\n前10条数据示例: ")
    with open("subtask-1.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:
                print(line.strip())
            else:
                break