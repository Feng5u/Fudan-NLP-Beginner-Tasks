# ================================
# 0. 环境和依赖
# ================================
import sys
import torch
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

TRAIN_PATH = '/kaggle/input/datasets/feng5u/input-data-1/train.tsv.zip'
TEST_PATH = '/kaggle/input/datasets/feng5u/input-data-1/test.tsv.zip'
GLOVE_PATH = '/kaggle/input/datasets/feng5u/input-data/glove.6B.100d.txt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{DEVICE}")

# ================================
# 1. 数据预处理类
# ================================
class DataProcessor:
    """
    数据处理类

    读取数据、构建词汇表、将文本数据转化为序列
    """
    def __init__(self, max_vocab_size=5000, max_len=50, 
                 val_rate = 0.2):
        """
        初始化函数

        参数：
            max_vocab_size: 词汇表最大大小
            max_len: 文本序列最大长度
            val_rate: 验证集比例
        """
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

        self.val_rate = val_rate
    
    def load_data(self, file_path, file_type):
        """
        加载数据

        参数：
            file_path: 数据来源路径
            file_type: 数据类型
        
        返回：
            返回从 file_path 中读取的文本数据及其对应的标签数据
        """
        texts = []
        labels = []
        phrase_ids = []

        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            
            for line in f:
                parts = line.strip().split('\t')
                if file_type == 'train' and len(parts) == 4:
                    phrase_id = parts[0]
                    sentence_id = parts[1]
                    phrase = parts[2].lower()
                    sentiment = int(parts[3])
                    texts.append(phrase)
                    labels.append(sentiment)
                    phrase_ids.append(phrase_id)
                elif file_type == 'test' and len(parts) == 3:
                    phrase_id = parts[0]
                    sentence_id = parts[1]
                    phrase = parts[2].lower()
                    texts.append(phrase)
                    phrase_ids.append(phrase_id)
                    
        if file_type == 'test':
            return texts, phrase_ids
        else:
            return texts, labels, phrase_ids

    def build_vocab(self, texts):
        """
        构建词汇表

        参数：
            texts: 原始文本数据
        """
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())

        most_common = word_counts.most_common(self.max_vocab_size - 2)

        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
            
        print(f"词汇表构建完成，大小：{self.vocab_size}")
    
    def text_to_sequence(self, text):
        """
        将原始文本数据转化为索引序列

        参数：
            text: 原始文本数据

        返回：
            构建好的索引序列，长度固定为 self.max_len
        """
        words = text.split()
        sequence = []
        for word in words[:self.max_len]:
            sequence.append(self.word2idx.get(word, 1))

        if len(sequence) < self.max_len:
            sequence += [0] * (self.max_len - len(sequence))

        return sequence

    def prepare_data(self, texts, labels=None):
        """
        将文本转化为序列数据
        
        参数：
            texts: 文本列表
            labels: 标签列表（可选）
        
        返回：
            序列张量和标签张量（如果有标签）
        """
        sequences = [self.text_to_sequence(text) for text in texts]
        if labels is not None:
            return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        else:
            return torch.tensor(sequences, dtype=torch.long)

    def train_val_split(self, texts, labels):
        """
        划分训练集和验证集

        参数：
            texts: 划分之前的文本数据
            labels: 划分之前的标签数据

        返回：
            返回划分好的训练集和验证集
        """
        n = len(texts)
        indices = np.random.permutation(n)
        val_size= int(n * self.val_rate)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]

        return train_texts, train_labels, val_texts, val_labels

# ================================
# 2. 数据集类
# ================================
class TextDataset(Dataset):
    """
    文本数据集类
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ================================
# 3. 模型（CNN, RNN, Transformer）类
# ================================
class CNN(nn.Module):
    """
    CNN 文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim=100, num_filters=100,
                 filter_sizes=[3,4,5], num_classes=5, dropout=0.5,
                 pretrained_embeddings=None):
        """
        CNN 模型初始化函数

        参数：
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            num_filters: 每种卷积核数量
            filter_sizes: 卷积核大小
            num_classes: 分类的类别数目
            dropout: dropout 比例
            pretrained_embeddings: 预训练得到的词向量
        """
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                      kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, X):
        """
        CNN 前馈过程

        参数：
            X: 输入数据

        返回：
            返回最后的分类结果
        """
        embedded = self.embedding(X)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        combined = torch.cat(conv_outputs, dim=1)

        combined = self.dropout(combined)
        output = self.fc(combined)

        return output

class RNN(nn.Module):
    """
    RNN 文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=120,
                 num_layers=2, num_classes=5, dropout=0.5, bidirectional=True,
                 pretrained_embeddings=None, nonlinearity='tanh'):
        """
        RNN 初始化函数

        参数：
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            hidden_dim: 隐藏层维度
            num_layers: RNN 层数
            num_classes: 分类的类别数目
            dropout: dropout 比例
            bidirectional: 是否启用双向
            pretrained_embeddings: 预训练词向量
            nonlinearity: 非线性激活函数类型
        """
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity
        )
        
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, X):
        """
        RNN 前馈过程

        参数：
            X: 输入数据

        返回：
            返回最后的分类结果
        """
        embedded = self.embedding(X)
        
        rnn_out, hidden = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_last = hidden[-1,:,:]
        
        hidden_last = self.dropout(hidden_last)
        output = self.fc(hidden_last)
        
        return output

class Transformer(nn.Module):
    """
    Transformer 文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim=100, nhead=4,
                 num_encoder_layers=2, dim_feedforward=256,
                 num_classes=5, dropout=0.5, pretrained_embeddings=None):
        """
        Transformer 初始化函数

        参数：
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            nhead: 多头注意力数目
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈隐藏层维度
            num_classes: 分类数
            dropout: dropout 比例
            pretrained_embeddings: 预训练词向量
        """
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, X):
        """
        Transformer 前馈过程

        参数：
            X: 输入数据
        
        返回
            分类结果
        """
        padding_mask = (X == 0)

        embedded = self.embedding(X)
        embedded = self.pos_encoder(embedded)

        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)

        cls_output = transformer_out[:, 0, :]

        cls_output = self.dropout(cls_output)
        output = self.fc(cls_output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:, :X.size(1), :]
        return self.dropout(X)

# ================================
# 4. 训练器类
# ================================
class MultiClassHingeLoss(nn.Module):
    """
    多分类 Hinge 分类器

    公式: loss = sum(max(0, scores_j - score_y + 1)) for all j != y
    """
    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()

    def forward(self, scores, targets):
        """
        Hinge 损失函数实现

        参数：
            scores: 模型输出的得分
            targets: 真实标签

        返回：
            损失值
        """
        batch_size = scores.size(0)
        num_classes = scores.size(1)

        correct_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        target_one_hot = torch.zeros_like(scores)
        target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        margin = scores - correct_scores.unsqueeze(1) + 1.0
        margin = margin * (1 - target_one_hot)
        loss = torch.clamp(margin, min=0).sum(dim=1).mean()
        
        return loss

class MultiClassMSELoss(nn.Module):
    """
    多分类 MSE 损失函数
    """
    def __init__(self):
        super(MultiClassMSELoss, self).__init__()
    
    def forward(self, scores, targets):
        """
        MSE 损失函数实现

        参数：
            scores: 模型输出的得分
            targets: 真实标签

        返回：
            损失值
        """
        probs = F.softmax(scores, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=scores.size(1)).float()
        
        loss = F.mse_loss(probs, target_one_hot)
        
        return loss

class MultiClassPerceptronLoss(nn.Module):
    def __init__(self):
        super(MultiClassPerceptronLoss, self).__init__()

    def forward(self, scores, targets):
        """
        Perceptron 损失函数实现

        参数：
            scores: 模型输出的得分
            targets: 真实标签

        返回：
            损失值
        """
        batch_size = scores.size(0)
        
        correct_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        max_scores, max_indices = scores.max(dim=1)
        
        condition = (max_indices != targets)
        loss = (-correct_scores * condition.float()).mean()
        
        return loss

class Trainer:
    """
    模型训练器
    """
    def __init__(self, model, device, learning_rate=0.0001, 
                 loss_func='cross_entropy', optimizer_name='adam'):
        """
        训练器初始化函数

        参数：
            model: 模型
            device: 设备类型
            learning_rate: 学习率
            loss_func: 损失函数类型
            optimizer_name: 优化器名称
        """
        self.model = model.to(device)
        self.device = device

        if loss_func == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == 'hinge':
            self.criterion = MultiClassHingeLoss()
        elif loss_func == 'MSE':
            self.criterion = MultiClassMSELoss()
        elif loss_func == 'perceptron':
            self.criterion = MultiClassPerceptronLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader):
        """
        一个 epoch 的训练函数

        参数：
            train_loader: 训练数据加载器

        返回：
            返回一个 epoch 训练之后的平均损失
        """
        self.model.train()
        total_loss = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader):
        """
        验证函数

        参数：
            val_loader: 验证集数据加载器

        返回：
            返回本次验证的准确率
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def train(self, train_loader, val_loader, epochs=1000):
        """
        训练函数

        参数：
            train_loader: 训练集数据加载器
            val_loader: 验证集数据加载器

        返回：
            所有训练损失和验证准确率
        """
        print(f"开始训练 {self.model.__class__.__name__}...")
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
                print(f"Epoch: [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            print(f"最佳验证准确率: {best_val_acc:.4f}")
            
        return self.train_losses, self.val_accuracies

    def predict(self, test_loader):
        """
        对测试集进行预测

        参数：
            test_loader: 测试集数据加载器

        返回：
            预测标签列表
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for sequences, _ in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

        return all_preds

# ================================
# 5. GloVe 预训练词向量加载函数
# ================================
def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    """
    加载 glove 预训练词向量

    参数：
        glove_path: glove 词向量存储路径
        word2idx: 词向序号的映射表
        embedding_dim: 词向量维度

    返回：
        词嵌入完成的矩阵
    """
    print(f"从 {glove_path} 中加载 GloVe 词向量")

    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    embeddings[0] = 0

    glove_dict = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                glove_dict[word] = vector
    except FileNotFoundError:
        print(f"找不到 GloVe 文件 {glove_path}")
        return torch.FloatTensor(embeddings)

    matched = 0
    for word, idx in word2idx.items():
        if word in glove_dict:
            embeddings[idx] = glove_dict[word]
            matched += 1

    print(f"GloVe 匹配率: {matched}/{len(word2idx)} ({matched/len(word2idx)*100:.2f}%)")
    return torch.FloatTensor(embeddings)

# ================================
# 6. 提交文件生成函数
# ================================
def create_submission_file(model, test_loader, phrase_ids, output_path='/kaggle/working/submission.csv'):
    """
    生成提交文件

    参数：
        model: 训练好的模型
        test_loader: 测试集数据加载器
        phrase_ids: 测试集的PhraseId列表
        output_path: 输出文件路径
    """
    print("\n生成提交文件...")
    
    trainer = Trainer(model, DEVICE)
    predictions = trainer.predict(test_loader)
    
    submission = pd.DataFrame({
        'PhraseId': phrase_ids,
        'Sentiment': predictions
    })
    
    submission['PhraseId'] = submission['PhraseId'].astype(int)
    
    submission.to_csv(output_path, index=False)
    
    print(f"提交文件已保存到: {output_path}")
    print(f"文件包含 {len(submission)} 条预测")
    print(f"前5行预览:")
    print(submission.head())
    
    validate_submission(submission)
    
    return submission

def validate_submission(submission):
    """
    验证提交文件格式

    参数：
        submission: 提交DataFrame
    """
    print("\n验证提交文件格式:")
    
    expected_columns = ['PhraseId', 'Sentiment']
    if list(submission.columns) != expected_columns:
        print(f"⚠️ 警告: 列名不正确!")
        print(f"当前列名: {list(submission.columns)}")
        print(f"期望列名: {expected_columns}")
    else:
        print("✅ 列名正确")
    
    unique_values = submission['Sentiment'].unique()
    print(f"预测值范围: {sorted(unique_values)}")
    
    if all(0 <= val <= 4 for val in unique_values):
        print("✅ 预测值范围正确 (0-4)")
    else:
        print(f"⚠️ 警告: 预测值超出范围 0-4!")
    
    if submission.isnull().any().any():
        print("⚠️ 警告: 存在缺失值!")
    else:
        print("✅ 无缺失值")
    
    if submission['PhraseId'].is_unique:
        print("✅ PhraseId 唯一")
    else:
        print("⚠️ 警告: PhraseId 不唯一!")
    
    # 检查行数
    print(f"✅ 提交文件行数: {len(submission)}")

# ================================
# 7. 主实验流程
# ================================
def run_experiment(config):
    """
    运行实验配置并生成提交文件
    
    参数：
        config: 实验参数配置
    """
    print(f"\n{'='*60}")
    print(f"实验配置: {config}")
    print(f"{'='*60}")
    
    processor = DataProcessor(
        max_vocab_size=config.get('max_vocab_size', 5000), 
        max_len=config.get('max_len', 50)
    )
    
    try:
        train_texts, train_labels, _ = processor.load_data(TRAIN_PATH, file_type='train')
        print(f"训练数据加载完成: {len(train_texts)} 条")
        
        test_texts, test_phrase_ids = processor.load_data(TEST_PATH, file_type='test')
        print(f"测试数据加载完成: {len(test_texts)} 条")
        
    except FileNotFoundError as e:
        print(f"未找到数据文件: {e}")
        print("请确认数据文件路径是否正确")
        sys.exit(1)
    
    processor.build_vocab(train_texts)
    
    train_texts, train_labels, val_texts, val_labels = processor.train_val_split(
        train_texts, train_labels)
    
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
    
    train_seq, train_lab = processor.prepare_data(train_texts, train_labels)
    val_seq, val_lab = processor.prepare_data(val_texts, val_labels)
    test_seq = processor.prepare_data(test_texts)
    
    train_dataset = TextDataset(train_seq, train_lab)
    val_dataset = TextDataset(val_seq, val_lab)
    
    dummy_labels = torch.zeros(len(test_seq), dtype=torch.long)
    test_dataset = TextDataset(test_seq, dummy_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 64))
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 64))
    
    pretrained_embeddings = None
    if config.get('use_glove', False):
        try:
            pretrained_embeddings = load_glove_embeddings(
                GLOVE_PATH, processor.word2idx, embedding_dim=100)
        except Exception as e:
            print(f"GloVe加载失败: {e}")
            print("使用随机初始化的词向量")
    
    model_type = config.get('model_type', 'cnn')
    if model_type == 'cnn':
        model = CNN(
            vocab_size=processor.vocab_size,
            embedding_dim=100,
            num_filters=config.get('num_filters', 100),
            filter_sizes=config.get('filter_sizes', [3,4,5]),
            num_classes=5,
            dropout=config.get('dropout', 0.5),
            pretrained_embeddings=pretrained_embeddings
        )
    elif model_type == 'rnn':
        model = RNN(
            vocab_size=processor.vocab_size,
            embedding_dim=100,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=5,
            dropout=config.get('dropout', 0.5),
            bidirectional=config.get('bidirectional', True),
            pretrained_embeddings=pretrained_embeddings
        )
    elif model_type == 'transformer':
        model = Transformer(
            vocab_size=processor.vocab_size,
            embedding_dim=100,
            nhead=config.get('nhead', 4),
            num_encoder_layers=config.get('num_encoder_layers', 2),
            num_classes=5,
            dropout=config.get('dropout', 0.5),
            pretrained_embeddings=pretrained_embeddings
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model=model,
        device=DEVICE,
        learning_rate=config.get('learning_rate', 0.001),
        loss_func=config.get('loss_func', 'cross_entropy'),
        optimizer_name=config.get('optimizer', 'adam')
    )
    
    train_losses, val_accuracies = trainer.train(
        train_loader, 
        val_loader, 
        epochs=config.get('epochs', 15)
    )
    
    submission = create_submission_file(
        model=model,
        test_loader=test_loader,
        phrase_ids=test_phrase_ids,
        output_path='/kaggle/working/submission.csv'
    )
    
    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"最佳验证准确率: {max(val_accuracies):.4f}")
    print("提交文件已生成在 /kaggle/working/submission.csv")
    print(f"{'='*60}")
    
    return {
        'config': config,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': max(val_accuracies),
        'submission': submission
    }

# ================================
# 8. 单次运行（生成提交文件）
# ================================
if __name__ == "__main__":
    config = {
        'model_type': 'cnn',
        'learning_rate': 0.001,
        'loss_func': 'cross_entropy',
        'optimizer': 'adam',
        'epochs': 15,
        'use_glove': False,
        'max_vocab_size': 10000,
        'max_len': 60,
        'batch_size': 64,
        'dropout': 0.5,
        'num_filters': 128,
        'filter_sizes': [3, 4, 5]
    }
    
    result = run_experiment(config)