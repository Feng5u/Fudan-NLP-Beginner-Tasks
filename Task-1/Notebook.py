# ================================
# 0. Import
# ================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

torch.manual_seed(42)

TRAIN_PATH = "/kaggle/input/text-classification-dataset-from-fudan-nlp/new_train.tsv"
TEST_PATH = "/kaggle/input/text-classification-dataset-from-fudan-nlp/new_test.tsv"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================================
# 1. Data Preprocessing
# ================================
class DataPreprocessor:
    def __init__(self, train_path, test_path, \
                 val_size=0.2, n=(1,1), max_features=5000, device='cpu'):
        """
        Initialization of DataPreprocessor
        
        :param train_path: The path of the train data
        :param test_path: The path of the test data
        :param val_size: The proportion of validation set
        :param n: The parameter of n-gram
        :param max_features: The maximum number of features
        :param device: The device
        """
        self.train_path = train_path
        self.test_path = test_path

        self.val_size = val_size

        self.n = n
        self.max_features = max_features

        self.device = device

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

    def read_data(self):
        """
        Read data from .tsv
        """
        self.df_train = pd.read_csv(self.train_path, sep='\t', header=None, names=['text', 'label'])
        self.df_test = pd.read_csv(self.test_path, sep='\t', header=None, names=['text', 'label'])

        print("Train Data Shape:", self.df_train.shape)
        print("Test Data Shape:", self.df_test.shape)

    def separate_data(self):
        """
        Separate the df_train into df_train and df_val
        """
        if (self.df_train is None) or (self.df_test is None):
            self.read_data()

        self.df_train, self.df_val = train_test_split(
            self.df_train,
            test_size=self.val_size,
            random_state=42,
            stratify=self.df_train['label']
        )

        print("After Split - Train Data Shape:", self.df_train.shape)
        print("After Split - Validation Data Shape:", self.df_val.shape)

    def vectorize_data(self):
        """
        Vectorize the data using n-gram
        """
        if self.df_val is None:
            self.separate_data()
        
        vectorizer = CountVectorizer(
            ngram_range=self.n,
            max_features=self.max_features
        )

        self.X_train = vectorizer.fit_transform(self.df_train['text']).toarray()
        self.X_val = vectorizer.transform(self.df_val['text']).toarray()
        self.X_test = vectorizer.transform(self.df_test['text']).toarray()

        self.y_train = self.df_train['label'].values
        self.y_val = self.df_val['label'].values
        self.y_test = self.df_test['label'].values

    def to_torch_tensors(self):
        """
        Transfer the data into torch tensors
        """
        self.vectorize_data()

        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)

        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.LongTensor(self.y_val).to(self.device)

        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.LongTensor(self.y_test).to(self.device)

        return X_train_tensor, X_test_tensor, X_val_tensor,\
               y_train_tensor, y_test_tensor, y_val_tensor

    def run(self):
        """
        Main interface function of DataPreprocessor
        """
        return self.to_torch_tensors()
    
# ================================
# 2. Training Model
# ================================
class LinearModel:
    def __init__(self, input_dim, num_classes, \
                 reg_lambda=0.0, device='cpu'):
        """
        Initialization of the LinearModel
        
        :param input_dim: Input feature dimension
        :param num_classes: Number of classes
        :param device: Device
        """
        self.reg_lambda = reg_lambda
        self.device = device

        self.W = torch.zeros(input_dim, num_classes, device=self.device)
        self.b = torch.zeros(num_classes, device=self.device)  # 修正：应该是(num_classes,)

        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

        self.num_classes = num_classes

    def forward(self, X):
        """
        Forward network
        
        :param X: Input data, shape:(batch_size, input_dim)
        """
        scores = torch.matmul(X, self.W) + self.b  # 修正：torch.nn改为torch.matmul

        scores_max = torch.max(scores, dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)

        return probs

    def predict(self, X):
        """
        Predict the label

        :param X: Input data, shape:(batch_size, input_dim)
        """
        probs = self.forward(X)
        return torch.argmax(probs, dim=1)

    def loss(self, y_hat, y, n):
        """
        Loss function
        
        :param y_hat: Predicted labels
        :param y: True labels
        :param n: Number of input data
        """
        return -torch.sum(y * torch.log(y_hat + 1e-8)) / n

    def compute_loss(self, X, y):
        """
        Compute the Loss
        
        :param X: Input data, shape:(batch_size, input_dim)
        :param y: True label, shape:(batch_size)
        """
        batch_size = X.shape[0]

        probs = self.forward(X)

        y_onehot = torch.zeros(batch_size, self.num_classes, device=self.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        loss = self.loss(probs, y_onehot, batch_size)

        if self.reg_lambda > 0:
            loss += self.reg_lambda * (torch.sum(self.W ** 2) + torch.sum(self.b ** 2))

        return loss

    def compute_accuracy(self, X, y):
        """
        Compute the accuracy
        
        :param X: Input data, shape:(batch_size, input_dim)
        :param y: True label, shape:(batch_size)
        """
        predictions = self.predict(X)
        correct = torch.sum(predictions == y).item()

        return correct / X.shape[0]

class LinearTrainer:
    def __init__(self, model, batch_size=64, learning_rate=0.01, \
                 reg_lambda=0.0, epochs=100, verbose=True, \
                 early_stopping_patience=5, device='cpu'):
        """
        Initialization of the LinearTrainer
        
        :param model: Model
        :param batch_size: Batch size
        :param learning_rate: Learning Rate
        :param reg_lambda: Regularization parameter
        :param epochs: Epochs (修正拼写错误)
        :param verbose: Visualization enable
        :param early_stopping_patience: When to stop
        :param device: Device
        """
        self.model = model
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs  # 修正：epocheds改为epochs
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience

        self.device = device

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def create_batches(self, X, y):
        """
        Create the batches
        
        :param X: Input data, shape:(n, input_dim)
        :param y: label, shape(n)
        """
        num_samples = X.shape[0]
        indices = torch.randperm(num_samples)
        batches = []

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            batches.append((batch_X, batch_y))

        return batches

    def train_epoch(self, X_train, y_train):
        """
        Train an epoch
        
        :param X_train: Train features, shape:(n, input_dim)
        :param y_train: Train labels, shape(n)
        """
        self.model.W.requires_grad_(True)
        self.model.b.requires_grad_(True)

        batches = self.create_batches(X_train, y_train)
        total_loss = 0.0
        total_acc = 0.0

        for batch_X, batch_y in batches:
            loss = self.model.compute_loss(batch_X, batch_y)

            loss.backward()

            with torch.no_grad():
                self.model.W -= self.learning_rate * self.model.W.grad  # 修正：应该是减法
                self.model.b -= self.learning_rate * self.model.b.grad  # 修正：应该是减法

            self.model.W.grad.zero_()
            self.model.b.grad.zero_()

            acc = self.model.compute_accuracy(batch_X, batch_y)
            total_loss += loss.item()  # 需要取数值
            total_acc += acc

        avg_loss = total_loss / len(batches)
        avg_acc = total_acc / len(batches)

        return avg_loss, avg_acc

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train
        
        :param X_train: Train data, shape:(n, input_dim)
        :param y_train: Train label, shape:(n)
        :param X_val: Validation_data, shape:(m, input_dim)
        :param y_val: Validation_label, shape(m)
        """
        best_val_loss = float('inf')
        patience_cnt = 0
        best_weights = None

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(X_train, y_train)

            self.model.W.requires_grad_(False)
            self.model.b.requires_grad_(False)
            
            val_loss = self.model.compute_loss(X_val, y_val).item()
            val_acc = self.model.compute_accuracy(X_val, y_val)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)  # 修正：appennd改为append

            if self.verbose and (epoch + 1) % (self.epochs / 10) == 0:
                print(f"Epoch {epoch + 1} / {self.epochs}:")
                print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                best_weights = (self.model.W.clone(), self.model.b.clone())
            else:
                patience_cnt += 1
                if patience_cnt >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    if best_weights is not None:
                        self.model.W.data = best_weights[0]
                        self.model.b.data = best_weights[1]
                    break

# ================================
# 3. Experiment
# ================================
def raw_experiment():
    print("=" * 50)
    print("0. Data Preprocessing")
    print("=" * 50)
    preprocessor = DataPreprocessor(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        device=DEVICE
    )

    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.run()
    print("Data ready for training.\n")

    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    print("=" * 50)
    print("1. Raw Experiment Start")
    print("=" * 50)
    model = LinearModel(
        input_dim=input_dim,
        num_classes=num_classes,
        device=DEVICE
    )

    trainer = LinearTrainer(
        model=model,
        device=DEVICE
    )
    
    trainer.train(X_train, y_train, X_val, y_val)

    test_loss = model.compute_loss(X_test, y_test)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print("\nTraining completed!")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
# ================================
# 4. Main
# ================================
if __name__ == "__main__":
    raw_experiment()