# ================================
# 0. Import
# ================================
import time
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
    def __init__(self, train_path, test_path, use_tfidf=True, \
                 val_size=0.2, n=(1,2), max_features=10000, device='cpu'):
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
        self.use_tfidf = use_tfidf

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
        
        if self.use_tfidf is True:
            vectorizer = TfidfVectorizer(
                ngram_range=self.n,
                max_features=self.max_features
            )
        else:
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

        feature_dimension = len(vectorizer.vocabulary_)
        print("Feature dimension: ", feature_dimension)

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
                 reg_lambda=0.0, function="cross_entropy", device='cpu'):
        """
        Initialization of the LinearModel
        
        :param input_dim: Input feature dimension
        :param num_classes: Number of classes
        :param device: Device
        """
        self.reg_lambda = reg_lambda
        self.function = function
        self.device = device

        self.W = torch.randn(input_dim, num_classes, device=self.device) * 0.01
        self.b = torch.zeros(num_classes, device=self.device)

        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

        self.num_classes = num_classes

    def forward(self, X):
        """
        Forward network
        
        :param X: Input data, shape:(batch_size, input_dim)
        """
        scores = torch.matmul(X, self.W) + self.b

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
        :param function: Loss function type
        """
        if self.function == "cross_entropy":
            return -torch.sum(y * torch.log(y_hat + 1e-8)) / n

        elif self.function == "mse":
            return torch.sum((y - y_hat) ** 2) / n
        
        elif self.function == "hinge":
            correct_scores = torch.sum(y_hat * y, dim=1)
            other_scores = y_hat * (1 - y) - y * 1e10
            max_other_scores, _ = torch.max(other_scores, dim=1)
            
            hinge_loss = torch.clamp(1 + max_other_scores - correct_scores, min=0)
            return torch.sum(hinge_loss) / n
        
        elif self.function == "perceptron":
            correct_scores = torch.sum(y_hat * y, dim=1)
            other_scores = y_hat * (1 - y) - y * 1e10
            max_other_scores, _ = torch.max(other_scores, dim=1)
            
            perceptron_loss = torch.clamp(-(correct_scores - max_other_scores), min=0)
            return torch.sum(perceptron_loss) / n

    def compute_loss(self, X, y):
        """
        Compute the Loss
        
        :param X: Input data, shape:(batch_size, input_dim)
        :param y: True label, shape:(batch_size)
        :param function: Loss function type
        """
        batch_size = X.shape[0]

        probs = self.forward(X)

        y_onehot = torch.zeros(batch_size, self.num_classes, device=self.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        loss = self.loss(probs, y_onehot, batch_size)

        if self.reg_lambda > 0:
            loss += self.reg_lambda * (torch.sum(self.W ** 2) + torch.sum(self.b ** 2))

        return loss
    
    def compute_val_loss(self, X, y):
        """
        Compute the Loss
        
        :param X: Input data, shape:(batch_size, input_dim)
        :param y: True label, shape:(batch_size)
        :param function: Loss function type
        """
        batch_size = X.shape[0]

        probs = self.forward(X)

        y_onehot = torch.zeros(batch_size, self.num_classes, device=self.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        loss = self.loss(probs, y_onehot, batch_size)
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
                 epochs=10000, verbose=True, \
                 early_stopping_patience=10, device='cpu'):
        """
        Initialization of the LinearTrainer
        
        :param model: Model
        :param batch_size: Batch size
        :param learning_rate: Learning Rate
        :param epochs: Epochs
        :param verbose: Visualization enable
        :param early_stopping_patience: When to stop
        :param device: Device
        """
        self.model = model
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
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
                self.model.W -= self.learning_rate * self.model.W.grad
                self.model.b -= self.learning_rate * self.model.b.grad

            self.model.W.grad.zero_()
            self.model.b.grad.zero_()

            acc = self.model.compute_accuracy(batch_X, batch_y)
            total_loss += loss.item()
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
            
            val_loss = self.model.compute_val_loss(X_val, y_val).item()
            val_acc = self.model.compute_accuracy(X_val, y_val)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

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
        reg_lambda=1e-4,
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
    
def feature_experiment(n_tmp, mf):
    print("=" * 50)
    print("0. Data Preprocessing")
    print("=" * 50)
    preprocessor = DataPreprocessor(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        n=n_tmp,
        max_features=mf,
        device=DEVICE
    )

    X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.run()
    print("Data ready for training.\n")

    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    print("=" * 50)
    print("1. Experiment Start")
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
    
    start_time = time.time()
    trainer.train(X_train, y_train, X_val, y_val)
    end_time = time.time()

    test_loss = model.compute_loss(X_test, y_test)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print("\nTraining completed!")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Time Cost: {end_time - start_time} s")

def lr_experiment(lr, fc):
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
    print("1. Experiment Start")
    print("=" * 50)
    model = LinearModel(
        input_dim=input_dim,
        num_classes=num_classes,
        function=fc,
        device=DEVICE
    )

    trainer = LinearTrainer(
        model=model,
        learning_rate=lr,
        device=DEVICE
    )
    
    start_time = time.time()
    trainer.train(X_train, y_train, X_val, y_val)
    end_time = time.time()

    test_loss = model.compute_loss(X_test, y_test)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print("\nTraining completed!")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Time cost: {end_time - start_time} s")

def rl_experiment(rl):
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
    print("1. Experiment Start")
    print("=" * 50)
    model = LinearModel(
        input_dim=input_dim,
        num_classes=num_classes,
        reg_lambda=rl,
        device=DEVICE
    )

    trainer = LinearTrainer(
        model=model,
        device=DEVICE
    )
    
    start_time = time.time()
    trainer.train(X_train, y_train, X_val, y_val)
    end_time = time.time()

    test_loss = model.compute_loss(X_test, y_test)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print("Training completed!")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Time cost: {end_time - start_time} s")

# ================================
# 4. Main
# ================================
if __name__ == "__main__":
    # To verify the code's function
    raw_experiment()

    # To compare different n and max_feature
    # for n in [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]:
    #     for mf in [1000, 5000, 10000, 20000]:
    #         print("=" * 50)
    #         print("Experiment: Changing parameter max_feature & n")
    #         print(f"max_feature: {mf}   n: {n}")
    #         print("=" * 50)
    #         feature_experiment(n, mf)
    #         print()

    # To compare different loss function and learning_rate
    # for fc in ("cross_entropy", "mse", "hinge", "perceptron"):
    #     for lr in [0.001, 0.01, 0.1, 0.5]:
    #         print("=" * 50)
    #         print("Experiment: Changing parameter learning_rate & loss function type")
    #         print(f"learning_rate: {lr}    loss function type: {fc}")
    #         print("=" * 50)
    #         lr_experiment(lr, fc)
    #         print()

    # To compare different regular lambda
    # for rl in [0.001, 0.01, 0.1, 1, 10]:
    #     print("=" * 50)
    #     print("Experimen: Changing parameter: regular lambda")
    #     print(f"regular lambda: {rl}")
    #     print("=" * 50)
    #     rl_experiment(rl)
    #     print()