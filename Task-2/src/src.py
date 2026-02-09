import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataPreproceesor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.df_train = None
        self.df_test = None
        
    def read_data(self):
        self.df_train = pd.read_csv(self.train_path, sep='\t', header=["text"], "label")
        self.df_test = pd.read_csv(self.test_path, sep='\t', header=["text", "label"])

        print("Train Data Shape: ", self.df_train.shape)
        print("Test Data Shape: ", self.df_test.shape)
    
    def separate_data(self):
        if self.df_train == None:
            self.read_data()

        self.df_train, self.df_val = train_test_split(
            self.df_train,
            test_size=self.val_size,
            random_state=42,
            stratify=self.df_tarin["label"]
        )

        print("Train Data Shape(After Split): ", self.df_train.shape)
        print("Test Data Shape(After Split): ", self.df_val.shape)
    
    def 

class CNNModel:

class Train:

def raw_experiment():

if __name__ == "__main__":
    raw_experiment()