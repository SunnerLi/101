import pandas as pd
import numpy as np

def load(train_file_name='train.csv', test_file_name='test.csv'):
    train_x = pd.read_csv(train_file_name).get_values().T[1:].T
    train_y = pd.read_csv(train_file_name).get_values().T[0].T
    test_x = pd.read_csv(test_file_name).get_values().T[1:].T
    test_y = pd.read_csv(test_file_name).get_values().T[0].T
    return train_x, test_x, train_y, test_y