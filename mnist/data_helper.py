import pandas as pd
import numpy as np

def load(train_file_name='train.csv'):
    x = pd.read_csv(train_file_name).get_values().T[1:].T
    y = pd.read_csv(train_file_name).get_values().T[0].T
    return x, y