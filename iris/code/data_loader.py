from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

# Mapping object
spice_2_id = dict()
id_2_spice = dict()

# Normalize scaler
scaler = MinMaxScaler()

def load(file_name='../input/Iris.csv'):
    global spice_2_id
    global id_2_spice
    global scaler

    # Load data
    df = pd.read_csv(file_name)
    x = df.get_values().T[1:5].T
    y = df.get_values().T[-1].T

    # Build mapping and transfer
    counter = 0
    for name in y:
        if name not in spice_2_id:
            spice_2_id[name] = counter
            counter += 1
    for i in range(len(y)):
        y[i] = spice_2_id[y[i]]
    y = np.asarray(y, dtype=np.float)
    id_2_spice = { spice_2_id[x]: x for x in spice_2_id }

    # normalize the x
    x = scaler.fit_transform(x)
    
    # Shuffle, split and return
    x, y = shuffle(x, y)
    return train_test_split(x, np.reshape(y, [-1, 1]), test_size=0.065)

def oneHotEncode(arr):
    return pd.get_dummies(np.reshape(arr, [-1])).values

def oneHotDecode(arr):
    return np.argmax(np.round(arr), axis=1)