from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

spice_2_id = dict()
id_2_spice = dict()

def load(file_name='../input/Iris.csv'):
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
    
    # Shuffle, split and return
    x, y = shuffle(x, y)
    return train_test_split(x, np.reshape(y, [-1, 1]), test_size=0.065)