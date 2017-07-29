from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random
import math

# Dictionary object which can help change string of type to numeric index
type_string_2_int = dict()
type_int_2_string = dict()

# scaler
scaler = MinMaxScaler()

def load(file_name='pokemonGO.csv'):
    global type_int_2_string
    global type_string_2_int
    global scaler

    # Drop useless feature
    df = pd.read_csv(file_name)
    df = df.drop(['Name', 'Pokemon No.', 'Image URL'], axis=1)
    

    """
    # Reshape the dataframe to accept multi-type
    df = pd.melt(df, id_vars=['Max CP', 'Max HP'], var_name='type')
    df = df.dropna()
    df = df.reset_index()
    df = df.drop(['index', 'type'], axis=1)
    """
    df.columns = ['type1', 'type2', 'cp', 'hp']

    # Build mapping
    _counter = 0
    for i in range(len(df)):
        if df['type1'][i] not in type_string_2_int:
            type_string_2_int[df['type1'][i]] = _counter
            _counter += 1
        if type(df['type2'][i]) == str:
            if df['type2'][i] not in type_string_2_int:
                type_string_2_int[df['type2'][i]] = _counter
                _counter += 1
    type_int_2_string = { type_string_2_int[i]: i for i in type_string_2_int }
    print(type_int_2_string)
    
    # Change categorical data to numeric index
    for i in range(len(df)):
        df.set_value(i, 'type1', type_string_2_int[df['type1'][i]])
        if type(df['type2'][i]) == str:
            df.set_value(i, 'type2', type_string_2_int[df['type2'][i]])
    #x, y = generateData(x, y, 10)
    df = generateData(df, 10)    
    df = shuffleDataFrame(df)
    
    # Return
    y = df.get_values().T[:-2].T
    x = df.get_values().T[-2:].T
    x = scaler.fit_transform(x)
    return train_test_split(x, y, test_size=0.005)

def generateData(x, y, times=1):
    """
    x = x.tolist()
    y = y.tolist()
    result_x = list(x)
    result_y = list(y)
    if len(result_x) != len(result_y) or times<=0:
        print('generate data error!')
        exit()
    for i in range(times):
        for j in range(len(x)):
            random_seed = 1 + (random.random() - 1) / 10
            _list = []
            for k in range(len(result_x[j])):
                _list.append(result_x[j][k] * random_seed)
            result_x.append(_list)
            result_y.append(y[j])
    print(np.shape(x), np.shape(result_x))
    print(np.shape(y), np.shape(result_y))
    return result_x, result_y
    """
    pass

def generateData(df, times=1):
    """
    _new = pd.DataFrame([[12, float('NaN'), 1, 1]], columns=df.columns)
    df = df.append(_new)
    

    """
    origin_len = len(df)
    columns_list = df.columns
    print(origin_len)
    print(df['type1'][150])
    for i in range(times):
        for j in range(origin_len):
            random_seed = 1 + (random.random() - 1) / 10
            _new_row = []
            _new_row.append(df['type1'][j])
            _new_row.append(df['type2'][j])
            _new_row.append(df['cp'][j] * random_seed)
            _new_row.append(df['hp'][j] * random_seed)
            df = df.append(pd.DataFrame([_new_row], columns=columns_list))
        df = df.reset_index().drop(['index'], axis=1)
    return df

def shuffleDataFrame(data_frame):
    """
        Shuffle the data frame
        Arg:    data_frame  - The data frame object you want to shuffle
        Ret:    The shuffle data frame result
    """
    result_table = pd.DataFrame(columns=data_frame.columns)
    rest_table = data_frame
    for i in range(len(data_frame)):
        sample_row = rest_table.sample(1)
        rest_table = rest_table.drop(sample_row.index[0])
        result_table = result_table.append(sample_row)
    return result_table

def errorSum(tag_arr, predict_arr):
    _count = 0
    for i in range(len(tag_arr)):
        if predict_arr[i] == tag_arr[i][0] or \
            predict_arr[i] == tag_arr[i][1]:
            _count += 1
    return _count / len(tag_arr)

def oneHotEncode(arr):
    res = np.zeros([len(arr), np.nanmax(arr) + 1])
    for i in range(len(res)):
        res[i][arr[i][0]] = 1
        if math.isnan(arr[i][1]) == False:
            res[i][arr[i][1]] = 1
    return res

def oneHotDecode(arr):
    return np.argmax(arr, axis=1)

"""
a = [[ 0.00138773, 0.0105505, 0.06052515, 0.01134599]]
print(oneHotDecode(a))
"""