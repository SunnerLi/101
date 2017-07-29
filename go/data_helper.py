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
    """
        Load the training data.
        This function will build the categorical mapping and transfer into numeric representation

        Arg:    file_name   - The name of the pokemon GO file
        Ret:    The pandas data frame object
    """
    global type_int_2_string
    global type_string_2_int
    global scaler

    # Drop useless feature
    df = pd.read_csv(file_name)
    df = df.drop(['Name', 'Pokemon No.', 'Image URL'], axis=1)
    df.columns = ['type1', 'type2', 'cp', 'hp']

    # Swap column
    column_list = list(df)
    column_list[0], column_list[2] = column_list[2], column_list[0]
    column_list[1], column_list[3] = column_list[3], column_list[1]
    df = df.ix[:, column_list]

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

    # Change categorical data to numeric index
    for i in range(len(df)):
        df.set_value(i, 'type1', type_string_2_int[df['type1'][i]])
        if type(df['type2'][i]) == str:
            df.set_value(i, 'type2', type_string_2_int[df['type2'][i]])    
    return df

def mergeMultipleTypes(df):
    """
        Reshape the dataframe to accept multi-type

        Arg:    df  - The pokemen data frame which has two type attributes
        Ret:    The merged data frame object
    """
    # 
    df = pd.melt(df, id_vars=['cp', 'hp'], var_name='type')
    df = df.dropna()
    df = df.reset_index()
    df = df.drop(['index', 'type'], axis=1)
    return df

def splitData(df, shuffle=True, split_rate=0.005):
    """
        Split the data into training and testing partition
        It will also shuffle the data if the flag is True

        Arg:    df          - The data frame object which will be cropped
                shuffle     - The flag that determine shuffling or not
                split_rate  - The partition rate of testing data
    """
    df = generateData(df, 10)
    if shuffle == True:   
        df = shuffleDataFrame(df)
    y = df.get_values().T[2:].T
    x = df.get_values().T[:2].T
    x = scaler.fit_transform(x)
    return train_test_split(x, y, test_size=0.005)

def generateData(df, times=1):
    """
        Generate data randomly

        Arg:    df      - The original data frame object
                times   - The times of number you want to generate
    """
    origin_len = len(df)
    columns_list = df.columns
    print(columns_list)
    for i in range(times):
        for j in range(origin_len):
            random_seed = 1 + (random.random() - 1) / 10
            _new_row = []
            _new_row.append(df['cp'][j] * random_seed)
            _new_row.append(df['hp'][j] * random_seed)
            if len(columns_list) == 4:
                _new_row.append(df['type1'][j])
                _new_row.append(df['type2'][j])
            if len(columns_list) == 3:
                _new_row.append(df['value'][j])
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

def matchRate(tag_arr, predict_arr):
    """
        Get the match rate for the predict result
        It only accepts two format:
        1. [[1, 0], [3, NaN], ...]
        2. [1, 0, 3, NaN, ...]

        Arg:    tag_arr     - The tag array, it can be 1D or 2D
                predict_arr - The predict array
        Ret:    The matching rate
    """
    _count = 0
    if len(np.shape(tag_arr)) == 2:
        for i in range(len(tag_arr)):
            if predict_arr[i] == tag_arr[i][0] or \
                predict_arr[i] == tag_arr[i][1]:
                _count += 1
    else:
        for i in range(len(tag_arr)):
            if predict_arr[i] == tag_arr[i]:
                _count += 1
    return _count / len(tag_arr)

def oneHotEncode(arr):
    """
        Transfer the array into one-hot format
        This function allow one row containing multiple classes

        Arg:    The original tag array
        Ret:    The encoded array
    """
    res = np.zeros([len(arr), np.nanmax(arr) + 1])
    for i in range(len(res)):
        res[i][arr[i][0]] = 1
        if math.isnan(arr[i][1]) == False:
            res[i][arr[i][1]] = 1
    return res

def oneHotDecode(arr):
    """
        Reverse the predict result into original format

        Arg:    The array of predict result
        Ret:    The array with original format
    """
    return np.argmax(arr, axis=1)