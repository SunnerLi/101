from sklearn import datasets
import numpy as np

def load():
    all = datasets.load_boston()
    return all['data'], all['target']

def shuffleDataAndLabel(src1, src2):
    """
        Training data shuffle

        Arg:    src1 - The source numpy object
                src2 - The label of the source
        Ret:    The source matrix and label after shuffled
    """
    if not np.shape(src1)[0] == np.shape(src2)[0]:
        return None
    else:
        # Shuffle the index list
        indexArr = np.asarray(range(np.shape(src1)[0]))
        np.random.shuffle(indexArr)

        # Build the corresponding result
        dst1 = []
        dst2 = []
        for i in range(np.shape(indexArr)[0]):
            dst1.append(src1[indexArr[i]])
            dst2.append(src2[indexArr[i]])
        return np.asarray(dst1), np.asarray(dst2)

def tenCrossValid(src1, src2, __index):
    if not len(src1) == len(src2):
        print "the length isn't equal..."
        return None
    else:
        train_src1 = []
        train_src2 = []
        test_src1 = []
        test_src2 = []
        min_index = (float(__index) / 10) * len(src1)
        max_index = (float(__index+1) / 10) * len(src1)
        for i in range(len(src1)):
            if i >= min_index and i < max_index:
                test_src1.append(src1[i])
                test_src2.append(src2[i])
            else:
                train_src1.append(src1[i])
                train_src2.append(src2[i])
        return train_src1, train_src2, test_src1, test_src2