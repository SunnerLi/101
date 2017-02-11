import numpy as np
import cv2
import os

imgs = []
tags = []
folderName = 'img/train/'

def load():
    dirList = os.listdir(folderName)
    for imgName in dirList:
        img = cv2.imread(folderName+imgName, 1)
        imgs.append(img)
        if imgName.split('-')[0] == 'o':
            tags.append(1)
        else:
            tags.append(0)
    return np.asarray(imgs), np.asarray(tags)

def raiseDim(tag):
    if type(tag) == np.int64:
        _res = np.zeros([1, 2])
        _res[0][tag] = 1
        return _res
    else:
        _res = np.zeros([len(tag), 2])
        for i in range(len(tag)):
            _res[i][tag[i]] = 1
        return _res

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