# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:46:44 2019

@author: Winham

LoadBatches1D.py: 迭代生成训练时的batch

实现参考：https://github.com/divamgupta/image-segmentation-keras/blob/master/LoadBatches.py

"""

import os
import itertools
import numpy as np
from sklearn import preprocessing as prep


def getSigArr(path, sigNorm='scale'):
    sig = np.load(path)
    if sigNorm == 'scale':
        sig = prep.scale(sig)
    elif sigNorm == 'minmax':
        min_max_scaler = prep.MinMaxScaler()
        sig = min_max_scaler.fit_transform(sig)
    return np.expand_dims(sig, axis=1)


def getSegmentationArr(path, nClasses=3, output_length=1800, class_value=[0, 0.5, 1]):
    # class_value是在generate_labels.py中定义的，背景0，正常0.5，早搏1，必须要保持一致
    seg_labels = np.zeros([output_length, nClasses])
    seg = np.load(path)
    for i in range(nClasses):
        seg_labels[:, i] = (seg == class_value[i]).astype(float)
    return seg_labels


def SigSegmentationGenerator(sigs_path, segs_path, batch_size, n_classes, output_length=1800):
    sigs = os.listdir(sigs_path)
    segmentations = os.listdir(segs_path)
    sigs.sort()
    segmentations.sort()
    for i in range(len(sigs)):
        sigs[i] = sigs_path + sigs[i]
        segmentations[i] = segs_path + segmentations[i]
    assert len(sigs) == len(segmentations)
    for sig, seg in zip(sigs, segmentations):
        assert (sig.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    zipped = itertools.cycle(zip(sigs, segmentations))
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            sig, seg = next(zipped)
            X.append(getSigArr(sig))
            Y.append(getSegmentationArr(seg, n_classes, output_length))
        yield np.array(X), np.array(Y)
