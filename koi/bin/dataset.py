#!/usr/bin/python3

import pickle
import os
import random
from PIL import Image
import numpy as np

def shuffle(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    # high dimensional array warning
    if len(y.shape) > 2:
        warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
    # flatten high dimensional array
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y

def load_from_dir(directory, resize=None):
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    label = 0
    dirs = sorted(os.walk(directory).__next__()[1])
    for d in dirs:
        files = sorted(os.walk(directory + d).__next__()[2])
        test_img_file = random.choice(files)
        for file in files:
            if test_img_file == file:
                test_img = Image.open(os.path.join(directory + d, file))
                if resize:
                    test_img = test_img.resize(resize, Image.ANTIALIAS)
                test_samples.append(np.asarray(test_img, dtype="float32")/255.)
                test_labels.append(label)
            img = Image.open(os.path.join(directory + d, file))
            if resize:
                img = img.resize(resize, Image.ANTIALIAS)
            train_samples.append(np.asarray(img, dtype="float32")/255.)
            train_labels.append(label)
        label += 1
    return train_samples, train_labels, test_samples, test_labels

def load_dataset(directory, dataset_file, resize=None, shuffle_data=False, one_hot=False):
    try:
        X_train, X_label, Y_test, Y_label = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X_train, X_label, Y_test, Y_label = load_from_dir(directory, resize)
        pickle.dump((X_train, X_label, Y_test, Y_label), open(dataset_file, 'wb'))
    if one_hot:
        X_label = to_categorical(X_label, np.max(X_label) + 1)
        Y_label = to_categorical(Y_label, np.max(Y_label) + 1)
    if shuffle_data:
        X_train, X_label = shuffle(X_train, X_label)
        Y_test, Y_label = shuffle(Y_test, Y_label)
    return X_train, X_label, Y_test, Y_label
