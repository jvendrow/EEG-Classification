from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import os

def load_data(type = 'alex'):
    os.chdir('data')
    ret = dict()

    if type == 'original':
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
        person_train_valid = np.load("person_train_valid.npy")
        X_train_valid = np.load("X_train_valid.npy")
        y_train_valid = np.load("y_train_valid.npy")
        person_test = np.load("person_test.npy")
        ret['X_test'] = X_test
        ret['y_test'] = y_test
        ret['person_train_valid'] = person_train_valid
        ret['X_train_valid'] = X_train_valid
        ret['y_train_valid'] = y_train_valid
        ret['person_test'] = person_test
    elif type == 'alex':
        total_X_test = np.load("total_X_test.npy")
        total_y_test = np.load("total_y_test.npy")
        total_X_train = np.load("total_X_train.npy")
        total_y_train = np.load("total_y_train.npy")
        total_X_val = np.load("total_X_val.npy")
        total_y_val = np.load("total_y_val.npy")
        ret['total_X_test'] = total_X_test
        ret['total_y_test'] = total_y_test
        ret['total_X_train'] = total_X_train
        ret['total_y_train'] = total_y_train
        ret['total_X_val'] = total_X_val
        ret['total_y_val'] = total_y_val
        ret['type'] = type
    os.chdir('..')
    return ret

if __name__ == '__main__':
    data = load_data('alex')
    print("data loaded")
