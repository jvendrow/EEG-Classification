from os.path import dirname, abspath, exists, join
from os import makedirs, getcwd
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle

def plot_channel_perclass(data_class, channel_list):
    for i in channel_list:
        channel_10_avg_class0 = np.mean(data_class[0][:, i, :],
                                        axis=0) - np.mean(
            data_class[0][:, i, :])
        channel_10_avg_class1 = np.mean(data_class[1][:, i, :],
                                        axis=0) - np.mean(
            data_class[1][:, i, :])
        channel_10_avg_class2 = np.mean(data_class[2][:, i, :],
                                        axis=0) - np.mean(
            data_class[2][:, i, :])
        channel_10_avg_class3 = np.mean(data_class[3][:, i, :],
                                        axis=0) - np.mean(
            data_class[3][:, i, :])

        plt.figure(figsize=(8, 4))
        plt.plot(channel_10_avg_class0, label='class 0')
        plt.plot(channel_10_avg_class1, label='class 1')
        plt.plot(channel_10_avg_class2, label='class 2')
        plt.plot(channel_10_avg_class3, label='class 3')
        plt.legend(loc="upper right")
        plt.title('Average EEG signal of channel {} for a given class'.format(i))
        plt.xlabel('timestep')
        plt.ylabel('potential')
        plt.show()


def save_data_pickle(dict_obj, save_path):
    path = join(save_path, 'aug_data.pickle')
    print("Saving data pickle...")
    with open(path, 'wb') as fp:
        pickle.dump(dict_obj, fp)
    print("Data pickle Saved.")

def load_data_pickle(load_path):
    path = join(load_path, 'aug_data.pickle')
    print("Loading data pickle...")
    with open(path, 'rb') as fp:
        data_dict = pickle.load(fp)
    print("Data pickle loaded.")
    return data_dict


def load_data_original():
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_train_valid = np.load("data/person_train_valid.npy")
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    person_test = np.load("data/person_test.npy")

    print('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print('Test target shape: {}'.format(y_test.shape))
    print('Person train/valid shape: {}'.format(person_train_valid.shape))
    print('Person test shape: {}'.format(person_test.shape))

    return {'X_test': X_test, 'y_test': y_test,
            'X_train_valid': X_train_valid, 'y_train_valid': y_train_valid}

def get_root_path():
    return getcwd()


def get_data_path():
    return join(get_root_path(), 'data')


def get_save_path():
    return join(get_root_path(), 'save')


def ensure_dir(d):
    if not exists(d):
        makedirs(d)
