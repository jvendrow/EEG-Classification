import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def augment_ops(data_dict, trim=True, average=5, subsample=5, noise=True, maxpool=True):
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    X_train_valid = data_dict['X_train_valid']
    y_train_valid = data_dict['y_train_valid']
    encoder = LabelBinarizer()
    y_train_valid = encoder.fit_transform(y_train_valid)
    y_test = encoder.fit_transform(y_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train_valid,
                                                      y_train_valid,
                                                      test_size=0.2,
                                                      random_state=42)

    total_X_train = None
    total_y_train = None
    total_X_val = None
    total_y_val = None
    total_X_test = None
    total_y_test = None

    if trim:
        X_train = X_train[:, :, 0:500]
        X_val = X_val[:, :, 0:500]
        X_test = X_test[:, :, 0:500]

    # MaxPooling
    if maxpool:
        X_train_max = np.max(X_train.reshape(X_train.shape[0], X_train.shape[1], -1, subsample), axis=3)
        X_val_max = np.max(X_val.reshape(X_val.shape[0], X_val.shape[1], -1, subsample), axis=3)
        X_test_max = np.max(X_test.reshape(X_test.shape[0], X_test.shape[1], -1, subsample), axis=3)

        total_X_train = X_train_max
        total_X_val = X_val_max
        total_X_test = X_test_max
        total_y_train = y_train
        total_y_val = y_val
        total_y_test = y_test


    # Average every 5 timesteps -> 1 set of 2115 samples of 100 timesteps
    k = 0
    if average > 0:
        X_train_average = np.mean(X_train.reshape(X_train.shape[0],
                                                  X_train.shape[1], -1, average),
                                  axis=3)
        if noise:
            X_train_average = X_train_average + np.random.normal(0.0, 0.5, X_train_average.shape)

        X_val_average = np.mean(X_val.reshape(X_val.shape[0],
                                              X_val.shape[1], -1, average),
                                axis=3)
        if noise:
            X_val_average = X_val_average + np.random.normal(0.0, 0.5,
                                                             X_val_average.shape)

        X_test_average = np.mean(X_test.reshape(X_test.shape[0],
                                                X_test.shape[1], -1, average),
                                 axis=3)
        if noise:
            X_test_average = X_test_average + np.random.normal(0.0, 0.5, X_test_average.shape)

        if total_X_train is None:
            total_X_train = X_train_average
            total_X_val = X_val_average
            total_X_test = X_test_average
            total_y_train = y_train
            total_y_val = y_val
            total_y_test = y_test
        else:
            total_X_train = np.vstack((total_X_train, X_train_average))
            total_X_val = np.vstack((total_X_val, X_val_average))
            total_X_test = np.vstack((total_X_test, X_test_average))
            total_y_train = np.vstack((total_y_train, y_train))
            total_y_val = np.vstack((total_y_val, y_val))
            total_y_test = np.vstack((total_y_test, y_test))

    elif subsample != -1:
        X_train_subsample = X_train[:, :, 0::subsample] + \
                            (np.random.normal(0.0, 0.5, X_train[:, :,
                                                        0::subsample].shape) if noise else 0.0)
        X_val_subsample = X_val[:, :, 0::subsample] + \
                            (np.random.normal(0.0, 0.5, X_val[:, :,
                                                        0::subsample].shape) if noise else 0.0)
        X_test_subsample = X_test[:, :, 0::subsample] + \
                           (np.random.normal(0.0, 0.5, X_test[:, :,
                                                        0::subsample].shape) if noise else 0.0)

        if total_X_train is None:
            total_X_train = X_train_subsample
            total_X_val = X_val_subsample
            total_X_test = X_test_subsample
            total_y_train = y_train
            total_y_val = y_val
            total_y_test = y_test
        else:
            total_X_train = np.vstack((total_X_train, X_train_subsample))
            total_X_val = np.vstack((total_X_val, X_val_subsample))
            total_X_test = np.vstack((total_X_test, X_test_subsample))
            total_y_train = np.vstack((total_y_train, y_train))
            total_y_val = np.vstack((total_y_val, y_val))
            total_y_test = np.vstack((total_y_test, y_test))
        k += 1

    # Subsample every 5 timetimes -> 5 sets of 2115 samples of 100 timesteps
    for i in range(k, subsample):
        X_train_subsample = X_train[:, :, i::subsample] + \
                            (np.random.normal(0.0, 0.5, X_train[:, :,i::subsample].shape) if noise else 0.0)
        X_test_subsample = X_test[:, :, i::subsample] + np.random.normal(0.0, 0.5,
                                                                      X_test[
                                                                      :, :,
                                                                      i::subsample].shape)
        X_val_subsample = X_val[:, :, 0::subsample] + \
                          (np.random.normal(0.0, 0.5, X_val[:, :,
                                                      0::subsample].shape) if noise else 0.0)

        # Concatenate them together to make a big dataset
        total_X_train = np.vstack((total_X_train, X_train_subsample))
        total_X_test = np.vstack((total_X_test, X_test_subsample))
        total_X_val = np.vstack((total_X_val, X_val_subsample))
        total_y_val = np.vstack((total_y_val, y_val))
        total_y_train = np.vstack((total_y_train, y_train))
        total_y_test = np.vstack((total_y_test, y_test))

    return {'total_X_train': total_X_train, 'total_X_test': total_X_test,
            'total_X_val': total_X_val, 'total_y_val': total_y_val,
            'total_y_train': total_y_train, 'total_y_test': total_y_test}


def windowing(X, y, window_size=500, step=100):
    X_window = []
    y_window = []
    for i, x_data in enumerate(X):
        steps = (1000 - window_size) // step + 1
        for j in range(steps):
            new_x_data = x_data[:,j*step:j*step+window_size]
            X_window.append(new_x_data)
            y_window.append(y[i])
    X_window = np.array(X_window)
    y_window = np.array(y_window)
    return X_window, y_window

