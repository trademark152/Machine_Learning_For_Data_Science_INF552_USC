from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    # Split the training data from test data in the ratio specified in
    # test_size
    # get the splitting index by division (floor)
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X,default is divided by L2 norm of each row data """
    # Convert inputs to arrays with at least one dimension.
    # Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))

    # in case norm is 0
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X):
    """ Standardize the dataset X, based on mean and std of each column variable """
    X_std = X

    # Compute the mean along the row.
    mean = X.mean(axis=0)

    # Compute the standard deviation along the column.
    std = X.std(axis=0)

    for col in range(np.shape(X)[1]):
        if std[col]: # only if std[col] is not zero
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


# the special syntax **kwargs in function definitions in python is used to pass a keyworded, variable-length argument list.
def linear_kernel(**kwargs):
    """linear kernel"""
    def f(x1, x2):
        # Inner product of two arrays.
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    """polynomial kernal"""
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    """ radia; basis function"""
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f
