import sys

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/8/18.
    Email : mn7697np@gmail.com
"""


def regression_MSE(y, y_pred):  # mean squared error
    if len(y) != len(y_pred):
        return Exception("y and y_pred must be in same length")
    s = sum((y - y_pred) ** 2.0)
    n = len(y)
    return s / n


def classification_MSE(y, y_pred):
    if len(y) != len(y_pred):
        return Exception("y and y_pred must be in same length")
    n = len(y)
    s = 0
    for i in range(n):
        if y[i] == y_pred[i]:
            s += 1
    return s / n


def linear_regression(data):
    pass


def linear_classification(data):
    pass