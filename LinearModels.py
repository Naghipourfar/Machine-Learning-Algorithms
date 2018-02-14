import sys
import pandas as pd
import numpy as np
from sklearn import linear_model

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


def regression_RSS(y, y_pred):
    if len(y) != len(y_pred):
        return Exception("y and y_pred must be in same length")
    n = len(y)
    return n * regression_MSE(y, y_pred)


def regression_RSE(y, y_pred):
    if len(y) != len(y_pred):
        return Exception("y and y_pred must be in same length")
    n = len(y)
    return pow(regression_RSS(y, y_pred), 0.5)


def correlation(x, y):
    if len(x) != len(y):
        return Exception("X and Y must be in same length")
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x - mean_x) * (mean_y))
    denominator = pow(sum((x - mean_x) ** 2), 0.5) * pow(sum((y - mean_y) ** 2), 0.5)
    return numerator / denominator


def linear_regression(x, y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model, model.coef_, model.intercept_


def ridge_regression(x, y):
    model = linear_model.Ridge(alpha=0.5)
    model.fit(x, y)
    return model, model.coef_, model.intercept_


def linear_classification(data):
    pass
