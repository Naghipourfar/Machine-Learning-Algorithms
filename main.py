import sys, LinearModels as lm, Graphics
import numpy as np
from sklearn import datasets

# from sklearn.metrics import mean_squared_error, r2_score

"""
    Created by Mohsen Naghipourfar on 2/8/18.
    Email : mn7697np@gmail.com
"""
NUMBER_OF_TEST_SAMPLES = 50
if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, None, 2]  # get only one feature of all
    diabetes_Y = diabetes.target
    diabetes_X_train = diabetes_X[:-NUMBER_OF_TEST_SAMPLES]
    diabetes_Y_train = diabetes_Y[:-NUMBER_OF_TEST_SAMPLES]
    diabetes_X_test = diabetes_X[-NUMBER_OF_TEST_SAMPLES:]
    # print(diabetes.keys())
    # print(diabetes.feature_names)
    # print(diabetes.data[0])
    # print(diabetes.target)
    linear_model, unused, unused = lm.linear_regression(diabetes_X_train, diabetes_Y_train)
    ridge_model, unused, unused = lm.ridge_regression(diabetes_X_train, diabetes_Y_train)
    lasso_model, unused, unused = lm.lasso(diabetes_X_train, diabetes_Y_train)
    Graphics.scatter(diabetes_X, diabetes_Y, color='purple')
    diabetes_Y_pred = linear_model.predict(diabetes_X_test)
    Graphics.plot(diabetes_X_test, diabetes_Y_pred, color='red', label='LinearReg')
    diabetes_Y_pred = ridge_model.predict(diabetes_X_test)
    Graphics.plot(diabetes_X_test, diabetes_Y_pred, color='green', label='Ridge')
    diabetes_Y_pred = lasso_model.predict(diabetes_X_test)
    Graphics.plot(diabetes_X_test, diabetes_Y_pred, color='blue', label='Lasso')
    Graphics.show()
