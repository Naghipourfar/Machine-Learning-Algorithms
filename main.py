import sys
import LinearModels as lm
import Graphics
import numpy as np
from sklearn import datasets

from sklearn.metrics import mean_squared_error, r2_score

"""
    Created by Mohsen Naghipourfar on 2/8/18.
    Email : mn7697np@gmail.com
"""
NUMBER_OF_TEST_SAMPLES = 50
if __name__ == '__main__':
    diabetes = datasets.load_diabetes()
    print(diabetes.data.shape)
    for i in range(diabetes.data.shape[1]):
        diabetes_X = diabetes.data[:, np.newaxis, i]  # get only one feature of all
        diabetes_Y = diabetes.target
        diabetes_X_train = diabetes_X[:-NUMBER_OF_TEST_SAMPLES]
        diabetes_Y_train = diabetes_Y[:-NUMBER_OF_TEST_SAMPLES]
        diabetes_X_test = diabetes_X[-NUMBER_OF_TEST_SAMPLES:]
        diabetes_Y_test = diabetes_Y[-NUMBER_OF_TEST_SAMPLES:]
        # print(diabetes.keys())
        # print(diabetes.feature_names)
        # print(diabetes.data[0])
        # print(diabetes.target)
        linear_model, unused, unused = lm.linear_regression(diabetes_X_train, diabetes_Y_train)
        ridge_model, unused, unused = lm.ridge_regression(diabetes_X_train, diabetes_Y_train)
        lasso_model, unused, unused = lm.lasso(diabetes_X_train, diabetes_Y_train)
        Graphics.scatter(diabetes_X, diabetes_Y, color='purple')
        diabetes_Y_prediction = linear_model.predict(diabetes_X_test)
        Graphics.plot(diabetes_X_test, diabetes_Y_prediction, color='red', label='LinearReg')
        print(mean_squared_error(diabetes_Y_test, diabetes_Y_prediction))
        diabetes_Y_prediction = ridge_model.predict(diabetes_X_test)
        Graphics.plot(diabetes_X_test, diabetes_Y_prediction, color='green', label='Ridge')
        diabetes_Y_prediction = lasso_model.predict(diabetes_X_test)
        Graphics.plot(diabetes_X_test, diabetes_Y_prediction, color='blue', label='Lasso')
        Graphics.show()
