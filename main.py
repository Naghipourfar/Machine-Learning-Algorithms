import sys, LinearModels, Graphics
import numpy as np
from sklearn import datasets

# from sklearn.metrics import mean_squared_error, r2_score

"""
    Created by Mohsen Naghipourfar on 2/8/18.
    Email : mn7697np@gmail.com
"""

if __name__ == '__main__':
    iris = datasets.load_iris()
    diabetes = datasets.load_diabetes()
    diabetes_X = diabetes.data[:, None, 2]  # ???
    diabetes_Y = diabetes.target
    diabetes_X_test = diabetes_X[-20:]
    print(diabetes.keys())
    print(diabetes.feature_names)
    print(diabetes.data[0])
    print(diabetes.target)
    model, coefs, intercept = LinearModels.linear_regression(diabetes_X, diabetes_Y)
    print('Coefficients: \n', coefs)
    # print("Mean squared error: {0}".format(mean_squared_error(diabets.data, diabets.target)))
    diabetes_Y_pred = model.predict(diabetes_X_test)
    Graphics.scatter(diabetes_X, diabetes_Y)
    Graphics.plot(diabetes_X_test, diabetes_Y_pred)
    Graphics.show()
