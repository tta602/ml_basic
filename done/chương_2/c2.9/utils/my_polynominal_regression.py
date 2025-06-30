import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MyPolynomialRegressionGD:
    def __init__(self, ):
        """
        Khởi tạo mô hình 
        """
        pass

    def polynomial_features(self, x, degree):
        """
        Biến đổi đầu vào x thành vector đặc trưng bậc đa thức
        """
        N = x.shape[0]
        X_poly = np.ones((N, degree + 1))
        for d in range(1, degree + 1):
            X_poly[:, d] = x.flatten() ** d
        return X_poly

    def fit(self, X, y):
        """
        Huấn luyện mô hình bằng Gradient Descent
        """
        XTX_inv = np.linalg.inv(np.dot(X.T, X))
        return np.dot(np.dot(XTX_inv, X.T), y.reshape(-1, 1))


    def mean_squared_error(self, y_true, y_pred):
        """
        Tính MSE giữa nhãn thật và nhãn dự đoán
        """
        return np.mean((y_true - y_pred) ** 2)

    def train_and_evaluate(self, x_train, y_train, x_test, y_test, degree_list):
        results = []
        predictions = {}

        for degree in degree_list:
            X_train_poly = self.polynomial_features(x_train, degree)
            X_test_poly = self.polynomial_features(x_test, degree)

            theta = self.fit(X_train_poly, y_train)

            # print(theta)

            y_train_pred = np.dot(X_train_poly, theta)
            y_test_pred = np.dot(X_test_poly, theta)

            E_in = self.mean_squared_error(y_train, y_train_pred)
            E_out = self.mean_squared_error(y_test, y_test_pred)

            results.append((degree, E_in, E_out))
            predictions[degree] = (y_train_pred, y_test_pred)

        return results, predictions
