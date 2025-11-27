import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, num_iterations=1000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y = y.reshape(-1)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for iteration in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            if iteration % 100 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"loss: {loss}")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
