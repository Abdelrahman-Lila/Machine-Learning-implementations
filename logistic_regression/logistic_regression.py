import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, num_iterations=1000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y = y.reshape(-1)
        n_samples, n_features = X.shape[0]
        self.bias = 0
        self.weights = np.zeros(n_features)

        for iteration in range(self.num_iterations):
            linear_val = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-linear_val))

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_val = np.dot(X, self.weights) + self.bias
        y_predictions = 1 / (1 + np.exp(-linear_val))
        classified = [1 if prediction > 0.5 else 0 for prediction in y_predictions]
        return classified
