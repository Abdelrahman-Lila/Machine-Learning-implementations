import numpy as np


class Perceptron:

    def __init__(self, lr=0.01, num_iterations=1000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.activation = self._unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update + x_i
                self.bias += update

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear)
        return y_predicted

    def _unit_step(self, x):
        return np.where(x >= 0, 1, 0)
