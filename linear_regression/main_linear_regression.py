import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from linear_regression import LinearRegression

X, y = datasets.make_regression(n_samples=10000, n_features=1, noise=8, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression(lr=0.001, num_iterations=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# print(predictions, end="\n\n")
# print(y_test)


def mse_loss(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


print(mse_loss(y_test, predictions))
