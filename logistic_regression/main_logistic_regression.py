import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classification = classifier.predict(X_test)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


accuracy = "{:.2f}".format(accuracy(y_test, classification) * 100)

print(f"Classification accuracy: {accuracy}% ")
