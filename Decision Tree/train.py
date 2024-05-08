from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTree()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(predictions, y_test)
print(acc)