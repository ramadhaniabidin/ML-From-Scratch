import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

breast_cancer_dataset = datasets.load_breast_cancer()
X, y = breast_cancer_dataset.data, breast_cancer_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(y)

model = LogisticRegression(learning_rate=0.01, epoch=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_test)
print(y_pred)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

print(accuracy(y_pred, y_test))