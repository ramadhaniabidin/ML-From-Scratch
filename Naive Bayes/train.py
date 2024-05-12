from sklearn.model_selection import train_test_split
from sklearn import datasets
from NaiveBayes import NaiveBayes
import numpy as np

def accuracy(target, pred):
    return (np.sum(target == pred) / len(target)) * 100

# X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

iris_dataset = datasets.load_iris()
X, y = iris_dataset.data, iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = NaiveBayes()
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy(y_test, pred)

print(f'Model accuracy = {acc} %')