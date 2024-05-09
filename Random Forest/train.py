from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

breast_cancer_dataset = datasets.load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def accuracy(target, pred):
    return np.sum(target == pred) / len(target)

model = RandomForest()
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = accuracy(y_test, pred)
print(acc)