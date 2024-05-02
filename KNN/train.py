import numpy as np
from KNN import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris_dateset = datasets.load_iris()
X, y = iris_dateset.data, iris_dateset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()

print(f'y_train: {y_train}')

classifier = KNN(k=7)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
accuracy = classifier._accuracy(predictions, y_test)

# print(f'k_nearest_labels = {k_nearest_labels}')
print(f'predictions: {predictions}')
print(f'target: {y_test}')
print(f'accuracy: {accuracy}')