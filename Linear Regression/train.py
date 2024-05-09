from LinearRegression import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, c="b", marker="o", s=30)
# plt.title("Dataset for testing")
# plt.show()

regression = LinearRegression(learning_rate=0.01)
regression.fit(X_train, y_train)
predictions = regression.predict(X_test)
y_predict_line = regression.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, s=10)
m2 = plt.scatter(X_test, y_test, s=10)
plt.plot(X, y_predict_line, c="black")
plt.show()

def error(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

err = error(y_test, predictions)
print(f'Error: {err}')

