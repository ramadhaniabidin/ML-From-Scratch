import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.001, epoch = 100):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epoch):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2/n_samples) * (np.dot(X.T, (y_pred - y)))
            db = (2/n_samples) * (np.sum(y_pred - y))

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred