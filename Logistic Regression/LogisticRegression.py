import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, epoch=100):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return (1/(1 + np.exp(-x)))

    def fit(self, X, y):
        # first initialization of weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epoch):
            v = np.dot(X, self.weights) + self.bias
            sigmoid_y = self.sigmoid(v)

            dw = (1/n_samples) * np.dot(X.T, (sigmoid_y - y))
            db = (1/n_samples) * np.sum(sigmoid_y - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        v = np.dot(X, self.weights) + self.bias
        sigmoid_y = self.sigmoid(v)
        predictions = [0 if y <= 0.5 else 1 for y in sigmoid_y]
        return predictions

