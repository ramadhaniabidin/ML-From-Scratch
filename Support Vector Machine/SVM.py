import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lamda=0.01, epoch=100):
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.epoch = epoch
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epoch):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lamda * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lamda * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        predictions = np.dot(X, self.w) - self.b
        return np.sign(predictions)



        