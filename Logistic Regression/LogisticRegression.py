import numpy as np
import matplotlib.pyplot as plt

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
        self.loss = []

        for i in range(self.epoch):
            v = np.dot(X, self.weights) + self.bias
            sigmoid_y = self.sigmoid(v)
            loss = np.mean(-y * np.log(sigmoid_y) - (1 - y) * (np.log(1 - sigmoid_y)))
            self.loss.append(loss)

            dw = (1/n_samples) * np.dot(X.T, (sigmoid_y - y))
            db = (1/n_samples) * np.sum(sigmoid_y - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # if i % 10 == 0:
            # print(f'Iteration: {i}, Loss: {loss}')
        
        plt.plot(np.arange(self.epoch), self.loss)
        plt.xlabel('Number of iteration')
        plt.ylabel('Loss')
        plt.title('Loss on training set')
        plt.show()        

    def predict(self, X):
        v = np.dot(X, self.weights) + self.bias
        sigmoid_y = self.sigmoid(v)
        predictions = [0 if y <= 0.5 else 1 for y in sigmoid_y]
        return predictions
    
    

