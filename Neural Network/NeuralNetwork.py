import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_input, n_output, n_hidden=5, learning_rate=0.0001, epoch=1000):
        limit = np.sqrt(3 / float(n_input))
        self.w1 = np.random.uniform(low=-limit, high=limit, size=(n_input, n_hidden))
        self.b1 = np.ones((1, n_hidden), dtype=float)
        self.w2 = np.random.uniform(low=-limit, high=limit, size=(n_hidden, n_output))
        self.b2 = np.ones((1, n_output), dtype=float)
        self.learning_rate = learning_rate
        self.epoch = epoch

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def deriv_sigmoid(self, s):
        return s*(1 - s)
    
    def softmax(self, s):
        output = np.zeros((s.shape), dtype=float)
        for i in range(0, s.shape[0]):
            output[i, :] = np.exp(s[i, :]) / np.sum(np.exp(s[i, :]))
        return output
    
    def fit(self, X, y):
        self.loss_list = []
        for i in range(self.epoch):
            self.Z1 = np.dot(X, self.w1) + self.b1
            self.A1 = self.sigmoid(self.Z1)
            self.Z2 = np.dot(self.A1, self.w2) + self.b2
            self.A2 = self.softmax(self.Z2)

            self.loss = np.mean((-y * np.log(self.A2) - (1 - y) * np.log(1 - self.A2)))
            self.loss_list.append(self.loss)
            print(f'Iteration: {i + 10} Loss: {self.loss:.2f}') if i%10==0 else None

            # Back propagation
            self.dLdZ2 = self.A2 - y
            self.dZ2dw2 = self.A1.T
            self.dZ2db2 = np.ones((1, X.shape[0]), dtype=float)
            self.dZ2dA1 = self.w2.T
            self.dA1dZ1 = self.deriv_sigmoid(self.A1)
            self.dZ1dw1 = X.T
            self.dZ1db1 = np.ones((1, X.shape[0]), dtype=float)


            self.dLdw2 = np.dot(self.dZ2dw2, self.dLdZ2)
            self.dLdb2 = np.dot(self.dZ2db2, self.dLdZ2)

            self.dLdw1 = self.dZ1dw1.dot(((self.dLdZ2).dot(self.dZ2dA1))*self.dA1dZ1)
            self.dLdb1 = self.dZ1db1.dot((self.dLdZ2.dot(self.dZ2dA1))*self.dA1dZ1)

            # Update parameters
            self.w1 -= self.learning_rate * self.dLdw1
            self.b1 -= self.learning_rate * self.dLdb1
            self.w2 -= self.learning_rate * self.dLdw2
            self.b2 -= self.learning_rate * self.dLdb2

        # plot the loss graph
        plt.plot(np.arange(self.epoch), self.loss_list)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Loss on training set')
        plt.show()


    def predict(self, X):
        self.Z1 = np.dot(X, self.w1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        output = self.softmax(self.Z2)
        return output