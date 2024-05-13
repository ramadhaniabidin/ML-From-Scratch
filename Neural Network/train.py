from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork

def encode_label(y):
    n_labels = (np.unique(y)).shape[0]
    label = np.zeros((y.shape[0], n_labels))
    label[np.arange(n_labels) == y] = 1
    return label

data = datasets.load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

y_test_encoded = encode_label(y_test)
y_train_encoded = encode_label(y_train)


model = NeuralNetwork(X_train.shape[1], n_output=y_train_encoded.shape[1], n_hidden=3, learning_rate=0.001)
model.fit(X_train, y_train_encoded)

encoded_pred = model.predict(X_test)
encoded_pred[encoded_pred > 0.5] = 1
encoded_pred[encoded_pred <= 0.5] = 0
acc = np.sum(encoded_pred == y_test_encoded) / y_test_encoded.size
print(encoded_pred)
print(acc)
