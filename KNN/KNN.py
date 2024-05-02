import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calculate the distance
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # print(f'distances: {distances}')
        # print(f'argsort_distances: {np.argsort(distances)}')

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # print(f'k_indices: {k_indices}')
        # print(f'k_nearest_labels: {k_nearest_labels}')

        # get the label by majority vote
        most_common = Counter(k_nearest_labels).most_common()
        # print(f'most_common: {most_common}')
        # return k_nearest_labels
        return most_common[0][0]
    
    def _accuracy(self, prediction, target):
        return (np.sum(prediction == target) / len(target)) * 100


