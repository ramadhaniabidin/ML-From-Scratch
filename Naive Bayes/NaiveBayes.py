import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        pass


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # calculate mean, variance, adn prior probability for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        # calculate the posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self.classes[np.argmax(posteriors)]
    
    def pdf(self, class_index, x):
        mean = self.mean[class_index]
        var = self.var[class_index]
        numerator = np.exp((-((x - mean) ** 2) / (2 * var)))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

