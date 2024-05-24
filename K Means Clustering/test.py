from sklearn import datasets
from KMeans import KMeans
import numpy as np

X, y = datasets.make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=1234)
print(f'The shape of X: {X.shape}')

clusters = len(np.unique(y))
print(f'Number of clusters: {clusters}')


kmeans = KMeans(k=clusters, max_iters=100, plot_steps=True)
y_pred = kmeans.predict(X)

# kmeans.plot()








