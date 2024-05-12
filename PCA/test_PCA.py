from matplotlib import pyplot as plt
from sklearn import datasets
from PCA import PCA
import numpy as np

dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

pca = PCA(n_components=2)
pca.fit(X)
X_projected = pca.transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", len(np.unique(y))))
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()
plt.show()

