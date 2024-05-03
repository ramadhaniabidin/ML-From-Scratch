import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

breast_cancer_dataset = datasets.load_breast_cancer()
X = np.array(breast_cancer_dataset['data'])
y = np.array(breast_cancer_dataset['target'])

print(breast_cancer_dataset['DESCR'])

# print(f'X: {X[:,3]}')
# print(f'y: {y[:3]}')
y_values = range(3) 
colorMap = plt.get_cmap('viridis')
fig, axs = plt.subplots(len(y_values), 1, figsize=(8, 8))

for i, col in enumerate(y_values):
    axs[i].scatter(X[:, 0], X[:, col], c=y, cmap=cmap, edgecolors='k', s=20)
    axs[i].set_xlabel('radius (mean)')
    axs[i].set_ylabel(f'Column {col}')
    axs[i].set_title(f'Scatter plot of radius (mean) vs Column {col}')

plt.tight_layout()
plt.show()

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.xlabel('radius (mean)')
# plt.ylabel('texture (mean)')
# plt.show()


# plt.figure()
# plt.scatter(X[:,0], X[:,2], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.xlabel('radius (mean)')
# plt.ylabel('texture (mean)')
# plt.show()

# print(breast_cancer_dataset['DESCR'])
# print(f'X: {X[:5]}')
# print(f'y: {y[:5]}')

# print(f'X dimension: {X.shape}')