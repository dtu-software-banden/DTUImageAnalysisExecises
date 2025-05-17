import numpy as np
from sklearn.datasets import load_breast_cancer

from utils.pca_utils import compute_dyn_pca, compute_pca

breast = load_breast_cancer()
x = breast.data
target = breast.target

x = x - np.mean(x,axis=0)
x = x / np.std(x,axis=0)

pca,projections = compute_dyn_pca(x)

print("Features:",len(pca.explained_variance_ratio_))

fpc = projections[:,0]
predictions = np.where((fpc < 0),1,0)
print("Predict no cancer",predictions.sum())

print("Accuracy:",np.mean(predictions == target))

negative_pred = np.mean(fpc[fpc < 0])
positive_pred = np.mean(fpc[fpc > 0])

print("negtive:",negative_pred,"positive:",positive_pred)

pc_1_2 = projections[:,:2]

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Example setup — replace these with your actual arrays
points = pc_1_2
labels = target

# Create a scatter plot with two classes
plt.figure(figsize=(8, 6))

# Class 0
plt.scatter(points[labels == 0, 0], points[labels == 0, 1],
            color='red', label='Class 0', alpha=0.7)

# Class 1
plt.scatter(points[labels == 1, 0], points[labels == 1, 1],
            color='blue', label='Class 1', alpha=0.7)

# Labeling the plot
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Scatter Plot Categorized by Binary Labels')
plt.legend()
plt.show()
