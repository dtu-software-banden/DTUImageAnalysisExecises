import pandas as pd

from utils.pca_utils import compute_pca, normalize_std


df = pd.read_csv('exams/2022-fall-Thor/section6/car_data.txt',sep='\s+')

norm = normalize_std(df)


# print(norm)


pca,projections = compute_pca(norm,n_components=8)

print(pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1])

print(projections)

from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Create a DataFrame of PCA projections
df_pca = pd.DataFrame(projections, columns=[f"PC{i+1}" for i in range(projections.shape[1])])

# Step 3: Find which PCs are most influenced by your 3 features of interest
# Get the loadings: pca.components_ has shape (n_components, n_features)
feature_names = norm.columns.tolist()
target_features = ['wheel-base', 'length', 'width']
target_indices = [feature_names.index(f) for f in target_features]

# For each of the 3 original features, find the PC with the highest loading
import numpy as np

# Absolute loadings of each PC for each of the 3 features
loadings = np.abs(pca.components_[:, target_indices])  # shape: (n_components, 3)

# Find top 3 PCs based on overall contribution from the 3 features
combined_importance = np.sum(loadings, axis=1)
top_pc_indices = np.argsort(combined_importance)[-3:][::-1]  # get indices of top 3 PCs

# Step 4: Make the pairplot of those top PCs
selected_pcs = [f"PC{i+1}" for i in top_pc_indices]  # convert to PC1-style names
df_pairplot = df_pca[selected_pcs]

# Plot
sns.pairplot(df_pairplot)
plt.suptitle("Pairplot of PCA Space Most Influenced by Wheel-base, Length, Width", y=1.02)
plt.show()