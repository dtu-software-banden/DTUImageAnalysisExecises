import numpy as np

# Load the data
wine_data = np.loadtxt("wine-data.txt", comments="%")
features = wine_data[:, :-1]
labels = wine_data[:, -1]

# Step 1: Mean-center
mean = np.mean(features, axis=0)
mean_centered = features - mean

# Step 2: Range-scale (max - min per feature)
feature_range = np.max(mean_centered, axis=0) - np.min(mean_centered, axis=0)
scaled_features = mean_centered / feature_range

# Step 3: Covariance matrix
cov_matrix = np.cov(scaled_features, rowvar=False)

average_cov_value = np.mean(cov_matrix)
print("Average value of elements in the covariance matrix:", average_cov_value)

# Step 4: Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Assuming you already have:
# eigenvalues = np.linalg.eig(cov_matrix)[0]

# Step 1: Sort eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]

# Step 2: Compute total variance
total_variance = np.sum(sorted_eigenvalues)

# Step 3: Compute variance from the first 5 components
top5_variance = np.sum(sorted_eigenvalues[:5])

# Step 4: Compute percentage
percent_explained = (top5_variance / total_variance) * 100

print("Percentage of total variance explained by first 5 principal components:", percent_explained)


# Step 5: First principal component
first_pc = eigenvectors[:, np.argmax(eigenvalues)]

# Step 6: Project samples onto the first principal component
projections = scaled_features @ first_pc

# Step 7: Compute average projections for Producer 1 and 2
avg_proj_1 = np.mean(projections[labels == 1])
avg_proj_2 = np.mean(projections[labels == 2])

# Step 8: Compute the difference
difference = abs(avg_proj_1 - avg_proj_2)

print("Difference between average projections (Producer 1 - Producer 2):", difference)

# Step 9: Compute the range of projections on PC1
min_proj = np.min(projections)
max_proj = np.max(projections)
range_proj = max_proj - min_proj

print("Range of projected coordinates on PC1:", range_proj)

