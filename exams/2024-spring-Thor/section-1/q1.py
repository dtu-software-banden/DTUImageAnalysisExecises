import numpy as np

# Load the data (excluding the first row with column names starting with '%')
wine_data = np.loadtxt("wine-data.txt", comments="%")

# Separate features and labels (assuming last column is Producer label)
features = wine_data[:, :-1]  # all columns except last
labels = wine_data[:, -1]     # last column

# Step 1: Compute mean of each feature column
mean = np.mean(features, axis=0)

# Step 2: Subtract mean from each feature
mean_centered = features - mean

# Step 3: Compute the range (max - min) for each column
feature_range = np.max(mean_centered, axis=0) - np.min(mean_centered, axis=0)

# Step 4: Scale the features by dividing by the range
scaled_features = mean_centered / feature_range

# If needed, you can recombine with the labels
processed_data = np.hstack((scaled_features, labels.reshape(-1, 1)))

# Compute the covariance matrix
cov_matrix = np.cov(scaled_features, rowvar=False)


# Compute eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print(eigenvalues,eigenvectors)