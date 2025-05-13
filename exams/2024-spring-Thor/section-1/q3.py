import numpy as np

# Load the data
wine_data = np.loadtxt("wine-data.txt", comments="%")
features = wine_data[:, :-1]

# Step 1: Get the alcohol values (first column)
alcohol_column = features[:, 0]

# Step 2: Compute mean and range of alcohol
mean_alcohol = np.mean(alcohol_column)
range_alcohol = np.max(alcohol_column - mean_alcohol) - np.min(alcohol_column - mean_alcohol)

# Step 3: Normalize alcohol for first wine
first_wine_alcohol = alcohol_column[0]
normalized_alcohol = (first_wine_alcohol - mean_alcohol) / range_alcohol

print("Normalized alcohol value of the first wine:", normalized_alcohol)
