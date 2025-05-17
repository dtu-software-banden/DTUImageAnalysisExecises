import numpy as np

with open("exams/2024-fall-Thor/section6/D_Cubes.txt", 'r') as file:
    class1_samples = [int(line.strip()) for line in file if line.strip()]

with open("exams/2024-fall-Thor/section6/E_Cubes.txt", 'r') as file:
    class2_samples = [int(line.strip()) for line in file if line.strip()]

mu1, sigma1 = np.mean(class1_samples), np.std(class1_samples)
mu2, sigma2 = np.mean(class2_samples), np.std(class2_samples)

from numpy import log

# Coefficients of the quadratic equation
a = (1 / (2 * sigma1**2)) - (1 / (2 * sigma2**2))
b = (mu2 / (sigma2**2)) - (mu1 / (sigma1**2))
c = ((mu1**2) / (2 * sigma1**2)) - ((mu2**2) / (2 * sigma2**2)) + log(sigma2 / sigma1)

# Solve quadratic: ax^2 + bx + c = 0
discriminant = b**2 - 4 * a * c
if discriminant < 0:
    thresholds = []  # No real intersection
else:
    sqrt_disc = np.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    thresholds = [x1, x2]

# Select the threshold between the two means
threshold = min(thresholds, key=lambda x: abs(x - (mu1 + mu2) / 2))
print(f"Optimal threshold: {threshold}")
