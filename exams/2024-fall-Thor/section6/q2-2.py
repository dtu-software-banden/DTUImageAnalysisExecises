import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

with open("exams/2024-fall-Thor/section6/D_Cubes.txt", 'r') as file:
    class1_samples = [int(line.strip()) for line in file if line.strip()]

with open("exams/2024-fall-Thor/section6/E_Cubes.txt", 'r') as file:
    class2_samples = [int(line.strip()) for line in file if line.strip()]

mu1, sigma1 = np.mean(class1_samples), np.std(class1_samples)
mu2, sigma2 = np.mean(class2_samples), np.std(class2_samples)

# Define the difference between the class-conditional PDFs
def pdf_difference(x):
    return norm.pdf(x, mu1, sigma1) - norm.pdf(x, mu2, sigma2)

# Initial guess near the midpoint
initial_guess = (mu1 + mu2) / 2

# Solve for the point where the two PDFs intersect
threshold = fsolve(pdf_difference, x0=initial_guess)[0]
print(f"Bayes threshold (equal priors): {threshold:.2f}")
