import numpy as np
import matplotlib.pyplot as plt

# Define your 5 points (x, y)
points = np.array([
    [7, 13],
    [9, 10],
    [6, 10],
    [6, 8],
    [3, 6]
])

# Define the Hough space range
thetas = np.deg2rad(np.linspace(0, 180, 360))  # angles from -90° to +90°
rhos_range = []

# Compute Hough curves
plt.figure(figsize=(10, 6))
for x, y in points:
    rhos = x * np.cos(thetas) + y * np.sin(thetas)
    rhos_range.append(rhos)
    plt.plot(np.rad2deg(thetas), rhos, label=f"Point ({x},{y})")

plt.title("Hough Space (θ vs ρ)")
plt.xlabel("θ (degrees)")
plt.ylabel("ρ (pixels)")
plt.grid(True)
plt.legend()
plt.show()
