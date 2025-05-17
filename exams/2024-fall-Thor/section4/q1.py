import numpy as np
import matplotlib.pyplot as plt

# Define the cost function and its gradient
def cost_function(x):
    x1, x2 = x
    return 7 * x1**2 + x1 * x2 + 3 * x2**2

def gradient(x):
    x1, x2 = x
    dc_dx1 = 14 * x1 + x2
    dc_dx2 = x1 + 6 * x2
    return np.array([dc_dx1, dc_dx2])

# === PARAMETERS ===
start_position = np.array([-4.0, -4.0])  # Change starting point here
step_size = 0.1                          # Learning rate
max_iters = 200                          # Max number of iterations
tolerance = 1e-6                         # Stopping criterion

# === Gradient Descent Execution ===
x = start_position.copy()
path = [x.copy()]

for _ in range(max_iters):
    grad = gradient(x)
    x_new = x - step_size * grad
    path.append(x_new.copy())
    if np.linalg.norm(x_new - x) < tolerance:
        break
    x = x_new

path = np.array(path)

# === Plotting ===
x1_vals = np.linspace(-6, 3, 400)
x2_vals = np.linspace(-6, 3, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = cost_function([X1, X2])

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'ro--', label='Gradient descent path')
plt.plot(path[0, 0], path[0, 1], 'go', label=f'Start {tuple(start_position)}')
plt.plot(path[-1, 0], path[-1, 1], 'bo', label='End')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gradient Descent Optimization Path')
plt.legend()
plt.grid(True)
plt.show()
