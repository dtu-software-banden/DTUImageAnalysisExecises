import numpy as np

def cost_function(x1, x2):
    return x1**2 - x1*x2 + 3*x2**2 + x1**3

def gradient(x1, x2):
    # Partial derivatives of the cost function
    dC_dx1 = 2*x1 - x2 + 3*x1**2
    dC_dx2 = -x1 + 6*x2
    return np.array([dC_dx1, dC_dx2])

def gradient_descent(initial_point, learning_rate=0.07, num_iterations=1000):
    x = np.array(initial_point, dtype=float)
    history = [x.copy()]
    
    for i in range(num_iterations):
        print(i, cost_function(x[0],x[1]))
        grad = gradient(x[0], x[1])
        x -= learning_rate * grad
        history.append(x.copy())
    
    return x, history

# Example usage
initial = [4.0, 3.0]
minimum, path = gradient_descent(initial, learning_rate=0.07, num_iterations=10)
print("Minimum at:", minimum)
