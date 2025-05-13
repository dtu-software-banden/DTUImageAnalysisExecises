import numpy as np

def gradient_descent(func, grad_func, x_init, step_size, max_iter=100, tol=0.2):
    x = x_init
    for i in range(max_iter):
        if func(*x) < tol:
            return x, i
        grad = grad_func(*x)
        x -= step_size * grad
    return x, max_iter
