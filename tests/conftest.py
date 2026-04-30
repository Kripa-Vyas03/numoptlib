import numpy as np
import pytest

@pytest.fixture
def rosenbrock():
    def f(x):
        return sum(100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1))
    
    def grad(x):
        g = np.zeros_like(x)
        
        for i in range(len(x)-1):
            g[i]   += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
            g[i+1] +=  200*(x[i+1] - x[i]**2)
        return g
    
    return f, grad


@pytest.fixture
def quadratic():
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n)
    A = A.T @ A + np.eye(n)     # positive definite
    b = np.random.randn(n)
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    def grad(x):
        return A @ x - b
    x_star = np.linalg.solve(A, b)   # known exact solution
    return f, grad, x_star