import numpy as np
from numoptlib.unconstrained.adams import adams_gradient_descent

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + \
           (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2
def beale_grad(x):
    a = 1.5 - x[0] + x[0]*x[1]
    b = 2.25 - x[0] + x[0]*x[1]**2
    c = 2.625 - x[0] + x[0]*x[1]**3
    dfdx = 2*a*(-1 + x[1]) + 2*b*(-1 + x[1]**2) + 2*c*(-1 + x[1]**3)
    dfdy = 2*a*x[0] + 2*b*2*x[0]*x[1] + 2*c*3*x[0]*x[1]**2
    return np.array([dfdx, dfdy])

functions = {
    "rosenbrock": (rosenbrock, rosenbrock_grad, np.array([-1.0, 1.0])),
    "himmelblau": (himmelblau, himmelblau_grad, np.array([0.0, 0.0])),
    "beale":      (beale,      beale_grad,      np.array([1.0, 1.0])),
}

alphas = [0.001, 0.005, 0.01, 0.05, 0.1]

for name, (f, grad_f, x0) in functions.items():
    print(f"\n--- {name} ---")
    print(f"{'alpha':<10} {'converged':<12} {'iters':<8} {'f(x*)':<15} {'x*'}")
    for alpha in alphas:
        result = adams_gradient_descent(f, grad_f, x0, alpha=alpha, max_iter=10000)
        print(f"{alpha:<10} {str(result.success):<12} {result.nit:<8} {result.fun:<15.8f} {result.x.round(4)}")
        
        

A = np.array([[4.0, 1.0],
              [1.0, 3.0]])
b = np.array([1.0, 2.0])

def f(x):
    return 0.5 * x @ A @ x - b @ x
def grad_f(x):
    return A @ x - b

x_star = np.linalg.solve(A, b)
x0 = np.zeros(2)

alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

print(f"x_star = {x_star.round(6)}")
print(f"\n{'alpha':<10} {'converged':<12} {'iters':<8} {'f(x*)':<15} {'x*'}")
for alpha in alphas:
    result = adams_gradient_descent(f, grad_f, x0, alpha=alpha, max_iter=10000)
    print(f"{alpha:<10} {str(result.success):<12} {result.nit:<8} {result.fun:<15.8f} {result.x.round(4)}")
    
    
np.random.seed(42)
eigvals = np.array([1.0, 50.0])
Q, _ = np.linalg.qr(np.random.randn(2, 2))
A_ill = Q @ np.diag(eigvals) @ Q.T
b_ill = np.array([1.0, 2.0])

def f_ill(x):
    return 0.5 * x @ A_ill @ x - b_ill @ x
def grad_f_ill(x):
    return A_ill @ x - b_ill

x_star_ill = np.linalg.solve(A_ill, b_ill)
x0 = np.zeros(2)

print(f"\n--- ill conditioned (kappa=50) ---")
print(f"x_star = {x_star_ill.round(6)}")
print(f"\n{'alpha':<10} {'converged':<12} {'iters':<8} {'f(x*)':<15} {'x*'}")
for alpha in alphas:
    result = adams_gradient_descent(f_ill, grad_f_ill, x0, alpha=alpha, max_iter=10000)
    print(f"{alpha:<10} {str(result.success):<12} {result.nit:<8} {result.fun:<15.8f} {result.x.round(4)}")