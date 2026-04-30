import numpy as np
from numoptlib.utils.line_search import line_search
from numoptlib._base import OptimizeResult, CountedFunction

def finite_diff_hessian(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            H[i,j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4*eps**2)
    return H

def newton(f, grad_f, x0, hessian_f, tol = 1e-6, max_iter = 250):
    f = CountedFunction(f)
    
    x = x0.copy()
    history = [x.copy()]
    
    
    if hessian_f is None:
        hessian_f = lambda x: finite_diff_hessian(f, x)

    for k in range(max_iter):
        grad = grad_f(x)    
        
        if np.linalg.norm(grad) < tol:
            return OptimizeResult(
                x,
                fun = f(x),
                grad = grad, 
                nit = k, 
                nfev = f.nfev, 
                success = True, 
                message = "Gradient norm below tolerance",
                history = history)
        
        hess = hessian_f(x)
        
        if np.any(np.linalg.eigvals(hess) <= 0):
            p = -grad
        
        else: 
            p = np.linalg.solve(hess, -grad) 
        
            
        alpha = line_search(f, grad_f, x, p)
        x = x + alpha * p
        history.append(x.copy())
        
    return OptimizeResult(
            x,
            fun = f(x),
            grad = grad, 
            nit = max_iter, 
            nfev = f.nfev, 
            success = False, 
            message = "Maximum iterations reached",
            history = history)
