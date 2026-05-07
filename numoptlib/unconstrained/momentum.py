import numpy as np
from numoptlib.utils.line_search import line_search
from numoptlib._base import OptimizeResult, CountedFunction

def momentum_gradient_descent(f, grad_f, x0, beta = 0.9, tol = 1e-6, max_iter = 250):
    f = CountedFunction(f)

    x = x0.copy()
    history = [x.copy()]
    v = np.zeros_like(x0)
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            return OptimizeResult(
                x=x,
                fun=f(x),
                grad=grad,
                nit=k,
                nfev=f.nfev,
                success=True,
                message="Gradient norm below tolerance",
                history=history
            )      
        
        v = beta * v + grad
        p = -v
        
        if grad @ p >= 0:
            p = -grad
            v = grad
            
        alpha = line_search(f, grad_f, x, p)
        x = x + alpha*p
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

