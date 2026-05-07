import numpy as np
from numoptlib._base import OptimizeResult, CountedFunction

def adams_gradient_descent(f, grad_f, x0, alpha = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, tol = 1e-6, max_iter = 250):
    f = CountedFunction(f)

    x = x0.copy()
    history = [x.copy()]
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    
    for k in range(1, max_iter+1):
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
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
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
        
        