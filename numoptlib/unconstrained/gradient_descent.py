import numpy as np
from numoptlib.utils.line_search import line_search
from numoptlib._base import OptimizeResult, CountedFunction

def gradient_descent(f, grad_f, x0, tol = 1e-6, max_iter = 250):
    '''
    Minimizes a function f using batch gradient descent with a strong Wolfe line search to choose the step size. 
    
    At each iteration, the descent direction is the steepest descent, using the negative gradient. The step size alpha is chosen by a strong Wolfe line search, satisfying the Armijo sufficient decrease condition and the curvature condition. Convergence is declared when the L2 norm of the gradient falls below the tol. If max_iter is reached before convergence, the best iterate is returned with success=False. 
    
    Parameters
    ----------
    f : callable
        Objective function to minimize. 
        Signature: f(x) -> float, where x is a 1D numpy array
    grad_f : callable
        Gradient of the objective function. 
        Signature: grad_f(x) -> ndarray, returning a 1D array of the same shape as x.
    x0 : array_like
        Starting point for optimization. Copied internally so the original is not modified.
    tol : float, optional [DEFAULT 1e-6]
        Convergence tolerance. Optimization stops when ||grad_f(x)|| < tol.
    max_iter : int, optional [DEFAULT 250]
        Maximum number of iterations. If reached before convergence, function returns with success=False
        
    Returns
    -------
    OptimizeResult 
        An object with the following fields
        
        x : ndarray
            Solution/best iterate found
        fun : float
            Objective value at x
        grad : ndarray
            Gradient at x
        nit : int
            Number of iteration performed
        nfev : int
            Number of function evaluations, including those made by line_search
        success : bool
            True if gradient norm fell below tol, False if max_iter was reached
        message : str
            Human-readable description of the stopping condition
        history : list of ndarray
            Iterates x_0, x_1, ... , x_k. history[0] is x0, history[-1] is the final iterate. Has length nit + 1
            
    Notes
    -----
    Gradient descent has O(1/k) convergence on strongly convex functions, where k is the iteration count. Convergence degrades significantly on ill-conditioned problems; the convergence rate depends on the condition number kappa of the Hessian as (kappa - 1)/(kappa + 1). For ill-conditioned problems, consider using Newton or BFGS.
    
    The line search uses strong Wolfe conditions with c1 = 1e-4 (Armijo) and c2 = 0.8 (curvature). See line_search in numoptlib.utils for details. 
    
    Examples
    --------
    >>> import numpy as np
    >>> from numoptlib.unconstrained.gradient_descent import gradient_descent
    >>>
    >>> def f(x):
    ...     return x[0]**2 + 10*x[1]**2
    >>> def grad_f(x):
    ...     return np.array([2*x[0], 20*x[1]])
    >>>
    >>> x0 = np.array([1.0, 1.0])
    >>> result = gradient_descent(f, grad_f, x0)
    >>> result.success
    True
    >>> result.x
    array([0., 0.])
    
    References
    ----------
    Nocedal, J. & Wright, S. (2006). Numerical Optimization (2nd ed.),
    Chapter 3. Springer.
    
    '''
    # initialize function and parameters
    f = CountedFunction(f)

    x = x0.copy()
    history = [x.copy()]
    
    # begin optimization iteration
    for k in range(max_iter):
        grad = grad_f(x)
        
        # check L2 norm of the gradient
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
        
        # define step direction and size
        p = -grad
        alpha = line_search(f, grad_f, x, p)
        
        # update parameters
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
