import numpy as np
from numoptlib.utils.line_search import line_search
from numoptlib._base import OptimizeResult, CountedFunction

def momentum_gradient_descent(f, grad_f, x0, beta = 0.9, tol = 1e-6, max_iter = 250):
    '''
    Minimizes a function f using momentum gradient descent with a strong Wolfe line search to choose the step size.
    
    The method differs from standard gradient descent by including a momentum term that accumulates past gradient 
    information to damp oscillations and accelerate progress in low-curvature directions. At each iteration, a 
    search direction is found using a weighted average of previous gradients. If the momentum direction stops
    being a descent direction, the function falls back to the steepest descent. The step size alpha is chosen by a 
    strong Wolfe line search, satisfying the Armijo sufficient decrease condition and the curvature condition. 
    
    Convergence is declared when the L2 norm of the gradient falls below the tol. If max_iter is reached before 
    convergence, the best iterate is returned with success=False.
    
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
    beta : float, optional [DEFAULT 0.9]
        Momentum parameter in [0, 1) controlling the influence of previous gradients. 
        Larger values increase smoothing and acceleration effects.
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
    Momentum gradient descent accelerates convergence over vanilla gradient descent by maintaining a velocity 
    vector v that accumulates a weighted sum of past gradients. The update rule is
        v_k = beta * v_{k-1} + grad_f(x_k)
        x_{k+1} = x_k - alpha * v_k
    where alpha is chosen by a strong Wolfe line search and beta controls the contribution of past gradients. 
    Higher beta values mean more momentum, past gradients influence the current step more strongly. 
    
    The key advantage over vanilla gradient descent is faster traversal of regions where the gradient consistently 
    points in the same direction. On ill-conditioned problems, using momentum can reduce zigzagging behaviour, but 
    it won't eliminate it.
    
    If the momentum direction is not a descent direction at x_k, the velocity is resent to the current gradient 
    and the direction reverts to the steepest descent. This prevents the line search from failing.
    
    The convergence rate depends on beta. For a strongly convex quadratic with condition number kappa, the 
    theoretically optimal beta is 
        beta_optimal = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2
    For well-conditioned problems with a small kappa, the optimal beta is close to zero. A high beta would cause 
    unnecessary oscillation. The default beta is suited to ill-conditioned problems. 
    
    Momentum gradient descent is typically used for medium to large scale smooth optimization problem. It can work well for objective functions that have long valleys or ill-conditioned Hessians. However, momentum gradient descent has a weak theoretical convergence guarantee. For problems where convergence rate matters, use BFGS. Additionally, it is less suited for objective functions that are not smooth or with discontinuous gradients.
    
    Examples
    --------
    >>> import numpy as np
    >>> from numoptlib.unconstrained.momentum import momentum
    >>>
    >>> def f(x):
    ...     return x[0]**2 + 10*x[1]**2
    >>> def grad_f(x):
    ...     return np.array([2*x[0], 20*x[1]])
    >>>
    >>> x0 = np.array([1.0, 1.0])
    >>> result = momentum_gradient_descent(f, grad_f, x0)
    >>> result.success
    True
    >>> result.x
    array( [1.53662117e-07, 2.55522901e-08])
    
    References
    ----------
    Nocedal, J. & Wright, S. (2006). Numerical Optimization (2nd ed.),
    Chapter 3. Springer.
    '''
    
    # initialize function and parameters
    f = CountedFunction(f)

    x = x0.copy()
    history = [x.copy()]
    v = np.zeros_like(x0)
    
    # begin optimization iteration    
    for k in range(max_iter):
        grad = grad_f(x)
        
        # check L2 norm of the gradient 
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
        
        # define step direction using the velocity vector
        v = beta * v + grad
        p = -v
        
        # if the step direction is not a descent direction, use steepest descent
        if grad @ p >= 0:
            p = -grad
            v = grad
            
        # find step size
        alpha = line_search(f, grad_f, x, p)
        
        # update parameters
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
