import numpy as np
from numoptlib._base import OptimizeResult, CountedFunction

def adam_gradient_descent(f, grad_f, x0, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, tol = 1e-6, max_iter = 250):
    '''
    Minimizes a function f using the Adam (Adaptive Moment Estimation) algorithm.
    
    Adam maintains per-parameter adaptive step sizes by tracking exponential moving averages 
    of both the gradient (first moment) and the squared gradient (second moment). Both 
    estimates are bias-corrected for their zero-initialization. The update rule at iteration 
    k is: 
        m_k = beta1 * m_{k-1} + (1 - beta1) * grad_f(x_k)
        v_k = beta2 * v_{k-1} + (1 - beta2) * grad_f(x_k)^2
        m_hat = m_k / (1 - beta1^k)
        v_hat = v_k / (1 - beta2^k)
        x_{k+1} = x_k - alpha * m_hat / (sqrt(v_hat) + epsilon)
        
    Unlike the other solvers in this library, Adam doesn't use line search. The per-
    parameter adaptive scaling plays the same role, adjusting the step size in each step. 
    Convergence is declared when the L2 norm of the gradient falls below the tol. If 
    max_iter is reached before convergence, the best iterate is returned with success=False. 
    
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
    alpha : float, optional [DEFAULT 0.001]
        Global learning rate. Scales the overall step magnitude. For the example functions 
        in this library (Rosenbrock, Himmelblau, Beale), alpha = 0.1 is recommended. The 
        default 0.001 follows Kingma & Ba (2015). 
        See examples/adam/rosenbrock_adam_comparingalphas.png for a plot of how the alpha 
        affects the optimization for the Rosenbrock function.
    beta1 : float, optional [DEFAULT 0.9]
        Exponential beta rate for the first moment. Controls the inlfuence of past gradients 
        on the current step direction. Must be in (0, 1).
    beta2 : float, optional [DEFAULT 0.999]
        Exponential decay rate for the second moment. Controls the per-parameter step size 
        adaptation. Must be in (0, 1).
    epsilon : float, optional [DEFAULT 1e-8]
        Small constant added to the denominator to prevent division by zero and bounds the 
        step size when the gradients are near zero.
    tol : float, optional [DEFAULT 1e-6]
        Convergence tolerance. Optimization stops when ||grad_f(x)|| < tol.
    max_iter : int, optional [DEFAULT 250]
        Maximum number of iterations. If reached before convergence, function returns with 
        success=False.
        Adam usually requires more iterations that Newton or BFGS on smooth functions.
        
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
            Iterates x_0, x_1, ... , x_k. history[0] is x0, history[-1] is the final 
            iterate. Has length nit + 1
            
    Notes
    -----
    Adam is designed for large-scale stochastic problems where the gradient is estimated 
    from minibatches and line search is too computationally expensive per iteration. On 
    small, smooth, deterministic problems like those in this library's benchmark/example 
    suite, Adam typically uses more iterations than Newton or BFGS because it doesn't use 
    second order curvature information. The advantage of Adam is robustness, it converges 
    correctly across many problems without tweaking parameters beyond the alpha. 
    
    The default hyperparameters follow Kingma & Ba (2015) and work well across a broad range 
    of problems that Adam is suited for. For the benchmark/example functions, alpha = 0.1 
    with max_iter = 2000-5000 is recommended.
    
    Adam's per-parameter scalling makes it naturally robust to ill-conditioned problems -- 
    the second moment estimate adapts the effective step size in each direction 
    independently, compensating for differences in curvature without using the Hessian. 
    
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
    >>> result = gradient_descent(f, grad_f, x0, alpha = 0.01, max_iter = 2500)
    >>> result.success
    True
    >>> result.x
    array([-9.64872017e-09, -9.65107561e-09])
    >>> result.nit
    354
    
    References
    ----------
    Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic
    Optimization. ICLR 2015. https://arxiv.org/abs/1412.6980    
    '''
    # initialize function and parameters
    f = CountedFunction(f)

    x = x0.copy()
    history = [x.copy()]
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    
    # begin iteration
    for k in range(1, max_iter+1):
        grad = grad_f(x)
        
        # check L2 norm of gradient
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
        
        # calculate first and second moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        
        # update parameters
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
          
