import numpy as np

def line_search(f, grad_f, x, p, alpha = 1, alpha_max = 10, c1 = 1e-4, c2 = 0.8, max_iter = 100, 
                zoom_max_iter = 20):
    '''
    Performs a line search that satisfies the Strong Wolfe conditions. 
    
    This searches for a step length (alpha) along a descent direction (p) such that the following conditions hold:
        1. Armijo condition for sufficient descent
        2. Curvature condition
    The algorithm follows the standard Strong Wolfe line search strategy outlined in Nocedal & Wright, using interval reginement via a zoom phase and cubic interpolation
        
    Parameters
    ----------
    f : callable
        Objective function to minimize.
        Signature: f(x) -> float, where x is a 1D numpy array
    grad_f : callable
        Gradient of the objective function.
        Signature: grad_f(x) -> ndarray, returning a 1D array of the same shape as x.
    x : array_like
        Current point in the optimization process.
    p : array_like
        Search direction, satisfying ``grad_f(x) @ p < 0``
    alpha : float, optional [DEFAULT 1]
        Initial trial step length
    alpha_max : float, optional [DEFAULT 10]
        Maximum allowable step length before using zoom
    c1 : float, optional [DEFAULT 1e-4]
        Armijo condition constant, must satisfy 0 < c1 < c2 < 1
    c2 : float, optional [DEFAULT 0.8]
        Curvative condition constant 
    max_iter : int, optional [DEFAULT 100]
        Maximum number of outer line search iterations
    zoom_max_iter : int, optional [DEFAULT 20]
        Maximum number of iterations in the zoom phase
    
    Returns
    -------
    float
        Step length satisfying the Strong Wolfe conditions, or the best available approximation
        
    Notes
    -----
    The Strong Wolfe conditions are:

    Armijo condition:
        f(x + alpha p) <= f(x) + c1 * alpha * grad_f(x)^T p

    Curvature condition:
        |grad_f(x + alpha p)^T p| <= c2 * |grad_f(x)^T p|

    The zoom phase uses safeguarded cubic interpolation to refine the
    interval containing an acceptable step length.

    References
    ----------
    Nocedal, J., & Wright, S. J. (2006).
    *Numerical Optimization* (2nd ed.).
    Springer.        
    '''
    # Ensure Wolfe constants are valid
    assert 0 < c1
    assert c2 < 1
    assert c1 < c2
    
    def phi(alpha):
        '''
        Restrict the multivariate objective function to the search line 
        Computes phi(alpha) = f(x + alpha * p)
        '''
        return f(x + alpha * p)

    def dphi(alpha):
        '''
        Directional derivative of phi along p
        Computes phi'(alpha) = grad_f(x + alpha * p)^T p
        '''
        return grad_f(x + alpha * p) @ p
    
    
    def zoom(alpha_low, alpha_high, zoom_max_iter = zoom_max_iter):
        '''
        Refine an interval containing a step length satisfying the Strong Wolfe conditions. 
        
        Parameters
        ----------
        alpha_low : float
            Lower bound of the interval
        alpha_high : float
            Upper bound of the interval
        zoom_max_iter : int 
            Maximum number of zoom iterations
            
        Returns
        -------
        float
            Step length satisfying the Strong Wolfe conditions or the best approximations found.
        '''
        
        def cubic_min(a, phi_a, dphi_a, b, phi_b, dphi_b, eps=1e-10):
            '''
            Compute a safeguarded cubic interpolation minimizer. 
            If the interpolation becomes unstable or invalid, the midpoint of the interval is used. 
            '''
            # avoid division by nearly zero interval lengths, use midpoint
            if abs(b - a) < eps:
                return 0.5 * (a + b)
            
            # coefficients for cubic interpolation
            d1 = dphi_a + dphi_b - 3*(phi_b - phi_a)/(b - a)
            d2_sq = d1**2 - dphi_a*dphi_b
        
            # if interpolation becomes complex, use midpoint
            if d2_sq < 0:
                return 0.5*(a + b)
        
            d2 = np.sqrt(d2_sq)
            
            # denominator used in interpolation
            denom = dphi_b - dphi_a + 2*d2  
            
            # guard against numerical instability, use midpoint
            if abs(denom) < eps:
                return 0.5*(a + b)
        
            # cubic minimizer
            alpha = b - (b - a)*((dphi_b + d2 - d1)/denom)
        
            # ensure alpha lies inside the interval
            if not (min(a, b) < alpha < max(a, b)):
                return 0.5*(a + b)
            
            # safeguard against values too close to the endpoints
            margin = 0.1 * abs(b - a)
            alpha = np.clip(alpha, min(a, b) + margin, max(a, b) - margin)
        
            return alpha
        
        # initial function and derivative values
        phi0 = phi(0)
        dphi0 = dphi(0)
    
        phi_low = phi(alpha_low)
    
        for _ in range(zoom_max_iter):
            # stop if interval becomes extremely small
            if abs(alpha_high - alpha_low) < 1e-10:  
                return alpha_low
            
            # evaulate endpoints
            phi_high = phi(alpha_high)
            dphi_low = dphi(alpha_low)
            dphi_high = dphi(alpha_high)
            
            # use cubic interpolation to estimate next alpha
            alpha = cubic_min(alpha_low, phi_low, dphi_low,
                              alpha_high, phi_high, dphi_high)
            phi_alpha = phi(alpha)
    
            # check Armijo condition and sufficient improvement
            if (phi_alpha > phi0 + c1 * alpha * dphi0) or (phi_alpha >= phi_low):
                alpha_high = alpha
    
            else:
                dphi_alpha = dphi(alpha)
    
                # check Strong Wolfe curvature condition
                if abs(dphi_alpha) <= -c2 * dphi0:
                    return alpha
    
                # Update interval bounds depending on the derivative sign
                if dphi_alpha * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
    
                alpha_low = alpha
                phi_low = phi_alpha
        
        # return best estimate if iteration limit reached
        return alpha
    
    # previous step length, objective, and directional derivative
    alpha_prev = 0
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    # main line search loop
    for iter_num in range(1, max_iter):
        # evaluate current step
        phi_alpha = phi(alpha)
        dphi_alpha = dphi(alpha)
        
        # armijo failure or lack of improvement
        if (phi_alpha > phi_0 + c1*alpha*dphi_0) or (phi_alpha >= phi(alpha_prev) and iter_num > 1):
            return zoom(alpha_prev, alpha)
        
        # strong wolfe curvature condition satisfied
        if abs(dphi_alpha) <= -c2 * dphi_0:
            return alpha
        
        # derivative changed sign -> minimizer bracketed
        if dphi_alpha >= 0:
            return zoom(alpha, alpha_prev)
        
        # expand interval and continue searching
        alpha_prev = alpha
        alpha *= 2
        
        # prevent alpha from growing beyond alpha_max
        if alpha > alpha_max:
            return zoom(alpha_prev, alpha_max)
    
    # return final alpha if max_iter reached
    return alpha



