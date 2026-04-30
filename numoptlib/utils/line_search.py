import numpy as np

def line_search(f, grad_f, x, p, alpha = 1, alpha_max = 10, c1 = 1e-4, c2 = 0.8, max_iter = 100, 
                zoom_max_iter = 20):
    '''
    STRONG WOLFE conditions 
    
    f = objective function
    grad_f = gradient
    p = descent direction
    x = current position
    alpha = initial step size
    c1 = Armijo sufficient decrease constant (0, 1)
    c2 = curvature condition constant (c1, 1)
    
    
    '''
    assert c1 < c2
    
    def phi(alpha):
        return f(x + alpha * p)

    def dphi(alpha):
        return grad_f(x + alpha * p) @ p
    
    
    def zoom(alpha_low, alpha_high, zoom_max_iter = zoom_max_iter):
        
        def cubic_min(a, phi_a, dphi_a, b, phi_b, dphi_b, eps=1e-10):
            if abs(b - a) < eps:
                return 0.5 * (a + b)    # bracket collapsed, fall back to bisection
            
            d1 = dphi_a + dphi_b - 3*(phi_b - phi_a)/(b - a)
            d2_sq = d1**2 - dphi_a*dphi_b
        
            if d2_sq < 0:
                return 0.5*(a + b)
        
            d2 = np.sqrt(d2_sq)
            alpha = b - (b - a)*((dphi_b + d2 - d1)/(dphi_b - dphi_a + 2*d2))
        
            if not (min(a, b) < alpha < max(a, b)):
                return 0.5*(a + b)
        
            return alpha
    
        phi0 = phi(0)
        dphi0 = dphi(0)
    
        phi_low = phi(alpha_low)
    
        for _ in range(zoom_max_iter):
            if abs(alpha_high - alpha_low) < 1e-10:  # add this
                return alpha_low
    
            phi_high = phi(alpha_high)
            dphi_low = dphi(alpha_low)
            dphi_high = dphi(alpha_high)
            alpha = cubic_min(alpha_low, phi_low, dphi_low,
                              alpha_high, phi_high, dphi_high)
                
            phi_alpha = phi(alpha)
    
            # --- Armijo failure OR not improving relative to left endpoint ---
            if (phi_alpha > phi0 + c1 * alpha * dphi0) or (phi_alpha >= phi_low):
                alpha_high = alpha
    
            else:
                dphi_alpha = dphi(alpha)
    
                # Wolfe curvature condition satisfied
                if abs(dphi_alpha) <= -c2 * dphi0:
                    return alpha
    
                # IMPORTANT: only use derivative sign vs initial slowpe
                if dphi_alpha * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
    
                alpha_low = alpha
                phi_low = phi_alpha
                    
        return alpha

    alpha_prev = 0
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    for iter_num in range(1, max_iter):
        phi_alpha = phi(alpha)
        dphi_alpha = dphi(alpha)
        
        if (phi_alpha > phi_0 + c1*alpha*dphi_0) or (phi_alpha >= phi(alpha_prev) and iter_num > 1):
            return zoom(alpha_prev, alpha)
        
        if abs(dphi_alpha) <= -c2 * dphi_0:
            return alpha
        
        if dphi_alpha >= 0:
            return zoom(alpha, alpha_prev)
        
        alpha_prev = alpha
        alpha *= 2
        
        if alpha > alpha_max:
            return zoom(alpha_prev, alpha_max)
            
    return alpha


# --- testing ---
# quadratic
# def f(x):
#     return x[0]**2 + 10*x[1]**2

# def grad_f(x):
#     return np.array([2*x[0], 20*x[1]])
# x = np.array([1.0, 1.0])
# p = -grad_f(x)

# alpha = line_search(f, grad_f, x, p, alpha = 3)
# print("quadratic", alpha)


# # # rosenbrock
# def f(x):
#     return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# def grad_f(x):
#     return np.array([
#         -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
#         200*(x[1] - x[0]**2)
#     ])
# x = np.array([-1.0, 1.0])
# p = -grad_f(x)

# alpha = line_search(f, grad_f, x, p, alpha = 3)
# print("rosenbrock", alpha)

