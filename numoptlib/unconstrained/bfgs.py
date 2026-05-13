import numpy as np
from numoptlib.utils.line_search import line_search
from numoptlib._base import OptimizeResult, CountedFunction

def bfgs(f, grad_f, x0, tol=1e-6, max_iter=250):
    f = CountedFunction(f)
    n = len(x0)
    I = np.eye(n)
    history = [x0.copy()]

    # --- step 0: initial gradient step to get s, y for H scaling ---
    x = x0.copy()
    grad_0 = grad_f(x)
    p_0 = -grad_0                          # steepest descent for first step
    alpha = line_search(f, grad_f, x, p_0)
    x_new = x + alpha * p_0
    s = x_new - x
    y = grad_f(x_new) - grad_0

    # N&W initial scaling: H_0 = (y^T s / y^T y) * I
    ys = y @ s
    yy = y @ y
    H_scale = (ys / yy) * I if yy > 1e-10 else I

    # --- first BFGS update ---
    if ys > 1e-10:
        rho = 1.0 / ys
        H = (I - rho * np.outer(s, y)) @ H_scale @ (I - rho * np.outer(y, s)) \
            + rho * np.outer(s, s)
    else:
        H = H_scale

    x = x_new
    history.append(x.copy())

    # --- main loop ---
    for k in range(1, max_iter):
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

        p = -H @ grad
        alpha = line_search(f, grad_f, x, p)
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - grad

        ys = y @ s
        if ys > 1e-10:
            rho = 1.0 / ys
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)

        x = x_new
        history.append(x.copy())

    return OptimizeResult(
        x=x,
        fun=f(x),
        grad=grad_f(x),
        nit=max_iter,
        nfev=f.nfev,
        success=False,
        message="Maximum iterations reached",
        history=history
    )