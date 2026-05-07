# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.gradient_descent import gradient_descent
from numoptlib.unconstrained.adams import adams_gradient_descent

# --- well conditioned quadratic ---
np.random.seed(42)
eigvals = np.linspace(1.0, 2.0, 2)
Q, _ = np.linalg.qr(np.random.randn(2, 2))
A = Q @ np.diag(eigvals) @ Q.T
b = np.zeros(2)
WC_quad_x_star = np.linalg.solve(A, b)

def WC_quad_f(x, A=A, b=b):
    return 0.5 * x @ A @ x - b @ x

def WC_quad_grad(x, A=A, b=b):
    return A @ x - b

# --- ill conditioned quadratic ---
eigvals_ill = np.linspace(1.0, 50.0, 2)   # condition number of 50
Q, _ = np.linalg.qr(np.random.randn(2, 2))
A = Q @ np.diag(eigvals) @ Q.T
b = np.zeros(2)
IC_quad_x_star = np.linalg.solve(A, b)

def IC_quad_f(x, A=A, b=b):
    return 0.5 * x @ A @ x - b @ x

def IC_quad_grad(x, A=A, b=b):
    return A @ x - b


# -- rosenbrock --
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

rosenbrock_x_star = np.array([1, 1])


# -- himmelblau --
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

starts = [
    np.array([0.0, 0.5]),
    np.array([-1.0, 3.0]),
    np.array([2.0, -2.0]),
    np.array([-3.0, -1.0]),
]

himmelblau_x_star = np.array([
                [3.0,       2.0],
                [-2.805,    3.131],
                [3.584,     -1.848],
                [-3.779,    -3.283]
                ])

colors = [("steelblue", "deepskyblue", "navy"),
          ("tab:red", "lightsalmon", "orangered"), 
          ("mediumseagreen", "limegreen", "darkgreen"), 
          ("mediumpurple", "mediumorchid", "rebeccapurple")]

def plot_quadratic_optim(name, f, grad_f, x_star, x0 = np.array([5.0, 3.0]), alpha = 0.5, max_iter = 500):
    # --- run solvers ---
    result_gd  = gradient_descent(f, grad_f, x0, max_iter = max_iter)
    result_adams = adams_gradient_descent(f, grad_f, x0, alpha = alpha, max_iter = max_iter)
    
    history_gd  = np.array(result_gd.history)
    history_adams = np.array(result_adams.history)
    
    # --- contour grid ---
    margin = 0.5
    all_x = np.concatenate([history_gd[:, 0], history_adams[:, 0], [WC_quad_x_star[0]]])
    all_y = np.concatenate([history_gd[:, 1], history_adams[:, 1], [WC_quad_x_star[1]]])
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    
    x_grid = np.linspace(x_min, x_max, 400)
    y_grid = np.linspace(y_min, y_max, 400)
    X, Y  = np.meshgrid(x_grid, y_grid)
    Z     = np.array([[WC_quad_f(np.array([xi, yi])) for xi in x_grid] for yi in y_grid])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- shared contour plot function ---
    def plot_path(ax, history, color, label, title):
        ax.contourf(X, Y, Z, levels=30, cmap="Blues", alpha=0.4)
        ax.contour(X, Y, Z, levels=30, colors="steelblue", alpha=0.5, linewidths=0.5)
    
        # optimization path
        ax.plot(history[:, 0], history[:, 1],
                "o-", color=color, markersize=2, linewidth=1, label=label)
    
        # start and end markers
        ax.plot(history[0, 0],  history[0, 1],
                "o", color=color, markersize=9, label="start")
        ax.plot(history[-1, 0], history[-1, 1],
                "*", color=color, markersize=12, label="end")
    
        # minimum
        ax.plot(x_star[0], x_star[1],
                "kx", markersize=12, markeredgewidth=2, label=f"minimum {WC_quad_x_star.round(3)}", zorder=5)
    
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
    
    plot_path(axes[0], history_gd,  color="coral",       
              label="GD path",       title=f"Gradient descent  ({result_gd.nit} iters)")
    plot_path(axes[1], history_adams, color="mediumseagreen",
              label="Adams path", title=f"Adams  ({result_adams.nit} iters)")
    
    plt.suptitle(f"{name.title()} Conditioned Quadratic — optimization paths", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{name.lower()}cond_quadratic_paths.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"Gradient descent  — iters: {result_gd.nit},  nfev: {result_gd.nfev},  converged: {result_gd.success}")
    print(f"Adams  — iters: {result_adams.nit}, nfev: {result_adams.nfev}, converged: {result_adams.success}")
    
plot_quadratic_optim("Well", WC_quad_f, WC_quad_grad, WC_quad_x_star)
plot_quadratic_optim("Ill", IC_quad_f, IC_quad_grad, IC_quad_x_star)


def convergence_curve(f, grad_f, x_star, x0_factor, name, alpha = 0.1, max_iter = 500):
    '''
    DON'T USE ON HIMMELBLAU
    '''

    x0 = x0_factor*np.ones(len(x_star))
    f_star = f(x_star)
    
    result_GD = gradient_descent(f, grad_f, x0, max_iter = max_iter) 
    result_m = adams_gradient_descent(f, grad_f, x0, max_iter = max_iter, alpha = alpha)   
    
    GD_gap = [f(x) - f_star for x in result_GD.history]
    m_gap = [f(x) - f_star for x in result_m.history]
    
    grad_norm_GD = [np.linalg.norm(grad_f(x)) for x in result_GD.history]
    grad_norm_m = [np.linalg.norm(grad_f(x)) for x in result_m.history]

    
    
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    
    if name.lower() == "rosenbrock":
        axs[0].plot(range(result_m.nit + 1), m_gap, color = "tab:blue", label = "Adams")
        axs[1].plot(range(result_m.nit + 1), grad_norm_m, color = "tab:blue", label = "Adams")
        
    else:
        axs[0].plot(range(result_m.nit), m_gap, color = "tab:blue", label = "Adams")
        axs[1].plot(range(result_m.nit), grad_norm_m, color = "tab:blue", label = "Adams")
        
    axs[0].plot(range(result_GD.nit + 1), GD_gap, color = "tab:red", label = "Gradient descent")
    axs[0].set_xlabel("Number of iterations")
    axs[0].set_ylabel("Suboptimality gap")
    axs[0].set_title("Suboptimality gap")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_yscale("log")

    axs[1].plot(range(result_GD.nit + 1), grad_norm_GD, color = "tab:red", label = "Gradient descent")
    axs[1].set_xlabel("Number of iterations")
    axs[1].set_ylabel("Gradient norm")
    axs[1].set_title("Gradient norm")
    axs[1].set_yscale("log")

    fig.suptitle(f"{name.title()} Comparison")
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)
    
    plt.tight_layout()
    plt.savefig(f"convergence_curve_{name.lower()}_Adams_GD.jpg", bbox_inches = "tight")
    plt.show()
    
convergence_curve(WC_quad_f, WC_quad_grad, WC_quad_x_star, 4, "Well-conditioned quadratic", alpha = 0.5)

convergence_curve(IC_quad_f, IC_quad_grad, IC_quad_x_star, 4, "Ill-conditioned quadratic", alpha = 0.5)

convergence_curve(rosenbrock, rosenbrock_grad, rosenbrock_x_star, 4, "Rosenbrock", alpha = 0.1)



# ---- rosenbrock ----
for x0 in ([[0, 0], [-1.1, 1], [-1.5, 1.3]]):
    x0 = np.array(x0)
    result_m = adams_gradient_descent(rosenbrock, rosenbrock_grad, x0, max_iter = 2000, alpha = 0.1)
    history_m = np.array(result_m.history)
    loss_values_m = [rosenbrock(x) for x in result_m.history]

    result_GD = gradient_descent(rosenbrock, rosenbrock_grad, x0, max_iter=2000)
    history_GD = np.array(result_GD.history)
    loss_values_GD = [rosenbrock(x) for x in result_GD.history]
    
    # ----- plotting -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- plot 1: contour + path ---
    x_grid = np.linspace(-2, 2, 400)
    y_grid = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 100*(Y - X**2)**2 + (1 - X)**2
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap="gray", alpha=0.15)
    ax1.set_title("Optimization path")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    ax1.plot(history_GD[:, 0], history_GD[:, 1], "o-", color="tab:blue", markersize=0.5, linewidth=1.5, label = f"Gradient descent, {result_GD.nit} iterations")
    ax1.plot(history_m[:, 0], history_m[:, 1], "o-", color="tab:red",  markersize=0.5, linewidth=1.5, label = f"Adams, {result_m.nit} iterations")

    ax1.scatter(x0[0], x0[1], marker = "*", color = "tab:green", s = 150, label = f"Start {x0}", zorder = 5)
    ax1.scatter(1, 1, marker = "*", color = "black", s = 150, label = "Minimum [1 1]", zorder = 5)    
    handles, labels = ax1.get_legend_handles_labels()
    
    # --- plot 2: iteration and loss ---
    ax2.set_yscale("log")
    ax2.set_title("Loss vs iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)  [log scale]")
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(loss_values_m, color="tab:red", linewidth=1.5)
    ax2.plot(loss_values_GD, color="tab:blue", linewidth=1.5)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)

    plt.tight_layout()
    plt.savefig(f"rosenbrock_Adams_GD_start{x0}.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    
    
# ---- himmelblau ----
x_grid = np.linspace(-5, 5, 400)
y_grid = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 30), colors="gray", alpha=0.3, linewidths=0.5)

for x0, x_star, cols in zip(starts, himmelblau_x_star, colors):
    markercolor, mcolor, GDcolor = cols
    
    plt.scatter(x0[0], x0[1], marker = "o", facecolors = "white", edgecolors=markercolor,
             s=80, label=f"Start {x0}")
    plt.scatter(x_star[0], x_star[1], marker = "*", color = markercolor, s = 100, label = "Minimum")

    result_m = adams_gradient_descent(himmelblau, himmelblau_grad, x0, max_iter = 2000, alpha = 0.1)
    history_m = np.array(result_m.history)
    loss_values_m = [rosenbrock(x) for x in result_m.history]

    result_GD = gradient_descent(himmelblau, himmelblau_grad, x0, max_iter=2000)
    history_GD = np.array(result_GD.history)
    loss_values_GD = [rosenbrock(x) for x in result_GD.history]


    plt.plot(history_m[:, 0], history_m[:, 1], color=mcolor, linewidth=1, label=f"Adams, {result_m.nit} iterations")
    plt.plot(history_GD[:, 0], history_GD[:, 1], "--", color=GDcolor, linewidth=1, label=f"Gradient Descent, {result_GD.nit} iterations")
    
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fontsize = 8, ncol = 4)
plt.title("Himmelblau Optimization Paths")
plt.savefig("himmelblau_Adams_GD.png", bbox_inches = "tight")
plt.show()
    