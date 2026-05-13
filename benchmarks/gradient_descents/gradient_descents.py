import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.gradient_descent import gradient_descent
from numoptlib.unconstrained.momentum import momentum_gradient_descent
from numoptlib.unconstrained.newton import newton
from numoptlib.unconstrained.adam import adam_gradient_descent

# --- rosenbrock ---
def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    return np.array([
        [-400*(x[1] - x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],
        [-400*x[0],                                  200.0    ]
    ])

x_star = np.ones(2)
x0_vals     = [np.array([-1.0, 1.0]), 
               np.array([-1.1, 1.1]),
               np.array([0, 0])]
max_iter = 500

for x0 in x0_vals:
    print(f"==== Start: {x0} ====")

    # --- run all solvers ---
    results = {
        "Gradient Descent": gradient_descent(
            rosenbrock, rosenbrock_grad, x0, max_iter = max_iter),
        "Momentum (β=0.9)": momentum_gradient_descent(
            rosenbrock, rosenbrock_grad, x0, beta=0.9, max_iter = max_iter),
        "Newton":           newton(
            rosenbrock, rosenbrock_grad, x0, hessian_f=rosenbrock_hess, max_iter = max_iter),
        "Adam (α=0.1)":     adam_gradient_descent(
            rosenbrock, rosenbrock_grad, x0, alpha=0.1, max_iter=5000),
    }
    
    # --- summary table ---
    print(f"\n{'Method':<20} {'Converged':<12} {'Iterations':<12} {'Func Evals':<12} {'f(x*)':<15} {'||x - x*||'}")
    print("-" * 85)
    for name, result in results.items():
        gap = np.linalg.norm(result.x - x_star)
        print(f"{name:<20} {str(result.success):<12} {result.nit:<12} {result.nfev:<12} {result.fun:<15.8f} {gap:.2e}")
    
    # --- suboptimality gap plot ---
    f_star = rosenbrock(x_star)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]
    
    # plot 1 — suboptimality gap vs iterations
    for (name, result), color in zip(results.items(), colors):
        gap = [rosenbrock(x) - f_star for x in result.history]
        axes[0].plot(gap, color=color, linewidth=1.5, label=name)
    
    axes[0].set_yscale("log")
    axes[0].set_title("Suboptimality gap vs iterations")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("f(x) - f(x*)  [log scale]")
    axes[0].grid(True, alpha=0.3)
    
    # plot 2 — suboptimality gap vs function evaluations
    for (name, result), color in zip(results.items(), colors):
        gap    = [rosenbrock(x) - f_star for x in result.history]
        nfevs  = np.linspace(0, result.nfev, len(gap))
        axes[1].plot(nfevs, gap, color=color, linewidth=1.5, label=name)
    
    axes[1].set_yscale("log")
    axes[1].set_title("Suboptimality gap vs function evaluations")
    axes[1].set_xlabel("Function evaluations")
    axes[1].set_ylabel("f(x) - f(x*)  [log scale]")
    axes[1].grid(True, alpha=0.3)    
    
    # plot 3 -- optimization pathways
    x_grid = np.linspace(-2, 2, 400)
    y_grid = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 100*(Y - X**2)**2 + (1 - X)**2
    
    axes[2].contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap="gray", alpha=0.15)
    
    for (name, result), color in zip(results.items(), colors):
        history = np.array(result.history)    
        axes[2].plot(history[:, 0], history[:, 1], color=color, markersize=3, linewidth=2, label=f"{name} -- {result.nit} iterations")
    
    axes[2].scatter(x0[0], x0[1], marker = "*", s=150, label=f"Start {x0}", edgecolor = "black", facecolor = "gold", zorder = 4)
    axes[2].scatter(x_star[0], x_star[1], marker = "*", s=150, label="Minimum", color = "tab:red", zorder = 4)
    axes[2].set_title("Optimization paths")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_ylim(-1, 2)
    
    handles, labels = axes[2].get_legend_handles_labels()
    
    
    plt.suptitle(f"Rosenbrock benchmark (start {x0}) — method comparison", fontsize=13)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)

    plt.tight_layout()
    plt.savefig(f"rosenbrock_benchmark_start{x0}.png", dpi=150, bbox_inches="tight")

    plt.show()
