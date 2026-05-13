import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.gradient_descent import gradient_descent
from numoptlib.unconstrained.newton import newton
from numoptlib.unconstrained.bfgs import bfgs
import time as time


# --- test functions ---
colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

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

def make_quadratic(n, condition_number):
    np.random.seed(42)
    eigvals = np.linspace(1.0, condition_number, n)
    Q, _   = np.linalg.qr(np.random.randn(n, n))
    A      = Q @ np.diag(eigvals) @ Q.T
    b      = np.random.randn(n)
    f      = lambda x: 0.5 * x @ A @ x - b @ x
    grad_f = lambda x: A @ x - b
    hess_f = lambda x: A
    x_star = np.linalg.solve(A, b)
    return f, grad_f, hess_f, x_star

def timed_run(solver, *args, n_runs=5, **kwargs):
    """Run solver n_runs times and return result + median wall time."""
    times = []
    result = None
    for _ in range(n_runs):
        start  = time.perf_counter()
        result = solver(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return result, np.median(times)

def run_benchmark(prob_name, f, grad_f, hess_f, x0, x_star, max_iter = 500):
    f_star = f(x_star)
    solvers = {
        "Gradient Descent": lambda: gradient_descent(f, grad_f, x0, max_iter = max_iter),
        "Newton":           lambda: newton(f, grad_f, x0, hessian_f=hess_f, max_iter=max_iter),
        "BFGS":             lambda: bfgs(f, grad_f, x0, max_iter=max_iter),
    }

    print(f"\n{'=' * 90}")
    print(f"  {prob_name}")
    print(f" start: {x0}")
    print(f"{'=' * 90}")    
    print(f"{'Method':<20} {'Conv':<6} {'Iters':<8} {'nfev':<8} "
          f"{'Time (ms)':<12} {'||x-x*||':<14} {'f(x)-f(x*)'}")
    print(f"{'-' * 90}")

    results = {}
    for name, solver in solvers.items():
        result, t = timed_run(solver)
        gap_x = np.linalg.norm(result.x - x_star)
        gap_f = abs(result.fun - f_star)
        print(f"{name:<20} {str(result.success):<6} {result.nit:<8} {result.nfev:<8} "
              f"{t*1000:<12.2f} {gap_x:<14.2e} {gap_f:.2e}")
        results[name] = result
        
    print(f"{'=' * 90}")

    return results


# benchmark rosenbrock
rosenbrock_starts = [np.array([-1.0, 1.0]), 
                     np.array([-1.1, 1.1]),
                     np.array([0, 0])]

for x0 in rosenbrock_starts:
    x_star = np.ones(2)
    f_star = rosenbrock(x_star)
    results = run_benchmark("Rosenbrock", rosenbrock, rosenbrock_grad, rosenbrock_hess, x0, x_star)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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

# -- benchmark quadratics --
quadratic_vals = [(2, 2, "Well-conditioned"), 
                  (2, 50, "Ill-conditioned"), 
                  (20, 10, "High-dimensional")]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, val in enumerate(quadratic_vals):
    n, kappa, spec = val
    x0 = 3*np.ones(n)
    problem_name = f"{spec} quadratic (n={n}, κ={kappa})"
    func, grad, hess, x_star = make_quadratic(n, kappa)
    f_star = func(x_star)
    results = run_benchmark(problem_name, func, grad, hess, x0, x_star)
    
    # plot 1 — suboptimality gap vs iterations
    for (name, result), color in zip(results.items(), colors):
        gap = [rosenbrock(x) - f_star for x in result.history]
        axes[0, i].plot(gap, color=color, linewidth=1.5, label=name)
    
    axes[0, i].set_yscale("log")
    axes[0, i].set_title(f"{problem_name}")
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].set_xlabel("Iteration")

    
    # plot 2 — suboptimality gap vs function evaluations
    for (name, result), color in zip(results.items(), colors):
        gap    = [rosenbrock(x) - f_star for x in result.history]
        nfevs  = np.linspace(0, result.nfev, len(gap))
        axes[1, i].plot(nfevs, gap, color=color, linewidth=1.5, label=name)
    
    axes[1, i].set_yscale("log")
    axes[1, i].grid(True, alpha=0.3)  
    axes[1, i].set_xlabel("Function evaluations")

axes[0, 0].set_ylabel("f(x) - f(x*)  [log scale]")
axes[1, 0].set_ylabel("f(x) - f(x*)  [log scale]")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 3)
fig.suptitle("Suboptimality gap", fontsize=15)

plt.tight_layout()
plt.savefig("quadratic_benchmark", bbox_inches = "tight")
plt.show()

# --- scaling benchmark: nfev vs problem dimension ---

print(f"\n{'=' * 60}")
print("  nfev scaling vs problem dimension")
print(f"{'=' * 60}")
print(f"{'n':<6} {'GD nfev':<12} {'Newton nfev':<14} {'BFGS nfev'}")
print(f"{'-' * 60}")

dims = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70]
nfev_arr = np.zeros((3, len(dims)))
for i, n in enumerate(dims):
    f_, g_, h_, xs = make_quadratic(n, condition_number=10)
    x0 = np.zeros(n)
    r_gd  = gradient_descent(f_, g_, x0)
    r_nt  = newton(f_, g_, x0)            # finite diff hessian
    r_bfgs = bfgs(f_, g_, x0)
    nfev_arr[:, i] = [r_gd.nfev, r_nt.nfev, r_bfgs.nfev]
    print(f"{n:<6} {r_gd.nfev:<12} {r_nt.nfev:<14} {r_bfgs.nfev}")
    
fig, axes = plt.subplots(1, 2, figsize = (8, 4))  
fig.suptitle("Function evaluation scaling with dimension")
fig.supxlabel("Dimensions")
fig.supylabel("Function evaluations")

axes[0].plot(dims, nfev_arr[0], "o-", label="Gradient descent", color = colors[0])
axes[0].plot(dims, nfev_arr[1], "o-", label="Newton", color = colors[1])
axes[0].plot(dims, nfev_arr[2], "o-", label="BFGS", color = colors[2])
handles, labels = axes[0].get_legend_handles_labels()

axes[1].plot(dims, nfev_arr[0], "o-", label="Gradient descent", color = colors[0])
axes[1].plot(dims, nfev_arr[2], "o-", label="BFGS", color = colors[2])

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 3)
plt.tight_layout()
plt.savefig("nfev_scaling.png", bbox_inches = "tight")
plt.show()
