import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.adam import adam_gradient_descent

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

# -- comparing starts --

for x0 in ([[-1, 1], [0, 0], [-1.1, 1]]):
    x0 = np.array(x0)
    result = adam_gradient_descent(rosenbrock, rosenbrock_grad, x0, alpha = 0.1, max_iter = 2000)
    print(f"-- Start: {x0} --")
    print(f"Solution: {result.x}")
    print(f"Function value: {result.fun:.6f}")
    print(f"Converged: {result.success}")
    print(f"Iterations: {result.nit}")
    
    # --- unpack history ---
    history = np.array(result.history)    # shape (nit, 2)
    xs = history[:, 0]
    ys = history[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- plot 1: contour + path ---
    x_grid = np.linspace(-2, 2, 400)
    y_grid = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 100*(Y - X**2)**2 + (1 - X)**2
    
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap="gray", alpha=0.3)
    ax1.plot(xs, ys, "o-", color="steelblue", markersize=3, linewidth=1, label="path")
    ax1.plot(xs[0], ys[0], "go", markersize=8, label="start")
    ax1.plot(xs[-1], ys[-1], "r*", markersize=12, label="end")
    ax1.plot(1, 1, "kx", markersize=10, label="minimum")
    ax1.set_title("Optimization path")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    
    # # --- plot 2: loss curve ---
    loss_values = [rosenbrock(x) for x in result.history]
    ax2.plot(loss_values, color="steelblue", linewidth=1.5)
    ax2.set_yscale("log")
    ax2.set_title("Loss vs iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)  [log scale]")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"rosenbrock_adam_path_start{x0}.png", dpi=150, bbox_inches="tight")
    plt.show()
    


# -- comparing alphas --
x0 = np.array([-1.5, 1.3])

result1 = adam_gradient_descent(rosenbrock, rosenbrock_grad, x0, alpha = 0.01,  max_iter=10000)
result2 = adam_gradient_descent(rosenbrock, rosenbrock_grad, x0, alpha = 0.1, max_iter=10000)
result3 = adam_gradient_descent(rosenbrock, rosenbrock_grad, x0, alpha = 0.5, max_iter=10000)

results_history = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- plot 1: contour + path ---
x_grid = np.linspace(-2, 2, 400)
y_grid = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = 100*(Y - X**2)**2 + (1 - X)**2
ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap="gray", alpha=0.15)
ax1.set_title("Optimization path with varying step sizes")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

for r, alpha, c in zip([result1, result2, result3], [0.01, 0.05, 0.1], ['tab:blue', "tab:red", "tab:orange"]):
    print(f"-- Alpha: {alpha} --")
    print(f"Solution: {r.x}")
    print(f"Function value: {r.fun:.6f}")
    print(f"Converged: {r.success}")
    print(f"Iterations: {r.nit}")

    
    results_history.append(np.array(r.history))
    history = np.array(r.history)
    
    ax1.plot(history[:, 0], history[:, 1], color = c, markersize=0.5, linewidth=1.5)
    
    loss_values = [rosenbrock(x) for x in r.history]
    ax2.plot(loss_values, color=c, linewidth=1.5, label = fr"$\alpha$ = {alpha}, {r.nit} iterations")

ax1.scatter(x0[0], x0[1], marker = "*", color = "tab:green", s = 150, label = f"Start {x0}", zorder = 5)
ax1.scatter(1, 1, marker = "*", color = "black", s = 150, label = "Minimum [1 1]", zorder = 5)    
ax1.legend()

ax2.set_yscale("log")
ax2.set_title("Loss vs iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("f(x)  [log scale]")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("rosenbrock_adam_comparingalphas.png", dpi=150, bbox_inches="tight")
plt.show()