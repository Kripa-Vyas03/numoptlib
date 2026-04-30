import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.newton import newton

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])

def rosenbrock_hessian(x):
    return np.array([
        [-400*(x[1] - x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],
        [-400*x[0],                                  200.0    ]
    ])

x0 = np.array([0, 0])
result = newton(rosenbrock, rosenbrock_grad, x0, rosenbrock_hessian, max_iter = 500)

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

ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 30), cmap="gray", alpha=0.6)
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
plt.savefig(f"examples/rosenbrock_newton_path_start{x0}.png", dpi=150, bbox_inches="tight")
plt.show()