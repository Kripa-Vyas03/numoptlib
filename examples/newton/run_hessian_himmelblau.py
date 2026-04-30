# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.newton import newton


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

starts = [
    np.array([0.0, 0.0]),
    np.array([-1.0, 3.0]),
    np.array([2.0, -2.0]),
    np.array([-3.0, -1.0]),
]
colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

x_grid = np.linspace(-5, 5, 400)
y_grid = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.contourf(X, Y, Z, levels=np.logspace(0, 3, 30), cmap="gray", alpha=0.3)
ax1.contour(X, Y, Z, levels=np.logspace(0, 3, 30), colors="gray", alpha=0.5, linewidths=0.5)

for x0, color in zip(starts, colors):
    result = newton(himmelblau, himmelblau_grad, x0)
    history = np.array(result.history)
    ax1.plot(history[:, 0], history[:, 1], "o-", color=color,
             markersize=2, linewidth=1, label=f"start {x0}")
    ax1.plot(history[0, 0], history[0, 1], "o", color=color, markersize=8)
    ax1.plot(history[-1, 0], history[-1, 1], "*", color=color, markersize=12)

ax1.set_title("Himmelblau — gradient descent paths")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(fontsize=8)

for x0, color in zip(starts, colors):
    result = newton(himmelblau, himmelblau_grad, x0, hessian_f = None)
    print(f"Start: {x0}")
    print(f"Solution: {result.x}")
    print(f"Function value: {result.fun:.6f}")
    print(f"Converged: {result.success}")
    print(f"Iterations: {result.nit}")
    losses = [himmelblau(x) for x in result.history]
    ax2.plot(losses, color=color, linewidth=1.5, label=f"start {x0}")

ax2.set_yscale("log")
ax2.set_title("Loss vs iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("f(x)  [log scale]")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/himmelblau_newton_paths.png", dpi=150, bbox_inches="tight")
plt.show()