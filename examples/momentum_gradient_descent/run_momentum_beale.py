import numpy as np
import matplotlib.pyplot as plt
from numoptlib.unconstrained.momentum import momentum_gradient_descent

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + \
           (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2

def beale_grad(x):
    a = 1.5 - x[0] + x[0]*x[1]
    b = 2.25 - x[0] + x[0]*x[1]**2
    c = 2.625 - x[0] + x[0]*x[1]**3
    dfdx = 2*a*(-1 + x[1]) + 2*b*(-1 + x[1]**2) + 2*c*(-1 + x[1]**3)
    dfdy = 2*a*x[0] + 2*b*2*x[0]*x[1] + 2*c*3*x[0]*x[1]**2
    return np.array([dfdx, dfdy])

starts = [
    np.array([0.0, 0.0]),
    np.array([-1.0, 1.5]),
    np.array([3.0, -1.0]),
    np.array([1.0, 2.0]),
]
colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

x_grid = np.linspace(-4.5, 4.5, 400)
y_grid = np.linspace(-4.5, 4.5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (1.5 - X + X*Y)**2 + (2.25 - X + X*Y**2)**2 + (2.625 - X + X*Y**3)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.contourf(X, Y, Z, levels=np.logspace(0, 4, 30), cmap="gray", alpha=0.3)
ax1.contour(X, Y, Z, levels=np.logspace(0, 4, 30), colors="gray", alpha=0.5, linewidths=0.5)
ax1.plot(3, 0.5, "kx", markersize=12, markeredgewidth=2, label="Minimum (3, 0.5)", zorder=5)

for x0, color in zip(starts, colors):
    result = momentum_gradient_descent(beale, beale_grad, x0, max_iter = 2000)

    print(f"Start: {x0}")
    print(f"Solution: {result.x}")
    print(f"Function value: {result.fun:.6f}")
    print(f"Converged: {result.success}")
    print(f"Iterations: {result.nit}")
    
    history = np.array(result.history)
    losses = [beale(x) for x in result.history]

    ax1.plot(history[:, 0], history[:, 1], "o-", color=color,
             markersize=2, linewidth=1, label=f"Start {x0}, {result.nit} iterations")
    ax1.plot(history[0, 0], history[0, 1], "o", color=color, markersize=8)
    ax1.plot(history[-1, 0], history[-1, 1], "*", color=color, markersize=12)
    handles, labels = ax1.get_legend_handles_labels()
    
    ax2.plot(losses, color=color, linewidth=1.5)


ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-4.5, 4.5)
ax1.set_title("Beale — momentum gradient descent paths")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.set_yscale("log")
ax2.set_title("Loss vs iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("f(x)  [log scale]")
ax2.grid(True, alpha=0.3)

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)

plt.tight_layout()
plt.savefig("beale_gradient_descent_paths.png", dpi=150, bbox_inches="tight")
plt.show()