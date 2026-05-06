import numpy as np
from numoptlib.unconstrained.newton import newton
from numoptlib.unconstrained.gradient_descent import gradient_descent
import matplotlib.pyplot as plt

# -- well conditioned quadratic --

# eigvals are 1.0, 2.0
eigvals = np.linspace(1.0, 2.0, 2)
# make random orthogonal matrix
Q, _ = np.linalg.qr(np.random.randn(2, 2))
A = Q @ np.diag(eigvals) @ Q.T   
b = np.array([1.0, 2.0])
b = np.zeros(2)
WC_quad_x_star = np.array(np.linalg.solve(A, b))

def WC_quad_f(x, A = A, b = b):
    return 0.5 * x @ A @ x - b @ x

def WC_quad_grad(x, A = A, b = b):
    return A @ x - b

def WC_quad_hess(x, A = A, b = b):
    return A

# -- ill-conditioned quadratic --

# eigvals are 1.0, 2.0
eigvals = np.logspace(0, np.log10(1e6), 2)
# make random orthogonal matrix
Q, _ = np.linalg.qr(np.random.randn(2, 2))
A = Q @ np.diag(eigvals) @ Q.T   
b = np.array([1.0, 2.0])
IC_quad_x_star = np.array(np.linalg.solve(A, b))

def IC_quad_f(x, A = A, b = b):
    return 0.5 * x @ A @ x - b @ x

def IC_quad_grad(x, A = A, b = b):
    return A @ x - b

def IC_quad_hess(x, A = A, b = b):
    return A


# -- rosenbrock --
def rosenbrock_f(x):
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
    
rosenbrock_x_star = np.array([1, 1])

# -- himmelblau --
def himmelblau_f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

def himmelblau_hess(x):
    return np.array([
                    [12*x[0]**2 + 4*x[1] - 42,  4*(x[0] + x[1])], 
                    [4*x[0] + 4*x[1],           4*x[0] + 12*x[1]**2 - 26]
        ])

himmelblau_x_star = np.array([
                [3.0,       2.0],
                [-2.805,    3.131],
                [3.584,     -1.848],
                [-3.779,    -3.283]
                ])
    

def convergence_curve(f, grad_f, hess_f, x_star, x0_factor, name, max_iter = 250):
    '''
    DON'T USE ON HIMMELBLAU
    '''

    x0 = x0_factor*np.ones(len(x_star))
    f_star = f(x_star)
    
    result_GD = gradient_descent(f, grad_f, x0, max_iter = max_iter) 
    result_newton = newton(f, grad_f, x0, hess_f, max_iter = max_iter)   
    
    GD_gap = [f(x) - f_star for x in result_GD.history]
    newton_gap = [f(x) - f_star for x in result_newton.history]
    
    grad_norm_GD = [np.linalg.norm(grad_f(x)) for x in result_GD.history]
    grad_norm_newton = [np.linalg.norm(grad_f(x)) for x in result_newton.history]
    
    
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    
    
    axs[0].plot(range(result_newton.nit + 1), newton_gap, color = "tab:blue", label = "Newton")
    axs[0].plot(range(result_GD.nit + 1), GD_gap, color = "tab:red", label = "Gradient descent")
    axs[0].set_xlabel("Number of iterations")
    axs[0].set_ylabel("Suboptimality gap")
    axs[0].set_title("Suboptimality gap")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_yscale("log")
    
    

    axs[1].plot(range(result_newton.nit + 1), grad_norm_newton, color = "tab:blue", label = 'Newton')
    axs[1].plot(range(result_GD.nit + 1), grad_norm_GD, color = "tab:red", label = "Gradient descent")
    axs[1].set_xlabel("Number of iterations")
    axs[1].set_ylabel("Gradient norm")
    axs[1].set_title("Gradient norm")
    axs[1].set_yscale("log")

    fig.suptitle(f"{name.title()} Comparison")
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)
    
    plt.tight_layout()
    plt.savefig(f"convergence_curve_{name.lower()}_newton_GD.jpg", bbox_inches = "tight")
    plt.show()
    
convergence_curve(WC_quad_f, WC_quad_grad, WC_quad_hess, WC_quad_x_star, 4, "Well-conditioned quadratic")

convergence_curve(IC_quad_f, IC_quad_grad, IC_quad_hess, IC_quad_x_star, 4, "Ill-conditioned quadratic")

convergence_curve(rosenbrock_f, rosenbrock_grad, rosenbrock_hess, rosenbrock_x_star, 4, "Rosenbrock")

# ---- comparing trajectory paths ----
# -- rosenbrock --
for x0 in ([[0, 0], [-1.1, 1], [-1.5, 1.3]]):
    x0 = np.array(x0)
    result_newton = newton(rosenbrock_f, rosenbrock_grad, x0, hessian_f = rosenbrock_hess, max_iter = 2000)
    history_newton = np.array(result_newton.history)
    loss_values_newton = [rosenbrock_f(x) for x in result_newton.history]

    result_GD = gradient_descent(rosenbrock_f, rosenbrock_grad, x0, max_iter=2000)
    history_GD = np.array(result_GD.history)
    loss_values_GD = [rosenbrock_f(x) for x in result_GD.history]
    
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
    ax1.plot(history_newton[:, 0], history_newton[:, 1], "o-", color="tab:red",  markersize=0.5, linewidth=1.5, label = f"Newton, {result_newton.nit} iterations")

    ax1.scatter(x0[0], x0[1], marker = "*", color = "tab:green", s = 150, label = f"Start {x0}", zorder = 5)
    ax1.scatter(1, 1, marker = "*", color = "black", s = 150, label = "Minimum [1 1]", zorder = 5)    
    handles, labels = ax1.get_legend_handles_labels()

    
    # --- plot 2: iteration and loss ---
    ax2.set_yscale("log")
    ax2.set_title("Loss vs iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)  [log scale]")
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(loss_values_newton, color="tab:red", linewidth=1.5)
    ax2.plot(loss_values_GD, color="tab:blue", linewidth=1.5)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, -0.01), fontsize = 12, ncol = 2)

    plt.tight_layout()
    plt.savefig(f"rosenbrock_newton_GD_start{x0}.png", dpi=150, bbox_inches="tight")
    plt.show()


# -- himmelblau -
starts = [
    np.array([0.0, 0.0]),
    np.array([-0.5, 2.0]),
    np.array([1.5, -2.0]),
    np.array([-3.0, -1.0]),
]

colors = [("steelblue", "deepskyblue", "navy"),
          ("tab:red", "lightsalmon", "orangered"), 
          ("mediumseagreen", "limegreen", "darkgreen"), 
          ("mediumpurple", "mediumorchid", "rebeccapurple")]

x_grid = np.linspace(-5, 5, 400)
y_grid = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 30), colors="gray", alpha=0.5, linewidths=0.5)

for x0, x_star, cols in zip(starts, himmelblau_x_star, colors):
    mcolor, Ncolor, GDcolor = cols
    result_newton = newton(himmelblau_f, himmelblau_grad, x0, hessian_f = himmelblau_hess)
    result_GD = gradient_descent(himmelblau_f, himmelblau_grad, x0)
    
    history_newton = np.array(result_newton.history)
    history_GD = np.array(result_GD.history)
    
    plt.plot(history_newton[:, 0], history_newton[:, 1], color = Ncolor, linewidth = 1.2, label = "Newton")
    plt.plot(history_GD[:, 0], history_GD[:, 1], "--", color=GDcolor, linewidth=1.2, label = 'Gradient descent')
    
    plt.scatter(x0[0], x0[1], marker = "o", facecolors = "white", edgecolors=mcolor,
             s=80, label=f"Start {x0}")
    plt.scatter(x_star[0], x_star[1], marker = "*", color = mcolor, s = 100, label = "Minimum")
    
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.2), fontsize = 8, ncol = 4)
plt.title("Himmelblau Optimization Paths")

plt.tight_layout()
plt.savefig(r"himmelblau_newton_GD.jpg", bbox_inches = "tight")
plt.show()


# ---- making a gif of optimization ----

# -- rosenbrock
import matplotlib.animation as animation

x0 = np.array([-1, 2])
max_iter = 500

result_newton = newton(rosenbrock_f, rosenbrock_grad, x0, rosenbrock_hess, max_iter=max_iter)
result_GD = gradient_descent(rosenbrock_f, rosenbrock_grad, x0, max_iter=max_iter)

# unpack history
history_newton = np.array(result_newton.history)
xs_newton, ys_newton = history_newton[:, 0], history_newton[:, 1]

history_GD = np.array(result_GD.history)
xs_GD, ys_GD = history_GD[:, 0], history_GD[:, 1]

frames = min(len(xs_newton), len(xs_GD))

# create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# grid for contours
x_grid = np.linspace(-2, 2, 400)
y_grid = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = 100*(Y - X**2)**2 + (1 - X)**2

for ax, title in zip([ax1, ax2], ["Newton Method", "Gradient Descent"]):
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 30),
               cmap="gray", alpha=0.3, linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # start & minimum
    ax.scatter(x0[0], x0[1], marker="*", color="green", s=100, label = "Start")
    ax.scatter(rosenbrock_x_star[0], rosenbrock_x_star[1],
               marker="*", color="tab:purple", s=100, zorder=1, label = "Minimum")

# initialize lines
line_newton, = ax1.plot([], [], "o-", color="tab:blue", markersize=2, linewidth=1.5)
line_GD, = ax2.plot([], [], "o-", color="tab:red", markersize=2, linewidth=1.5)

ax1.scatter(xs_newton[-1], ys_newton[-1], marker = "x", color="black", s=80, label = "Final point")
ax2.scatter(xs_GD[-1], ys_GD[-1], marker = "x", color="black", s=80, label = "Final point")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize = 8)

def update(i):
    line_newton.set_data(xs_newton[:i], ys_newton[:i])
    line_GD.set_data(xs_GD[:i], ys_GD[:i])
    return line_newton, line_GD

ani = animation.FuncAnimation(fig, update, frames=frames, interval=300, blit=True)

ani.save(r"rosenbrock_newton_GD_path.gif", writer='pillow')