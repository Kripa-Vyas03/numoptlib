import numpy as np
import pytest
from numoptlib.utils.line_search import line_search

# --- fixtures ---

@pytest.fixture
def quadratic():
    A = np.array([[2, 0], [0, 20]])   # corresponds to f = x^2 + 10y^2
    def f(x):
        return x[0]**2 + 10*x[1]**2
    def grad_f(x):
        return np.array([2*x[0], 20*x[1]])
    return f, grad_f, A

@pytest.fixture
def rosenbrock():
    def f(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    def grad_f(x):
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])
    return f, grad_f

# --- correctness tests ---

def test_quadratic_armijo_satisfied(quadratic):
    f, grad_f, _ = quadratic
    x = np.array([1.0, 1.0])
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    c1 = 1e-4
    lhs = f(x + alpha*p)
    rhs = f(x) + c1*alpha*(grad_f(x) @ p)
    assert lhs <= rhs, f"Armijo failed: {lhs} > {rhs}"

def test_quadratic_curvature_satisfied(quadratic):
    f, grad_f, _ = quadratic
    x = np.array([1.0, 1.0])
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    c2 = 0.9
    dphi_0 = grad_f(x) @ p
    dphi_alpha = grad_f(x + alpha*p) @ p
    assert abs(dphi_alpha) <= c2*abs(dphi_0), "Curvature condition failed"

def test_quadratic_exact_solution(quadratic):
    """For a pure quadratic, the exact alpha has a closed form."""
    f, grad_f, A = quadratic
    x = np.array([1.0, 1.0])
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    g = grad_f(x)
    alpha_exact = -(g @ p) / (p @ A @ p)
    assert abs(alpha - alpha_exact) < 1e-4, f"Got {alpha}, expected {alpha_exact}"

def test_rosenbrock_armijo_satisfied(rosenbrock):
    f, grad_f = rosenbrock
    x = np.array([-1.0, 1.0])   # not at minimum
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    c1 = 1e-4
    assert f(x + alpha*p) <= f(x) + c1*alpha*(grad_f(x) @ p)
    
def test_rosenbrock_curvature_satisfied(rosenbrock):
    f, grad_f = rosenbrock
    x = np.array([-1.0, 1.0])
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    c2 = 0.9
    dphi_0 = grad_f(x) @ p
    dphi_alpha = grad_f(x + alpha*p) @ p
    assert abs(dphi_alpha) <= c2*abs(dphi_0), "Curvature condition failed"

def test_returns_positive_step(rosenbrock):
    f, grad_f = rosenbrock
    x = np.array([-1.0, 1.0])
    p = -grad_f(x)
    alpha = line_search(f, grad_f, x, p)
    assert alpha > 0

def test_descent_direction_required(quadratic):
    """Line search should only be called with a descent direction."""
    f, grad_f, _ = quadratic
    x = np.array([1.0, 1.0])
    p = grad_f(x)    # ascent direction — dphi(0) > 0
    # this isn't a test that line_search raises an error (it might not)
    # it's a reminder that results are undefined for non-descent directions
    dphi_0 = grad_f(x) @ p
    assert dphi_0 > 0   # documents the assumption, not a bug