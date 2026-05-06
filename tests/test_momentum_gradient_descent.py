import numpy as np
import pytest
from numoptlib.unconstrained.momentum_gradient_descent import momentum_gradient_descent

# --- fixtures ---

@pytest.fixture
def quadratic():
    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    def grad_f(x):
        return A @ x - b
    x_star = np.linalg.solve(A, b)
    return f, grad_f, x_star

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

def test_quadratic_converges(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = momentum_gradient_descent(f, grad_f, x0)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-4)

def test_rosenbrock_converges(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = momentum_gradient_descent(f, grad_f, x0)
    assert result.success
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-3)

def test_zero_beta_matches_gradient_descent(rosenbrock):
    """With beta=0, velocity never accumulates — should behave like vanilla GD."""
    from numoptlib.unconstrained.gradient_descent import gradient_descent
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result_momentum = momentum_gradient_descent(f, grad_f, x0, beta=0.0)
    result_gd = gradient_descent(f, grad_f, x0)
    np.testing.assert_allclose(result_momentum.x, result_gd.x, atol=1e-6)

def test_higher_beta_fewer_iterations(rosenbrock):
    """More momentum should converge faster on a well-behaved function."""
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result_low  = momentum_gradient_descent(f, grad_f, x0, beta=0.1)
    result_high = momentum_gradient_descent(f, grad_f, x0, beta=0.9)
    assert result_high.nit <= result_low.nit

def test_history_is_decreasing(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = momentum_gradient_descent(f, grad_f, x0)
    values = [f(x) for x in result.history]
    assert all(values[i] >= values[i+1] for i in range(len(values)-1))

# --- result structure tests ---

def test_result_has_expected_fields(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = momentum_gradient_descent(f, grad_f, x0)
    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert hasattr(result, "grad")
    assert hasattr(result, "nit")
    assert hasattr(result, "nfev")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "history")

def test_history_length_matches_nit(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = momentum_gradient_descent(f, grad_f, x0)
    assert len(result.history) == result.nit + 1

def test_already_at_minimum(quadratic):
    f, grad_f, x_star = quadratic
    result = momentum_gradient_descent(f, grad_f, x_star)
    assert result.nit <= 2

def test_max_iter_respected(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-2.0, 1.0])
    result = momentum_gradient_descent(f, grad_f, x0, max_iter=5)
    assert result.nit == 5
    assert not result.success

def test_nfev_positive(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = momentum_gradient_descent(f, grad_f, x0)
    assert result.nfev > 0

# --- momentum specific tests ---

def test_velocity_reset_on_bad_direction(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.array([10.0, 10.0])
    # beta=0.9 is realistic — 0.99 causes excessive oscillation near minimum
    result = momentum_gradient_descent(f, grad_f, x0, beta=0.9, max_iter=500)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-4)

def test_beta_zero_velocity_never_accumulates(quadratic):
    """With beta=0, v = grad each iteration — momentum has no effect."""
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = momentum_gradient_descent(f, grad_f, x0, beta=0.0)
    assert result.success