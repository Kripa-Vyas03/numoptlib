import numpy as np
import pytest
from numoptlib.unconstrained.newton import newton

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
    def hess_f(x):
        return A
    x_star = np.linalg.solve(A, b)
    return f, grad_f, hess_f, x_star

@pytest.fixture
def rosenbrock():
    def f(x):
        return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    def grad_f(x):
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])
    def hess_f(x):
        return np.array([
            [-400*(x[1] - x[0]**2) + 800*x[0]**2 + 2, -400*x[0]],
            [-400*x[0],                                  200.0    ]
        ])
    return f, grad_f, hess_f

# --- correctness tests ---

def test_quadratic_exact_solution(quadratic):
    """Newton should solve a quadratic exactly in one step."""
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.zeros(2)
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    np.testing.assert_allclose(result.x, x_star, atol=1e-10)

def test_quadratic_converges(quadratic):
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.zeros(2)
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-6)

def test_rosenbrock_converges_with_exact_hessian(rosenbrock):
    f, grad_f, hess_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert result.success
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-6)

def test_rosenbrock_converges_with_finite_diff_hessian(rosenbrock):
    """Newton should converge without an analytical Hessian supplied."""
    f, grad_f, hess_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = newton(f, grad_f, x0, hessian_f = None)    # no hess_f passed
    assert result.success
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-4)

def test_finite_diff_close_to_analytical(rosenbrock):
    """Finite difference Hessian should be close to analytical Hessian."""
    from numoptlib.unconstrained.newton import finite_diff_hessian
    f, grad_f, hess_f = rosenbrock
    x = np.array([-1.0, 1.0])
    H_analytical = hess_f(x)
    H_numerical = finite_diff_hessian(f, x)
    np.testing.assert_allclose(H_numerical, H_analytical, atol=1e-5)

# --- result structure tests ---

def test_result_has_expected_fields(quadratic):
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.zeros(2)
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert hasattr(result, "grad")
    assert hasattr(result, "nit")
    assert hasattr(result, "nfev")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "history")

def test_history_length_matches_nit(quadratic):
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.zeros(2)
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert len(result.history) == result.nit + 1

def test_history_is_decreasing(rosenbrock):
    f, grad_f, hess_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    values = [f(x) for x in result.history]
    assert all(values[i] >= values[i+1] for i in range(len(values)-1))

def test_already_at_minimum(quadratic):
    f, grad_f, hess_f, x_star = quadratic
    result = newton(f, grad_f, x_star, hessian_f=hess_f)
    assert result.nit <= 2

# --- Newton-specific tests ---

def test_quadratic_converges_in_one_step(quadratic):
    """
    Newton is exact for quadratics — it should converge in exactly
    one iteration when given the analytical Hessian.
    """
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.zeros(2)
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert result.nit == 1

def test_faster_than_gradient_descent_on_rosenbrock(rosenbrock):
    """Newton should converge in far fewer iterations than gradient descent."""
    from numoptlib.unconstrained.gradient_descent import gradient_descent
    f, grad_f, hess_f = rosenbrock
    x0 = np.array([-1.1, 1.1])
    result_newton = newton(f, grad_f, x0, hessian_f=hess_f)
    result_gd = gradient_descent(f, grad_f, x0)
    assert result_newton.nit < result_gd.nit

def test_hessian_solve_not_invert(quadratic):
    """
    Indirectly verify we're solving the linear system rather than
    inverting — result should be numerically stable on a well-conditioned
    problem.
    """
    f, grad_f, hess_f, x_star = quadratic
    x0 = np.array([10.0, 10.0])    # far from minimum
    result = newton(f, grad_f, x0, hessian_f=hess_f)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-6)

def test_max_iter_respected(rosenbrock):
    f, grad_f, hess_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = newton(f, grad_f, x0, hessian_f=hess_f, max_iter=2)
    assert result.nit == 2
    assert not result.success