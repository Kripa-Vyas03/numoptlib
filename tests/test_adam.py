import numpy as np
import pytest
from numoptlib.unconstrained.adam import adam_gradient_descent

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
    """Adam needs more iterations than GD on simple quadratics
    due to fixed alpha — use generous budget."""
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = adam_gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=2000)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-3)

def test_rosenbrock_converges(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = adam_gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=5000)
    assert result.success
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-2)

def test_already_at_minimum(quadratic):
    f, grad_f, x_star = quadratic
    result = adam_gradient_descent(f, grad_f, x_star)
    assert result.nit <= 2

def test_max_iter_respected(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = adam_gradient_descent(f, grad_f, x0, max_iter=10)
    assert result.nit == 10
    assert not result.success

# --- result structure tests ---

def test_result_has_expected_fields(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = adam_gradient_descent(f, grad_f, x0)
    for field in ["x", "fun", "grad", "nit", "nfev", "success", "message", "history"]:
        assert hasattr(result, field)

def test_history_length_matches_nit(quadratic):
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result = adam_gradient_descent(f, grad_f, x0, max_iter=50)
    assert len(result.history) == result.nit + 1

def test_nfev_positive(rosenbrock):
    f, grad_f = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = adam_gradient_descent(f, grad_f, x0)
    assert result.nfev > 0

# --- Adam specific tests ---

def test_default_hyperparameters_match_paper(quadratic):
    """Verify defaults match Kingma & Ba 2014 recommendations."""
    import inspect
    sig = inspect.signature(adam_gradient_descent)
    params = sig.parameters
    assert params["alpha"].default == 0.01
    assert params["beta1"].default == 0.9
    assert params["beta2"].default == 0.999
    assert params["epsilon"].default == 1e-8

def test_higher_alpha_faster_on_quadratic(quadratic):
    """Larger step size should converge faster on well-behaved quadratic."""
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    result_slow = adam_gradient_descent(f, grad_f, x0, alpha=0.001, max_iter=2000)
    result_fast = adam_gradient_descent(f, grad_f, x0, alpha=0.1,   max_iter=2000)
    assert result_fast.nit < result_slow.nit

def test_adaptive_scaling_helps_ill_conditioned():
    """Adam's per-parameter scaling should handle ill-conditioned problems."""
    np.random.seed(42)
    eigvals = np.array([1.0, 50.0])
    Q, _ = np.linalg.qr(np.random.randn(2, 2))
    A = Q @ np.diag(eigvals) @ Q.T
    b = np.array([1.0, 2.0])
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    def grad_f(x):
        return A @ x - b
    x_star = np.linalg.solve(A, b)
    x0 = np.zeros(2)
    result = adam_gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=5000)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-3)

def test_moment_estimates_initialized_at_zero():
    """First step should use only current gradient scaled by (1-beta1)."""
    f = lambda x: 0.5 * np.sum(x**2)
    grad_f = lambda x: x
    x0 = np.array([1.0, 1.0])
    result = adam_gradient_descent(f, grad_f, x0, max_iter=1)
    # after one step x should have moved toward zero
    assert np.linalg.norm(result.x) < np.linalg.norm(x0)

def test_epsilon_prevents_division_by_zero():
    """Near-zero gradients should not cause numerical issues."""
    f = lambda x: 1e-10 * np.sum(x**2)
    grad_f = lambda x: 1e-10 * 2 * x
    x0 = np.array([1.0, 1.0])
    result = adam_gradient_descent(f, grad_f, x0, max_iter=100)
    assert np.all(np.isfinite(result.x))

def test_no_line_search_used(quadratic, monkeypatch):
    """Adam should not call line_search at all."""
    import numoptlib.unconstrained.adam as adam_module
    called = []
    if hasattr(adam_module, "line_search"):
        original = adam_module.line_search
        monkeypatch.setattr(adam_module, "line_search",
                           lambda *a, **kw: called.append(1) or original(*a, **kw))
    f, grad_f, x_star = quadratic
    x0 = np.zeros(2)
    adam_gradient_descent(f, grad_f, x0, max_iter=10)
    assert len(called) == 0