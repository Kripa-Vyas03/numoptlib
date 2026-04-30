import numpy as np
from numoptlib.unconstrained.gradient_descent import gradient_descent

def test_quadratic_converges_to_known_sol(quadratic):
    f, grad, x_star = quadratic
    x0 = np.zeros(len(x_star))
    result = gradient_descent(f, grad, x0)
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol = 1e-4)
    
def test_rosenbrock_reaches_minimum(rosenbrock):
    f, grad = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = gradient_descent(f, grad, x0)
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-3)

def test_result_has_expected_fields(quadratic):
    f, grad, x_star = quadratic
    x0 = np.zeros(len(x_star))
    result = gradient_descent(f, grad, x0)
    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert hasattr(result, "nit")
    assert hasattr(result, "history")
    assert len(result.history) == result.nit + 1

def test_already_at_minimum(quadratic):
    f, grad, x_star = quadratic
    result = gradient_descent(f, grad, x_star)
    assert result.nit <= 2

def test_history_is_decreasing(rosenbrock):
    f, grad = rosenbrock
    x0 = np.array([-1.0, 1.0])
    result = gradient_descent(f, grad, x0)
    values = [f(x) for x in result.history]
    assert all(values[i] >= values[i+1] for i in range(len(values)-1))