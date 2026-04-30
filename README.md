# numoptlib
A Python library implementing classical numerical optimization algorithms from scratch, built as a learning project alongside graduate coursework in computational geophysics. The goal is to provide clean, well-tested implementations of gradient-based optimization methods with a consistent API, convergence diagnostics, and benchmarking tools.

## Motivation
Most scientific Python code treats optimization as a black box — pass a function to scipy.optimize.minimize and get an answer back. This library is an attempt to open that box: to implement the algorithms from mathematical foundations up, understand the tradeoffs between methods, and build software around them that is usable by others.
The methods implemented here are drawn primarily from Nocedal & Wright, Numerical Optimization (2nd ed.), which is the standard reference for the field.

## Methods
- Gradient descent with backtracking and strong Wolfe line search
- Momentum / heavy ball accelerated gradient descent
- Newton's method with exact or finite difference Hessian
- BFGS (quasi-Newton)
- Projected gradient descent for box-constrained problems

## Installation
```
git clone https://github.com/Kripa-Vyas03/numoptlib
cd numoptlib
pip install -e .
```

## Usage
To be completed with real examples and API signatures once implementation is finalised.

## Benchmarks
To be completed — will include convergence rate comparisons across methods on standard test functions (Rosenbrock, Beale, Himmelblau) with iteration count, function evaluations, and suboptimality gap plots.

## Running the tests
```
python -m pytest tests/ -v
```

## Dependencies
- Python >= 3.9
- NumPy
- Matplotlib (for examples)

## References
Nocedal, J. & Wright, S. (2006). Numerical Optimization. Springer.
Boyd, S. & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press. (freely available)
