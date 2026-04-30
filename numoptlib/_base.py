from dataclasses import dataclass
import numpy as np

@dataclass
class OptimizeResult:
    x: np.ndarray          # solution
    fun: float             # objective value at solution
    grad: np.ndarray       # gradient at solution
    nit: int               # number of iterations
    nfev: int              # number of function evaluations
    success: bool          # did it converge
    message: str           # why it stopped
    history: list          # x at each iteration, for plotting convergence
    
    
class CountedFunction:
    def __init__(self, f):
        self.f = f
        self.nfev = 0

    def __call__(self, x):
        self.nfev += 1
        return self.f(x)