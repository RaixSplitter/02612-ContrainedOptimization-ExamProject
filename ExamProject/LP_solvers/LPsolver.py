import cvxpy as cp
import numpy as np
import time
from utils import SolutionStats

def LPsolver(H, g, bl, A, bu, l, u):
    n = H.shape[0]
    m = A.shape[1]

    # Variables
    x = cp.Variable(n)
    s1 = cp.Variable(m, nonneg=True)
    s2 = cp.Variable(m, nonneg=True)
    s3 = cp.Variable(n, nonneg=True)
    s4 = cp.Variable(n, nonneg=True)

    # Flatten inputs
    bl = bl.flatten()
    bu = bu.flatten()
    l = l.flatten()
    u = u.flatten()

    # Constraints (relaxed)
    constraints = [
        A.T @ x + s1 >= bl,
        A.T @ x - s2 <= bu,
        x + s3 >= l,
        x - s4 <= u
    ]

    # Objective: minimize violations
    objective = cp.Minimize(cp.sum(s1) + cp.sum(s2) + cp.sum(s3) + cp.sum(s4))

    prob = cp.Problem(objective, constraints)
    start = time.time()
    # solve problem
    prob.solve()
    end = time.time()

    return SolutionStats(
            x=x.value,
            iterations=prob.solver_stats.num_iters,
            time=end - start,
            obj=None,
            feasibility = (prob.status == "optimal" and prob.value < 1e-6)
        )