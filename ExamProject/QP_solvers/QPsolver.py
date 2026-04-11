import cvxpy as cp
import numpy as np
import time
from utils import SolutionStats

class QPsolver:

    def __init__(self, H, g, bl, A, bu, l, u):
        self.H = H
        self.g = g.flatten()
        self.bl = bl.flatten()
        self.A = A
        self.bu = bu.flatten()
        self.l = l.flatten()
        self.u = u.flatten()

    def __name__(self):
        return "CVXPY QP Solver"
    
    def solve(self):
        n = self.H.shape[0]

        # Define variable
        x = cp.Variable(n)

        # Objective 
        H_psd = cp.psd_wrap(self.H) #— psd_wrap skips the ARPACK PSD check for large sparse H 
                                    #since I decided to reuse the presentation code, which generates sparse H 
        objective = 0.5 * cp.quad_form(x, H_psd) + self.g @ x

        # Constraints
        constraints = [
            self.A.T @ x >= self.bl,
            self.A.T @ x <= self.bu,
            x >= self.l,
            x <= self.u
        ]

        # Problem
        prob = cp.Problem(cp.Minimize(objective), constraints)

        start = time.time()
        # solve problem
        prob.solve()
        end = time.time()

        x = x.value
        obj = 0.5 * x @ (self.H @ x) + self.g @ x

        return SolutionStats(
            x=x,
            iterations=prob.solver_stats.num_iters,
            time=end - start,
            obj=obj,
            feasibility=None 
        )