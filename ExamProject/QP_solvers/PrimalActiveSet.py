from LP_solvers.LPsolver import LPsolver
from utils import SolutionStats
from EqualityConstrainQP_solvers.KKTSolver_temporary import KKTSolver
import numpy as np
import time

class PrimalActiveSetSolver:

    def __init__(self, H, g, bl, A, bu, l, u):
        self.H = H
        self.g = g.flatten()
        self.bl = bl.flatten()
        self.A = A
        self.bu = bu.flatten()
        self.l = l.flatten()
        self.u = u.flatten()

        self.k = 0
        self.max_iterations = 1000

    def __name__(self):
        return "Primal Active Set Solver"
    
    def build_working_set(self, x, tol=1e-6):
        W = []

        Ax = self.A.T @ x

        # Linear constraints
        for i in range(len(Ax)):
            if abs(Ax[i] - self.bl[i]) <= tol:
                W.append(("bl", i))
            elif abs(Ax[i] - self.bu[i]) <= tol:
                W.append(("bu", i))

        # Box constraints
        for i in range(len(x)):
            if abs(x[i] - self.l[i]) <= tol:
                W.append(("xl", i))
            elif abs(x[i] - self.u[i]) <= tol:
                W.append(("xu", i))

        return W
    
    def build_A_W(self, W):
        rows = []

        for (ctype, i) in W:

            if ctype == "bl":
                # a_i^T x >= bl  →  row = +a_i
                row = self.A[:, i].toarray().flatten()
            elif ctype == "bu":
                # a_i^T x <= bu  →  rewrite as -a_i^T x >= -bu  →  row = -a_i
                row = -self.A[:, i].toarray().flatten()
            elif ctype == "xl":
                # x_i >= l_i  →  row = +e_i
                row = np.zeros(self.H.shape[0])
                row[i] = 1.0
            elif ctype == "xu":
                # x_i <= u_i  →  rewrite as -x_i >= -u_i  →  row = -e_i
                row = np.zeros(self.H.shape[0])
                row[i] = -1.0

            rows.append(row)

        if len(rows) == 0:
            return np.zeros((0, self.H.shape[0]))

        A_W = np.vstack(rows)

        return A_W

    def compute_alpha(self, x, p, W, tol=1e-6):
        alpha = 1.0
        hit_constraint = None

        n = len(x)
        m = self.A.shape[1]

        Ax = self.A.T @ x
        Ap = self.A.T @ p

        #Liniear constraints
        for i in range(m):

            #skip active constraints already in W
            if ("bl", i) in W or ("bu", i) in W:
                continue

            if Ap[i] > tol:
                alpha_i = (self.bu[i] - Ax[i]) / Ap[i]
                if alpha_i >= 0 and alpha_i < alpha:
                    alpha = alpha_i
                    hit_constraint = ("bu", i)

            elif Ap[i] < -tol:
                alpha_i = (self.bl[i] - Ax[i]) / Ap[i]
                if alpha_i >= 0 and alpha_i < alpha:
                    alpha = alpha_i
                    hit_constraint = ("bl", i)

        #Box constraints
        for i in range(n):

            #skip active constraints already in W
            if ("xl", i) in W or ("xu", i) in W:
                continue

            if p[i] > tol:
                alpha_i = (self.u[i] - x[i]) / p[i]
                if alpha_i >= 0 and alpha_i < alpha:
                    alpha = alpha_i
                    hit_constraint = ("xu", i)

            elif p[i] < -tol:
                alpha_i = (self.l[i] - x[i]) / p[i]
                if alpha_i >= 0 and alpha_i < alpha:
                    alpha = alpha_i
                    hit_constraint = ("xl", i)

        return alpha, hit_constraint
    
    def solve(self):
        start = time.time()
        # Init
        #find feasible starting point
        x = LPsolver(self.H, self.g, self.bl, self.A, self.bu, self.l, self.u).x

        #build working set W_0 - that includes all active constraints at the feasible starting point
        W = self.build_working_set(x)

        #Main loop
        while self.k < self.max_iterations:
            A_W = self.build_A_W(W)
            kkt_solver = KKTSolver()
            p, lambda_ = kkt_solver.solve(self.H, self.g, x, A_W)

            #Case 1: p = 0
            if np.linalg.norm(p) < 1e-6:
                #if all lambda >= 0 - all constaints in W are satisfied - we found a solution
                if np.all(lambda_ >= 0):
                    obj = 0.5 * x @ (self.H @ x) + self.g @ x
                    end = time.time()
                    return SolutionStats(
                        x=x,
                        iterations=self.k,
                        time=end - start,
                        obj=obj,
                        feasibility=None 
                    )
                else:
                    #remove the constraint with the  negative lambda from W and repeat (most penalized constraint)
                    neg_idx = np.where(lambda_ < -1e-8)[0]
                    j = neg_idx[0]
                    W.pop(j) # W_k+1 = W_k \ {penalized_constraint}
                             # x_k+1 = x_k
                    self.k += 1

            #Case 2: p != 0
            else:
                #we compute the max step we can take (alpha) following p direstion without violating a constraints inactive
                alpha, hit_constraint = self.compute_alpha(x, p, W) 

                #a constaint was hit
                if alpha < 1.0:
                    #we add the hit constraint to our active set and add it to W_k+1 and take the biggest step (alpha) we can 
                    if hit_constraint not in W:
                        W.append(hit_constraint) # W_k+1 = W_k + {hit_constraint}
                    x = x + alpha * p # x_k+1 = x_k + alpha * p

                #no constraint was hit - we can take the full step
                else:
                    x = x + p # x_k+1 = x_k + p 

                self.k += 1


        #no solution found within max iterations
        end = time.time()
        obj = 0.5 * x @ (self.H @ x) + self.g @ x
        return SolutionStats(
            x=None,
            iterations=self.k,
            time=end - start,
            obj=obj,
            feasibility=None 
        )