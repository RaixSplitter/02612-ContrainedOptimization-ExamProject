from utils import SolutionStats
import numpy as np
import time
from scipy.linalg import ldl

class PrimalDualInteriorPointSolver:

    def __init__(self, H, g, A, b, C, d):
        self.H = H
        self.g = g.flatten()
        self.A = A
        self.b = b.flatten()
        self.C = C
        self.d = d.flatten()

        self.k = 0
        self.max_iter = 100
        self.tol = 1e-6
        self.ni = 0.995  # step size reduction factor to ensure we stay in the interior of the feasible region

    def __name__(self):
        return "Primal-Dual Interior Point Solver"
    
    def initialize(self):
        #decided to take the same initial point as in the lecture slides 
        n = self.H.shape[0]
        mc = self.C.shape[1]

        x = np.zeros(n)
        y = np.zeros(self.A.shape[1])

        z = np.ones(mc)
        s = np.ones(mc)

        return x, y, z, s
    
    def compute_residuals(self, x, y, z, s):
        #penalties for not satisfying the KKT conditions - we want to drive these to zero
        #they tell us how far we are form optimlatity
        rL = self.H @ x + self.g - self.A @ y - self.C @ z
        rA = self.b - self.A.T @ x
        rC = s + self.d - self.C.T @ x
        rSZ = s * z  #just rSZ[i] = s[i] * z[i] product of s and z - like we multiply the diagonal matrix of s with z or the diagonal matrix of z with s - it is the same

        return rL, rA, rC, rSZ
    
    def compute_mu(self, z, s):
        #measure distance to central path - we want to drive this to zero
        return (z @ s) / len(z)
    
    def check_convergence(self, rL, rA, rC, rSZ, mu):
        #check convergence conditions - if we are close enough to solution 
        if (np.linalg.norm(rL) < self.tol and np.linalg.norm(rA) < self.tol and np.linalg.norm(rC) < self.tol and mu < self.tol):
            return True
        return False
    
    def solve_ldl(self,L, D, perm, rhs):
        rhs_perm = rhs[perm]

        # Solve L y = rhs
        y = np.linalg.solve(L, rhs_perm)

        # Solve D z = y
        z = y / np.diag(D)

        # Solve L^T x = z
        x = np.linalg.solve(L.T, z)

        # Undo permutation
        x_final = np.zeros_like(rhs)
        x_final[perm] = x

        return x_final
    
    def compute_newton_direction(self, L, D, perm, rL, rA, rC, rSZ, x, z, s):

        S_inv_Z = np.diag(z / s)

        # rL tilde
        rL_tilde = rL - self.C @ (S_inv_Z @ (rC - rSZ / z))

        rhs = -np.concatenate([rL_tilde, rA])
        sol = self.solve_ldl(L, D, perm, rhs)

        dx = sol[:len(x)]
        dy = sol[len(x):]

        dz = (z / s) * (rC - rSZ / z - self.C.T @ dx)
        ds = -rSZ / z - (s / z) * dz

        return dx, dy, dz, ds
    
    def compute_step_alpha(self, z, dz, s, ds):
        alpha = 1.0

        idx = dz < 0
        if np.any(idx):
            alpha = min(alpha, np.min(-z[idx] / dz[idx]))

        idx = ds < 0
        if np.any(idx):
            alpha = min(alpha, np.min(-s[idx] / ds[idx]))

        # we take a fraction of the step to ensure we stay in the interior of the feasible region
        return self.ni * alpha 
    
    def solve(self):
        start = time.time()

        x, y, z, s = self.initialize()

        while self.k < self.max_iter :
            rL, rA, rC, rSZ = self.compute_residuals(x, y, z, s)
            mu = self.compute_mu(z, s)
            if self.check_convergence(rL, rA, rC, rSZ, mu):
                obj = 0.5 * x @ (self.H @ x) + self.g @ x
                end = time.time()
                return SolutionStats(
                    x=x,
                    iterations=self.k,
                    time=end - start,
                    obj=obj,
                    feasibility=None
                )

            #Predictor step

            #we build the KKT system first
            S_inv_Z = np.diag(z / s)
                #compute H_bar
            H_bar = self.H + self.C @ S_inv_Z @ self.C.T
            KKT_matrix = np.block([
                [H_bar, -self.A],
                [-self.A.T, np.zeros((self.A.shape[1], self.A.shape[1]))]
            ])
                # compute the LDL factorization of the KKT matrix
            L, D, perm = ldl(KKT_matrix)

            #affine direction - just get the aggressive direction without centering

            dx_aff, dy_aff, dz_aff, ds_aff = self.compute_newton_direction(L, D, perm, rL, rA, rC, rSZ, x, z, s)
            alpha_aff = self.compute_step_alpha(z, dz_aff, s, ds_aff)


            #compute affine duality gap 
            mu_aff = ((z + alpha_aff * dz_aff) @ (s + alpha_aff * ds_aff)) / len(z)
            #compute centering parameter 
            sigma = (mu_aff / mu)**3

            #Corrector step

            #affine-centering correction direction
            rSZ_corr = rSZ + dz_aff * ds_aff - sigma * mu * np.ones_like(z)

            dx, dy, dz, ds = self.compute_newton_direction(L, D, perm, rL, rA, rC, rSZ_corr, x, z, s)
            alpha = self.compute_step_alpha(z, dz, s, ds)

            #update iteration
            x = x + alpha * dx
            y = y + alpha * dy
            z = z + alpha * dz
            s = s + alpha * ds

            self.k += 1

        #no solution found within max iterations
        end = time.time()
        obj = 0.5 * x @ (self.H @ x) + self.g @ x
        return SolutionStats(
            x=x,
            iterations=self.k,
            time=end - start,
            obj=obj,
            feasibility=None 
        )