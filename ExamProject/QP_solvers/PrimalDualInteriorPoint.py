from utils import SolutionStats
import numpy as np
import time
from scipy.linalg import solve
from scipy.sparse import issparse

class PrimalDualInteriorPointSolver:

    def __init__(self, H, g, A, b, C, d):
        self.H = H.toarray() if issparse(H) else np.asarray(H)
        self.g = g.flatten()
        self.A = A
        self.b = b.flatten()
        self.C = C
        self.d = d.flatten()

        self.k = 0
        self.max_iter = 1000
        self.tol = 1e-6
        self.ni = 0.995  # step size reduction factor to ensure we stay in the interior of the feasible region

    def __name__(self):
        return "Primal-Dual Interior Point Solver"        
    
    def initialize(self):
        #we follow the heuristic for an initial point from the lectures
        n = self.H.shape[0]
        m = self.A.shape[1]
        mc = self.C.shape[1]

        #same as in the primal dual interior point pdf
        x_bar = np.zeros(n)
        y_bar = np.zeros(m)
        z_bar = np.ones(mc)
        s_bar = np.ones(mc)

        #Compute residuals 
        rL, rA, rC, rSZ = self.compute_residuals(x_bar, y_bar, z_bar, s_bar)

        #compute affine direction using augmented system
        dx_aff, dy_aff, dz_aff, ds_aff = self.compute_newton_direction(
            rL, rA, rC, rSZ, x_bar, z_bar, s_bar
        )

        #enforce positivity
        z = np.maximum(1.0, np.abs(z_bar + dz_aff))
        s = np.maximum(1.0, np.abs(s_bar + ds_aff))
        x = x_bar + dx_aff
        y = y_bar + dy_aff

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
        if (np.linalg.norm(rL) < self.tol and 
            np.linalg.norm(rA) < self.tol and 
            np.linalg.norm(rC) < self.tol and 
            mu < self.tol):
            return True
        return False
    
    def compute_newton_direction(self, rL, rA, rC, rSZ, x, z, s, target=None):
        """Solve the augmented 2-block system to get Newton directions.

        Instead of the reduced normal equations (which require z/s ratios
        that blow up for near-active constraints), solve the larger but
        numerically stable augmented system:

            [H,     -C ] [dx]   =  [-rL                  ]
            [Z*C^T,  S ] [dz]      [Z*rC - rSZ + target  ]

        where Z = diag(z), S = diag(s).  No z/s ratio appears.
        dz is O(1) even when z/s → ∞.
        """
        if target is None:
            target = np.zeros_like(z)

        n  = len(x)
        mc = len(z)

        Z  = z          # diagonal entries of diag(z)
        S  = s          # diagonal entries of diag(s)

        # Build the (n + mc) x (n + mc) augmented system
        #   [H,      -C  ]
        #   [Z*C^T,   S  ]
        top_left  = self.H                     # n x n
        top_right = -self.C                    # n x mc
        bot_left  = (Z[:, None] * self.C.T)    # mc x n  (row-wise scale of C^T)
        bot_right = np.diag(S)                 # mc x mc

        A_aug = np.block([
            [top_left,  top_right],
            [bot_left,  bot_right]
        ])

        rhs_top = -rL
        rhs_bot = Z * rC - rSZ + target        # = z*rC - z*s + target
        rhs_aug = np.concatenate([rhs_top, rhs_bot])

        sol    = solve(A_aug, rhs_aug)
        dx     = sol[:n]
        dz     = sol[n:]

        # dy is zero (no equality constraints in this formulation; A is empty)
        dy = np.zeros(self.A.shape[1])

        # ds from primal feasibility: ds = C^T dx - rC
        ds = self.C.T @ dx - rC

        return dx, dy, dz, ds
    
    def compute_step_alpha(self, z, dz, s, ds):
        alpha_p = 1.0   # primal: governs s (and x, which is unconstrained but fine)
        alpha_d = 1.0   # dual:   governs z

        idx = ds < 0
        if np.any(idx):
            alpha_p = min(alpha_p, np.min(-s[idx] / ds[idx]))

        idx = dz < 0
        if np.any(idx):
            alpha_d = min(alpha_d, np.min(-z[idx] / dz[idx]))

        return alpha_p, alpha_d
    
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

            #Predictor step — affine direction (no centering)
            dx_aff, dy_aff, dz_aff, ds_aff = self.compute_newton_direction(
                rL, rA, rC, rSZ, x, z, s, target=None
            )
            alpha_aff_p, alpha_aff_d = self.compute_step_alpha(z, dz_aff, s, ds_aff)

            #compute affine duality gap (split step)
            mu_aff = ((z + alpha_aff_d * dz_aff) @ (s + alpha_aff_p * ds_aff)) / len(z)
            mu_aff = max(mu_aff, 0.0)
            #compute centering parameter
            sigma = np.clip((mu_aff / mu)**3, 0.0, 1.0)

            #Corrector step — Mehrotra centering + second-order correction
            # dz_aff and ds_aff are O(1) (augmented system avoids z/s blowup)
            # so the cross product dz_aff*ds_aff is bounded and the correction is valid
            target_corr = sigma * mu - dz_aff * ds_aff

            dx, dy, dz, ds = self.compute_newton_direction(
                rL, rA, rC, rSZ, x, z, s, target=target_corr
            )
            alpha_p, alpha_d = self.compute_step_alpha(z, dz, s, ds)

            # we take a fraction of the step to ensure we stay in the interior of the feasible region
            alpha_p = self.ni * alpha_p
            alpha_d = self.ni * alpha_d

            #update iteration
            x = x + alpha_p * dx
            y = y + alpha_d * dy
            z = z + alpha_d * dz
            s = s + alpha_p * ds

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