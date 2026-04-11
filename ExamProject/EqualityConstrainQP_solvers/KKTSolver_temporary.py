import numpy as np

class KKTSolver:

    def solve(self, H, g, x, A_W):
        """
        Solves:
            min 1/2 p^T H p + (Hx + g)^T p
            s.t. A_W^T p = 0

        Returns:
            p (search direction)
            lambda_ (Lagrange multipliers)
        """

        n = H.shape[0]
        m = A_W.shape[0]  # number of active constraints

        # Gradient at current point
        gk = H @ x + g

        # Build KKT matrix
        if m > 0:
            KKT = np.block([
                [H,           -A_W.T],
                [-A_W,        np.zeros((m, m))]
            ])

            rhs = np.concatenate([
                -gk,
                np.zeros(m)
            ])

        else:
            # No constraints → unconstrained step
            KKT = H
            rhs = -gk

        # Solve system
        sol = np.linalg.solve(KKT, rhs)

        if m > 0:
            p = sol[:n]
            lambda_ = sol[n:]
        else:
            p = sol
            lambda_ = np.array([])

        return p, lambda_