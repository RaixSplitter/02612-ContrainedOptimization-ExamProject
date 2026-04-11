import numpy as np
import scipy.sparse as sp

class KKTSolver:

    def solve(self, H, g, x, A_W):
        H = np.asarray(H.toarray() if sp is not None and sp.issparse(H) else H)
        n = H.shape[0]
        m = A_W.shape[0]

        gk = H @ x + g

        if m == 0:
            p = np.linalg.solve(H, -gk)
            return p, np.array([])

        # Build KKT matrix
        KKT = np.block([
            [H, -A_W.T],
            [-A_W, np.zeros((m, m))]
        ])

        rhs = -np.concatenate([gk, np.zeros(m)])

        sol = np.linalg.solve(KKT, rhs)

        p = sol[:n]
        lambda_ = sol[n:]

        return p, lambda_
