import numpy as np
import scipy.sparse as sp

class RandomQPGenerator:
    """
    Generates data for a random convex QP

    min (1/2)x^T H x + g^T x
    s.t. bl <= A^T x <= bu
        l <= x <= u

    Parameters:
        n       : number of variables
        alpha   : regularization factor (alpha > 0)
        density : density of sparse matrices (0 < density < 1)

    Returns:
        H, g, bl, A, bu, l, u
        H and A are sparse matrices
    """
    def __init__(self, n, alpha, density, beta, flag="sparse"):
        self.n = n
        self.alpha = alpha
        self.density = density
        self.beta = beta
        self.flag = flag

        self.H = None
        self.g = None
        self.bl = None
        self.A = None
        self.bu = None
        self.l = None
        self.u = None

        self.C = None
        self.d = None
        self.A_eq = None
        self.b_eq = None

    def generate_interior_point_form(self, H, g, bl, A, bu, l, u):
        n, m = A.shape

        # Build C — keep sparse or dense to match A
        if sp.issparse(A):
            I = sp.eye(n, format='csr')
            self.C = sp.hstack([A, -A, I, -I], format='csr')
        else:
            self.C = np.hstack([A, -A, np.eye(n), -np.eye(n)])

        # Build d
        self.d = np.vstack([
            bl,
            -bu,
            l,
            -u
        ]).flatten()

        # No equality constraints
        self.A_eq = np.zeros((n, 0))
        self.b_eq = np.zeros(0)


    def generate(self):
        """
        Returns:
            H, g, bl, A, bu, l, u - we call this the "general" form
            H, g, A_eq, b_eq, C, d - we call this the "interior-point" form
            H and A are sparse matrices
        """

        m = round(self.beta * self.n)

        # Sparse A ~ N(0,1)
        self.A = sp.random(self.n, m, density=self.density, format='csr', data_rvs=np.random.randn)

        # Sparse M
        M = sp.random(self.n, self.n, density=self.density, format='csr', data_rvs=np.random.randn)

        # H = M M^T + alpha I (positive definite)
        self.H = M @ M.T + self.alpha * sp.eye(self.n, format='csr')

        # Convert to dense immediately if requested
        if self.flag == 'dense':
            self.A = self.A.toarray()
            self.H = self.H.toarray()

        # Bounds
        self.bl = -np.random.rand(m, 1)     # U([-1,0])
        self.bu = np.random.rand(m, 1)      # U([0,1])

        # Linear term g ~ U([-1,1])
        self.g = np.random.uniform(-1, 1, (self.n, 1))

        # Variable bounds
        self.l = -np.ones((self.n, 1))
        self.u = np.ones((self.n, 1))

        self.generate_interior_point_form(self.H, self.g, self.bl, self.A, self.bu, self.l, self.u)
        

    def get_general_problem(self):
        return self.H, self.g, self.bl, self.A, self.bu, self.l, self.u
    
    def get_interior_point_problem(self):
        return self.H, self.g, self.A_eq, self.b_eq, self.C, self.d