# Equality Constrained Convex QP

## 1.1 Lagrangian Function

Given

$$
\begin{aligned}
\min_{x} \quad & \phi = \frac{1}{2}x' H x + g' x \quad (1a) \\
\mathrm{s.t.} \quad & A' x = b \quad (1b),
\end{aligned}
$$


Introduce multiplier $\lambda$ for the equality constraints. The Lagrangian is

$$
\mathcal{L}(x,\lambda) = \frac{1}{2}x' H x + g' x + \lambda'\left(A' x-b\right).
$$

Equivalent expanded form:

$$
\mathcal{L}(x,\lambda) = \frac{1}{2}x' H x + g' x + \lambda' A' x - \lambda' b.
$$

Note: The equivalent sign convention
$\mathcal{L}(x,\lambda)=\frac{1}{2}x' H x+g' x-\lambda'(A' x-b)$
is obtained by redefining $\lambda \leftarrow -\lambda$. Flipping the sign of $\lambda$ flips the sign of the multipliers.

## 1.2 First-Order Necessary Optimality Conditions

From

$$
\mathcal{L}(x,\lambda)=\frac{1}{2}x' H x+g' x+\lambda'(A'x-b),
$$

the first-order necessary optimality conditions are:

$$
\nabla_x \mathcal{L}(x,\lambda)=Hx+g+A\lambda=0,
$$

$$
\nabla_\lambda \mathcal{L}(x,\lambda)=A'x-b=0.
$$

So a primal-dual optimum $(x^\star,\lambda^\star)$ must satisfy

$$
\begin{aligned}
Hx^\star + g + A\lambda^\star &= 0, \\
A'x^\star - b &= 0.
\end{aligned}
$$

Equivalent KKT linear system:

$$
\begin{bmatrix}
H & A \\
A' & 0
\end{bmatrix}
\begin{bmatrix}
x^\star \\
\lambda^\star
\end{bmatrix}
=
\begin{bmatrix}
-g \\
b
\end{bmatrix}.
$$

Are they also sufficient?

Yes. For this problem they are sufficient (for global optimality), because:

1. The objective $\frac{1}{2}x'Hx+g'x$ is convex when $H\succeq 0$.
2. The constraints $A'x=b$ are affine, so the feasible set is convex.
3. For a convex problem with differentiable objective and affine equality constraints, any point satisfying the KKT conditions is a global minimizer.

Hence, under feasibility, the above first-order conditions are both necessary and sufficient for optimality. If additionally $H\succ 0$ on the feasible subspace, the primal optimizer $x^\star$ is unique.
