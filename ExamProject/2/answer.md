# Quadratic Programming

## a)

For problem (2)

$$
\begin{aligned}
\min_{x\in\mathbb{R}^n} \quad & \phi(x)=\frac{1}{2}x' Hx+g' x \\
\mathrm{s.t.} \quad & b_l \le A' x \le b_u, \\
& x_l \le x \le x_u,
\end{aligned}
$$

introduce nonnegative multipliers for each inequality block:

- $\lambda_l \ge 0$ for $A' x-b_l \ge 0$,
- $\lambda_u \ge 0$ for $b_u-A' x \ge 0$,
- $z_l \ge 0$ for $x-x_l \ge 0$,
- $z_u \ge 0$ for $x_u-x \ge 0$.

Then the Lagrangian is

$$
\mathcal{L}(x,\lambda_l,\lambda_u,z_l,z_u)
=\frac{1}{2}x' Hx+g' x
-\lambda_l'(A' x-b_l)
-\lambda_u'(b_u-A' x)
-z_l'(x-x_l)
-z_u'(x_u-x).
$$

Equivalent expanded form:

$$
\mathcal{L}
=\frac{1}{2}x' Hx+g' x
-\lambda_l' A' x+\lambda_l' b_l
-\lambda_u' b_u+\lambda_u' A' x
-z_l' x+z_l' x_l
-z_u' x_u+z_u' x.
$$

## b)

Assume the QP is convex ($H\succeq 0$), and let a primal-dual point be

$$
(x,\lambda_l,\lambda_u,z_l,z_u).
$$

The necessary optimality conditions (KKT) are:

1. Stationarity

$$
\nabla_x\mathcal{L}=Hx+g-A\lambda_l+A\lambda_u-z_l+z_u=0.
$$

2. Primal feasibility

$$
b_l \le A' x \le b_u, \qquad x_l \le x \le x_u.
$$

3. Dual feasibility

$$
\lambda_l\ge 0,\;\lambda_u\ge 0,\;z_l\ge 0,\;z_u\ge 0.
$$

4. Complementary slackness

$$
\lambda_l \odot (A' x-b_l)=0,
$$

$$
\lambda_u \odot (b_u-A' x)=0,
$$

$$
z_l \odot (x-x_l)=0,
$$

$$
z_u \odot (x_u-x)=0,
$$

where $\odot$ is elementwise product.

