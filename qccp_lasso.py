import numpy as np
from sklearn.datasets import make_spd_matrix

def make_sample_dataset(n):
    Q = make_spd_matrix(n)
    c = np.random.uniform(-10, 10, size=n)
    lam = np.exp(np.random.uniform(-2, 2))
    return Q, c, lam

def objective_function(x, Q, c, lam):
    return 0.5 * np.dot(x, Q @ x) - np.dot(c, x) + lam * np.linalg.norm(x, 1)

def armijo_ls(x, dx, grad, Q, c, lam, alpha_init=1.0, beta=0.5, sigma=1e-4):
    alpha = alpha_init
    grad_dot = np.dot(grad, dx)
    while True:
        new_x = x + alpha * dx
        # Check Armijo condition
        if objective_function(new_x, Q, c, lam) <= \
           objective_function(x, Q, c, lam) + sigma * alpha * grad_dot:
            break  # Armijo condition is satisfied
        alpha *= beta  # Reduce alpha by beta
    return alpha

def subgrad_lasso(x, Q, c, lam):
    g = np.zeros_like(x)
    resid = Q @ x - c
    defaultgrad = resid + lam * np.sign(x)
    g = np.where((x == 0) & (np.abs(resid) <= lam), 0, defaultgrad)
    return g

def sgs_decompose(Q: np.ndarray):
    return np.tril(Q), Q.diagonal()

def solve_diagonal(resid, lam, d):
    if resid > lam:
        return (resid - lam) / d
    if resid < -lam:
        return (resid + lam) / d
    return 0

def iter_qccp_lasso(x, Q, c, lam):
    n = len(c)
    L, dia = sgs_decompose(Q)
    xhalf = c - L.T @ x + dia * x
    for i in range(n):
        # solve for diagonal
        xhalf[i] = solve_diagonal(xhalf[i], lam, dia[i])
        if i == n - 1:
            continue
        xhalf[i+1:] -= Q[i+1:, i] * xhalf[i]
    xfull = c - L @ xhalf + dia * xhalf
    for i in range(n - 1, -1, -1):
        xfull[i] = solve_diagonal(xfull[i], lam, dia[i])
        if i == 0:
            continue
        xfull[:i] -= Q[:i, i] * xfull[i]
    return xfull
