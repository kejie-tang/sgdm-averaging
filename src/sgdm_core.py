"""
sgdm_core.py
============

Shared building blocks for reproducing the numerical experiments in

    "Acceleration of stochastic gradient descent with momentum by averaging:
     finite-sample rates and asymptotic normality"

This module collects the data-generation routines, the SGD / SGDM / averaged
SGDM iterates and the helper quantities (condition number, adaptive momentum
weight, asymptotic covariance matrix) that are reused across the experiment
scripts.  Keeping them in one place guarantees that every experiment uses
*exactly* the same conventions as the paper.

Conventions
-----------
Quadratic loss (Eq. (eq:quadratic) of the paper):
    f_i(x) = 1/2 x^T A_i x + b_i^T x,      A_i = rho * V_i^T V_i + c * I_d
with x^* = -(mean_i A_i)^{-1} (mean_i b_i).

SGDM update (Eq. (sgdm) of the paper) with momentum weight gamma:
    m_t = gamma * m_{t-1} + (1 - gamma) g_{eta_t}(x_t)
    x_{t+1} = x_t - alpha * m_t
SGD is the special case gamma = 0.

Polyak (averaged) iterate (Eq. (def:ave) of the paper):
    xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j

The adaptive momentum weight (Eq. (gammaset) of the paper) is
    gamma = ((1 - mu*alpha) / (1 + mu*alpha))**2
where mu is the smallest eigenvalue of the population Hessian A = E[A_i].
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_quadratic_data(n, p, seed, rho=1.0, shift=10.0, normalize=False):
    """Generate a quadratic-loss problem as in Section 6.1 of the paper.

    Parameters
    ----------
    n : int
        Sample size N.
    p : int
        Dimension d.
    seed : int
        Seed for ``np.random`` (set inside, so results are reproducible).
    rho : float
        Scale of the low-rank part of A_i, controls the condition number L/mu.
    shift : float
        Ridge added to each A_i (``+shift * I``).  With rho=1, shift=10 the
        average condition number is L/mu = 35/10, matching Section 6.1.
    normalize : bool
        If True divide ``(rho * V_i^T V_i + shift * I)`` by ``1 + shift`` so
        that the eigenvalues lie in [0, 1]; used by the sensitivity experiment
        (Section 6.2) so that learning rates can be taken over a wider range.

    Returns
    -------
    dict with keys:
        Ai : (n, p, p) individual Hessians
        bi : (n, p, 1) individual linear terms
        A  : (p, p)    population Hessian (mean of Ai)
        b  : (p, 1)    population linear term (mean of bi)
        x_star : (p, 1) minimizer of the population risk
        mu, L  : smallest / largest eigenvalue of A
        initial : (p, 1) starting point x_1 (x^* plus a small perturbation)
    """
    rng = np.random.RandomState(seed)
    V = rng.rand(n, p, p)
    Ai = np.empty((n, p, p))
    for i in range(n):
        Vi = V[i]
        Ai[i] = rho * (Vi @ Vi.T) + shift * np.eye(p)
    if normalize:
        Ai = Ai / (1.0 + shift)
    bi = rng.rand(n, p, 1)

    A = Ai.mean(axis=0)
    b = bi.mean(axis=0)
    x_star = np.linalg.solve(A, -b)

    noise = rng.rand(p, 1)
    noise = noise / np.linalg.norm(noise)
    initial = x_star + noise * 0.1

    ev = np.linalg.eigvalsh(A)
    mu, L = float(np.min(ev)), float(np.max(ev))
    return dict(Ai=Ai, bi=bi, A=A, b=b, x_star=x_star,
                mu=mu, L=L, initial=initial)


def generate_logistic_data(n, p, seed, theta0=None):
    """Generate binary-logistic data as in Section 6.4 of the paper.

    a_i ~ N(0, I_d);  b_i = 1 with prob. sigmoid(theta0^T a_i), 0 otherwise.
    The default theta0 = (1/sqrt(d), ..., 1/sqrt(d)) matches the paper.
    """
    rng = np.random.RandomState(seed)
    if theta0 is None:
        theta0 = np.ones((p, 1)) / np.sqrt(p)
    X = rng.multivariate_normal(np.zeros(p), np.eye(p), size=n)
    thres = 1.0 / (1.0 + np.exp(-(X @ theta0))).reshape(-1)
    y = (thres < rng.random(n)).astype(np.int64)
    return X, y, theta0


# --------------------------------------------------------------------------- #
# Mini-batch schedule
# --------------------------------------------------------------------------- #
def make_batch_schedule(n, batch_size, K, seed, replace=False):
    """Pre-sample the index sequence used by every iteration.

    Returns a flat int array of length ``K * batch_size``; iteration k uses the
    slice ``schedule[k*batch_size:(k+1)*batch_size]``.  Pre-sampling guarantees
    that SGD and SGDM see the *same* batches (variance reduction in the plots).
    """
    rng = np.random.RandomState(seed)
    if replace:
        return rng.randint(n, size=K * batch_size)
    idx = np.empty(K * batch_size, dtype=np.int64)
    for k in range(K):
        idx[k * batch_size:(k + 1) * batch_size] = rng.choice(
            n, batch_size, replace=False)
    return idx


# --------------------------------------------------------------------------- #
# Iterates
# --------------------------------------------------------------------------- #
def run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, K):
    """Last-iterate SGD. Returns the (K, p, 1) trajectory."""
    p = initial.shape[0]
    x = initial.copy()
    traj = np.empty((K, p, 1))
    for k in range(K):
        i = schedule[k * batch_size:(k + 1) * batch_size]
        A_batch = Ai[i].mean(axis=0)
        b_batch = bi[i].mean(axis=0)
        traj[k] = x
        x = x - alpha * (A_batch @ x + b_batch)
    return traj


def run_sgdm(Ai, bi, x_star, initial, batch_size, schedule, alpha, gamma, K):
    """Last-iterate SGDM. Returns the (K, p, 1) trajectory."""
    p = initial.shape[0]
    x = initial.copy()
    m = np.zeros((p, 1))
    traj = np.empty((K, p, 1))
    for k in range(K):
        i = schedule[k * batch_size:(k + 1) * batch_size]
        A_batch = Ai[i].mean(axis=0)
        b_batch = bi[i].mean(axis=0)
        traj[k] = x
        m = gamma * m + (1.0 - gamma) * (A_batch @ x + b_batch)
        x = x - alpha * m
    return traj


def averaged_errors(traj, x_star, n0, T):
    """|| mean_{j=n0+1}^{n0+t} x_j - x^* || for t = 1..T.

    ``traj`` is the (K, p, 1) last-iterate trajectory, ``n0`` the burn-in and
    ``T`` the number of averaged increments to report.
    """
    T = min(T, traj.shape[0] - n0)
    out = np.empty(T)
    run = np.zeros_like(x_star)
    for t in range(1, T + 1):
        run = run + traj[n0 + t - 1]
        out[t - 1] = np.linalg.norm(run / t - x_star)
    return out


def last_iterate_errors(traj, x_star):
    """||x_t - x^*|| for every t (traj is (K, p, 1))."""
    return np.linalg.norm(traj - x_star[None, :, :], axis=(1, 2))


# --------------------------------------------------------------------------- #
# Helper quantities
# --------------------------------------------------------------------------- #
def adaptive_gamma(mu, alpha):
    """Adaptive momentum weight gamma = ((1-mu*alpha)/(1+mu*alpha))^2."""
    return ((1.0 - mu * alpha) / (1.0 + mu * alpha)) ** 2


def asymptotic_covariance(Ai, bi, x_star, A, batch_size):
    """Covariance of the averaged SGD/SGDM limit (Corollary 3 / Eq. (Sigma)).

        Sigma^{-1} Omega Sigma^{-1},  with
        Omega = (1/B) * (1/N) sum_i (A_i x^* + b_i)(A_i x^* + b_i)^T.
    """
    n = Ai.shape[0]
    A_inv = np.linalg.inv(A)
    Omega = np.zeros_like(A)
    for i in range(n):
        delta = Ai[i] @ x_star + bi[i]
        Omega += (delta @ delta.T) / n
    Omega = Omega / batch_size
    return A_inv @ Omega @ A_inv


def clt_projection(Ai, bi, x_star, initial, batch_size, schedule, alpha, K, n0,
                   cov, gamma=0.0, phi=None, seed=0):
    """One replication of the projected CLT statistic Z (Eq. (def:z)).

    Returns phi^T (xbar_{n0:n} - x^*) * sqrt(n-n0) / sqrt(phi^T Sigma phi),
    which is asymptotically N(0,1) by Corollary 3.
    """
    p = initial.shape[0]
    if phi is None:
        rng = np.random.RandomState(seed + 12345)
        phi = rng.rand(p, 1)
        phi = phi / np.linalg.norm(phi)
    if gamma == 0.0:
        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, K)
    else:
        traj = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                        alpha, gamma, K)
    x_age = traj[n0:].mean(axis=0)
    num = float((phi.T @ (x_age - x_star)).sum()) * np.sqrt(K - n0)
    den = np.sqrt(float((phi.T @ cov @ phi).sum()))
    return num / den, phi
