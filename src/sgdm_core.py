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
def generate_quadratic_data(n, p, seed, rho=1.0, shift=10.0, norm_const=None):
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
    norm_const : float or None
        If given, divide ``(rho * V_i^T V_i + shift * I)`` by this constant
        before forming A, b and x^*.  The sensitivity experiment (Section 6.2)
        uses ``rho=1, shift=1.03, norm_const=1.04``, i.e. the authors'
        ``(V_i^T V_i + 1.03 I)/1.04``.

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
    if norm_const is not None:
        Ai = Ai / float(norm_const)
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


def generate_logistic_data(n, p, seed, theta0=None, rng=None):
    """Generate binary-logistic data as in Section 6.4 of the paper.

    a_i ~ N(0, I_d);  b_i = 1 with prob. sigmoid(theta0^T a_i), 0 otherwise.
    The default theta0 = (1/sqrt(d), ..., 1/sqrt(d)) matches the paper.

    Note that the authors' notebook draws the label as
    ``y = (sigmoid(theta0^T a) < U(0, 1))``, i.e. with probability
    ``1 - sigmoid(theta0^T a) = sigmoid(-theta0^T a)``; that is what is
    implemented here, since it is what produced Figures EC.1-EC.3.  The two
    conventions differ only by the global sign of the population solution, and
    since a_i is centred Gaussian the marginal law of the error curves is the
    same either way (the adaptive weight is unchanged: the realised mean over
    200 replications is 0.75 under both).

    ``rng`` may be supplied to draw from a caller-owned random stream.  The
    authors generate the features, the labels and the mini-batch schedule from
    one unbroken ``np.random`` stream per replication; the logistic experiment
    passes the same ``RandomState`` here and to :func:`make_batch_schedule` so
    that the realisation matches theirs exactly.
    """
    if rng is None:
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
def make_batch_schedule(n, batch_size, K, seed, replace=False, rng=None):
    """Pre-sample the index sequence used by every iteration.

    Returns a flat int array of length ``K * batch_size``; iteration k uses the
    slice ``schedule[k*batch_size:(k+1)*batch_size]``.  Pre-sampling guarantees
    that SGD and SGDM see the *same* batches (variance reduction in the plots).

    ``rng`` may be supplied to draw from a caller-owned stream; see
    :func:`generate_logistic_data`.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    if replace:
        return rng.randint(n, size=K * batch_size)
    idx = np.empty(K * batch_size, dtype=np.int64)
    for k in range(K):
        idx[k * batch_size:(k + 1) * batch_size] = rng.choice(
            n, batch_size, replace=False)
    return idx


def batch_gradient_sequence(Ai, bi, batch_size, schedule, K):
    """Pre-compute the mean mini-batch gradient at every iteration.

    Iteration k sees the batch ``schedule[k*batch_size:(k+1)*batch_size]``, so
    the mini-batch gradient is ``A_batch @ x + b_batch`` with
    ``A_batch = mean_i A_i``, ``b_batch = mean_i b_i`` over that batch.  Because
    the batches do not depend on the learning rate or the momentum weight, the
    same pair of arrays can be replayed by *every* configuration in a sweep
    (as in Figures 2-4, 6, 7, 9-11).  Doing so turns a whole sweep into a single
    pass over the data and is numerically identical to recomputing them.

    Returns
    -------
    (A_seq, b_seq) : ((K, p, p), (K, p, 1))
    """
    p = Ai.shape[1]
    A_seq = np.empty((K, p, p))
    b_seq = np.empty((K, p, 1))
    for k in range(K):
        i = schedule[k * batch_size:(k + 1) * batch_size]
        A_seq[k] = Ai[i].mean(axis=0)
        b_seq[k] = bi[i].mean(axis=0)
    return A_seq, b_seq


# --------------------------------------------------------------------------- #
# Iterates
# --------------------------------------------------------------------------- #
def _step_size(alpha, k):
    """Return the step size at iteration ``k``.

    ``alpha`` may be a scalar (constant learning rate) or a sequence of length
    at least ``k + 1`` (a decaying schedule, as in Figure 7).
    """
    if np.isscalar(alpha):
        return alpha
    return alpha[k]


def run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, K,
            A_seq=None, b_seq=None):
    """Last-iterate SGD. Returns the (K, p, 1) trajectory.

    ``alpha`` is either a scalar or a length-K per-iteration schedule.

    ``A_seq``/``b_seq`` are the optional output of
    :func:`batch_gradient_sequence`; when supplied they are replayed instead of
    re-averaging the mini-batches (identical results, much faster sweeps).
    """
    p = initial.shape[0]
    x = initial.copy()
    traj = np.empty((K, p, 1))
    for k in range(K):
        if A_seq is None:
            i = schedule[k * batch_size:(k + 1) * batch_size]
            A_batch = Ai[i].mean(axis=0)
            b_batch = bi[i].mean(axis=0)
        else:
            A_batch, b_batch = A_seq[k], b_seq[k]
        traj[k] = x
        x = x - _step_size(alpha, k) * (A_batch @ x + b_batch)
    return traj


def run_sgdm(Ai, bi, x_star, initial, batch_size, schedule, alpha, gamma, K,
             A_seq=None, b_seq=None):
    """Last-iterate SGDM. Returns the (K, p, 1) trajectory.

    ``alpha`` is either a scalar or a length-K per-iteration schedule.
    ``A_seq``/``b_seq`` behave as in :func:`run_sgd`.
    """
    p = initial.shape[0]
    x = initial.copy()
    m = np.zeros((p, 1))
    traj = np.empty((K, p, 1))
    for k in range(K):
        if A_seq is None:
            i = schedule[k * batch_size:(k + 1) * batch_size]
            A_batch = Ai[i].mean(axis=0)
            b_batch = bi[i].mean(axis=0)
        else:
            A_batch, b_batch = A_seq[k], b_seq[k]
        traj[k] = x
        m = gamma * m + (1.0 - gamma) * (A_batch @ x + b_batch)
        x = x - _step_size(alpha, k) * m
    return traj


def decaying_schedule(n0, K, alpha0):
    """Decaying learning rate of Figure 7.

        alpha_t = alpha0 * sqrt(n0 / max(n0, t)),   t = 1, ..., K

    i.e. alpha0 for t <= n0 and alpha0 * sqrt(n0/t) afterwards, which is the
    paper's alpha_t = alpha0 * (max{n0, t})^{-1/2} up to the constant sqrt(n0).
    """
    t = np.arange(1, K + 1, dtype=float)
    return alpha0 * np.sqrt(n0 / np.maximum(n0, t))


def averaged_errors(traj, x_star, n0, T):
    """|| mean_{j=n0+1}^{n0+t} x_j - x^* || for t = 1..T.

    ``traj`` is the (K, p, 1) last-iterate trajectory with ``traj[k] = x_k``,
    ``n0`` the burn-in and ``T`` the number of averaged increments to report.
    The first reported point is x_{n0+1}, as in the paper's definition
    xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j, so at most K - n0 - 1 points
    can be produced.
    """
    T = min(T, traj.shape[0] - n0 - 1)
    out = np.empty(T)
    run = np.zeros_like(x_star)
    for t in range(1, T + 1):
        run = run + traj[n0 + t]
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
                   cov, gamma=0.0, phi=None, seed=0, A_seq=None, b_seq=None):
    """One replication of the projected CLT statistic Z (Eq. (def:z)).

    Returns phi^T (xbar_{n0:n} - x^*) * sqrt(n-n0) / sqrt(phi^T Sigma phi),
    which is asymptotically N(0,1) by Corollary 3.

    ``A_seq``/``b_seq`` are forwarded to :func:`run_sgd` / :func:`run_sgdm`;
    supplying them lets a whole sweep over (gamma, alpha) share one pass over
    the data.
    """
    p = initial.shape[0]
    if phi is None:
        rng = np.random.RandomState(seed + 12345)
        phi = rng.rand(p, 1)
        phi = phi / np.linalg.norm(phi)
    if gamma == 0.0:
        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, K,
                       A_seq=A_seq, b_seq=b_seq)
    else:
        traj = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                        alpha, gamma, K, A_seq=A_seq, b_seq=b_seq)
    x_age = traj[n0:].mean(axis=0)
    num = float((phi.T @ (x_age - x_star)).sum()) * np.sqrt(K - n0)
    den = np.sqrt(float((phi.T @ cov @ phi).sum()))
    return num / den, phi
