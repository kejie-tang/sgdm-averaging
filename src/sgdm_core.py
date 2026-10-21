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

Notation
--------
The identifiers below are the paper's symbols with the paper's meaning, so that
a line of code can be read against the text directly:

    paper                    code                     meaning
    -----------------------------------------------------------------------
    N                        ``N``                    training-set size
    d                        ``d``                    dimension of x
    B                        ``batch_size``           mini-batch size
    n                        ``n``                    number of iterations
    n0                       ``n0``                   averaging burn-in
    alpha                    ``alpha``                learning rate
    gamma                    ``gamma``                momentum weight
    Sigma / A                ``A``                    population Hessian E[A_i]
    A_i, b_i                 ``Ai``, ``bi``           per-sample loss parameters
    x*                       ``x_star``               minimizer of the risk
    mu, L                    ``mu``, ``L``            smallest/largest eigenvalue
    x_t                      ``traj[t]``              last iterate at time t
    xbar_{n0+t}              ``averaged_errors(...)`` Polyak-averaged iterate
    Z                        ``clt_projection(...)`` projected CLT statistic

Conventions
-----------
Quadratic loss (Eq. (eq:quadratic) of the paper):
    f_i(x) = 1/2 x^T A_i x + b_i^T x,      A_i = rho * V_i V_i^T + shift * I_d
with x^* = -(mean_i A_i)^{-1} (mean_i b_i).  The paper writes the quadratic
form as V_i^T V_i; for the square V_i drawn here the two products have the same
eigenvalues but are different matrices, and the release keeps V_i V_i^T, which
is what the authors' notebooks compute (so the figures reproduce them).

SGDM update (Eq. (sgdm) of the paper) with momentum weight gamma:
    m_t = gamma * m_{t-1} + (1 - gamma) g_eta_t(x_t)
    x_{t+1} = x_t - alpha * m_t
SGD is the special case gamma = 0.

Polyak (averaged) iterate (Eq. (def:ave) of the paper):
    xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j

The adaptive momentum weight (Eq. (gammaset) of the paper) is
    gamma = ((1 - mu*alpha) / (1 + mu*alpha))**2
where mu is the smallest eigenvalue of the population Hessian A = E[A_i].

Indexing
--------
``traj`` is zero-based: ``traj[k]`` is the paper's x_{k+1}, so ``traj[0]`` is
the initial point x_1.  The last-iterate panels plot ``traj[0:n]`` on a
``t = 0, ..., n-1`` axis, which is the initial point followed by the first
``n - 1`` updates.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def draw_quadratic_problem(N, d, rng):
    """Draw the raw random tensors of the Section 5.1 problem, in the authors'
    order: ``V`` (N, d, d), then ``b_i`` (N, d, 1), then the noise direction
    ``u`` (d, 1) of the starting point.

    The authors' notebooks take all three from one unbroken ``np.random``
    stream and then go on drawing the mini-batch indices from the *same*
    stream.  Keeping the draws in a helper of their own is what lets a caller
    share that single stream with :func:`make_batch_schedule`; see
    :func:`generate_quadratic_data`.
    """
    V = rng.rand(N, d, d)
    bi = rng.rand(N, d, 1)
    u = rng.rand(d, 1)
    return V, bi, u


def build_quadratic_data(V, bi, u, rho=1.0, shift=10.0, norm_const=None):
    """Assemble a quadratic-loss problem from the draws of
    :func:`draw_quadratic_problem`.

    Parameters
    ----------
    V : (N, d, d) array
        The draws that form the individual Hessians.
    bi : (N, d, 1) array
        The draws that form the individual linear terms.
    u : (d, 1) array
        The unnormalised direction of the initial perturbation.
    rho : float
        Scale of the low-rank part of A_i, controls the condition number L/mu.
    shift : float
        Ridge added to each A_i (``+shift * I``).  With rho=1, shift=10 the
        average condition number is L/mu = 3.31, close to the 3.5 quoted in
        Section 5.1.
    norm_const : float or None
        If given, divide ``(rho * V_i V_i^T + shift * I)`` by this constant
        before forming A, b and x^*.  The sensitivity experiment (Section 5.2)
        uses ``rho=1, shift=1.03, norm_const=1.04``, i.e. the authors'
        ``(V_i V_i^T + 1.03 I)/1.04``.

    Returns
    -------
    dict with keys:
        Ai : (N, d, d) individual Hessians
        bi : (N, d, 1) individual linear terms
        A  : (d, d)    population Hessian (mean of Ai)
        b  : (d, 1)    population linear term (mean of bi)
        x_star : (d, 1) minimizer of the population risk
        mu, L  : smallest / largest eigenvalue of A
        initial : (d, 1) starting point x_1 (x^* plus a small perturbation)
    """
    N, d = V.shape[0], V.shape[1]
    Ai = np.empty((N, d, d))
    for i in range(N):
        Vi = V[i]
        Ai[i] = rho * (Vi @ Vi.T) + shift * np.eye(d)
    if norm_const is not None:
        Ai = Ai / float(norm_const)

    A = Ai.mean(axis=0)
    b = bi.mean(axis=0)
    x_star = np.linalg.solve(A, -b)

    u = u / np.linalg.norm(u)
    initial = x_star + u * 0.1

    ev = np.linalg.eigvalsh(A)
    mu, L = float(np.min(ev)), float(np.max(ev))
    return dict(Ai=Ai, bi=bi, A=A, b=b, x_star=x_star,
                mu=mu, L=L, initial=initial)


def generate_quadratic_data(N, d, seed, rho=1.0, shift=10.0, norm_const=None,
                            rng=None):
    """Generate a quadratic-loss problem as in Section 5.1 of the paper.

    This is :func:`draw_quadratic_problem` followed by
    :func:`build_quadratic_data`.  ``rho``, ``shift`` and ``norm_const`` are
    forwarded to the latter; the random draws depend on the seed (and on
    ``rho`` only through ``Ai``, so every ``rho`` describes the *same*
    realisation of ``V``, ``b_i`` and the starting point).

    ``rng`` may be supplied to draw from a caller-owned stream; when it is
    given the three tensors are taken from it, so the caller can keep using the
    same stream for the mini-batch schedule -- which is exactly what the
    authors' notebooks do.  The experiment scripts therefore call
    :func:`draw_quadratic_problem` and :func:`make_batch_schedule` directly
    rather than this wrapper.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    V, bi, u = draw_quadratic_problem(N, d, rng)
    return build_quadratic_data(V, bi, u, rho=rho, shift=shift,
                                norm_const=norm_const)


def generate_logistic_data(N, d, seed, theta0=None, rng=None):
    """Generate binary-logistic data as in Section EC.7.1 of the paper.

    a_i ~ N(0, I_d);  b_i = 1 with probability sigmoid(-theta0^T a_i) =
    1 - sigmoid(theta0^T a_i), and 0 otherwise.  The default theta0 =
    (1/sqrt(d), ..., 1/sqrt(d)) matches the paper.

    ``rng`` may be supplied to draw from a caller-owned random stream.  The
    authors generate the features, the labels and the mini-batch schedule from
    one unbroken ``np.random`` stream per replication; the logistic experiment
    passes the same ``RandomState`` here and to :func:`make_batch_schedule` so
    that the realisation matches theirs exactly.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    if theta0 is None:
        theta0 = np.ones((d, 1)) / np.sqrt(d)
    X = rng.multivariate_normal(np.zeros(d), np.eye(d), size=N)
    thres = 1.0 / (1.0 + np.exp(-(X @ theta0))).reshape(-1)
    y = (thres < rng.random(N)).astype(np.int64)
    return X, y, theta0


# --------------------------------------------------------------------------- #
# Mini-batch schedule
# --------------------------------------------------------------------------- #
def make_batch_schedule(N, batch_size, n, seed, replace=False, rng=None):
    """Pre-sample the index sequence used by every iteration.

    ``N`` is the paper's training-set size and ``n`` its number of iterations,
    so the returned flat int array has length ``n * batch_size``; iteration k
    uses the slice ``schedule[k*batch_size:(k+1)*batch_size]``.  Pre-sampling
    guarantees that SGD and SGDM see the *same* batches (variance reduction in
    the plots).

    ``rng`` may be supplied to draw from a caller-owned stream; see
    :func:`generate_logistic_data`.
    """
    if rng is None:
        rng = np.random.RandomState(seed)
    if replace:
        return rng.randint(N, size=n * batch_size)
    idx = np.empty(n * batch_size, dtype=np.int64)
    for k in range(n):
        idx[k * batch_size:(k + 1) * batch_size] = rng.choice(
            N, batch_size, replace=False)
    return idx


def batch_gradient_sequence(Ai, bi, batch_size, schedule, n):
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
    (A_seq, b_seq) : ((n, d, d), (n, d, 1))
    """
    d = Ai.shape[1]
    A_seq = np.empty((n, d, d))
    b_seq = np.empty((n, d, 1))
    for k in range(n):
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


def _weights(gamma, k):
    """Return ``(gamma_k, 1 - gamma_k)``, the momentum weights at iteration
    ``k``.

    ``gamma`` may be a scalar (constant momentum weight) or a sequence of
    length at least ``k + 1`` (a momentum weight that changes over time).
    """
    if np.isscalar(gamma):
        return gamma, 1.0 - gamma
    g = gamma[k]
    return g, 1.0 - g


def run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, n,
            A_seq=None, b_seq=None):
    """Last-iterate SGD. Returns the (n, d, 1) trajectory.

    ``alpha`` is either a scalar or a length-n per-iteration schedule.

    ``A_seq``/``b_seq`` are the optional output of
    :func:`batch_gradient_sequence`; when supplied they are replayed instead of
    re-averaging the mini-batches (identical results, much faster sweeps).
    """
    d = initial.shape[0]
    x = initial.copy()
    traj = np.empty((n, d, 1))
    for k in range(n):
        if A_seq is None:
            i = schedule[k * batch_size:(k + 1) * batch_size]
            A_batch = Ai[i].mean(axis=0)
            b_batch = bi[i].mean(axis=0)
        else:
            A_batch, b_batch = A_seq[k], b_seq[k]
        traj[k] = x
        x = x - _step_size(alpha, k) * (A_batch @ x + b_batch)
    return traj


def run_sgdm(Ai, bi, x_star, initial, batch_size, schedule, alpha, gamma, n,
             A_seq=None, b_seq=None):
    """Last-iterate SGDM. Returns the (n, d, 1) trajectory.

    ``alpha`` and ``gamma`` are each either a scalar or a length-n
    per-iteration schedule.
    ``A_seq``/``b_seq`` behave as in :func:`run_sgd`.
    """
    d = initial.shape[0]
    x = initial.copy()
    m = np.zeros((d, 1))
    traj = np.empty((n, d, 1))
    for k in range(n):
        if A_seq is None:
            i = schedule[k * batch_size:(k + 1) * batch_size]
            A_batch = Ai[i].mean(axis=0)
            b_batch = bi[i].mean(axis=0)
        else:
            A_batch, b_batch = A_seq[k], b_seq[k]
        traj[k] = x
        g, one_minus_g = _weights(gamma, k)
        m = g * m + one_minus_g * (A_batch @ x + b_batch)
        x = x - _step_size(alpha, k) * m
    return traj


def decaying_schedule(n0, n, alpha0):
    """Decaying learning rate of Figure 7.

        alpha_t = alpha0 * sqrt(n0 / max(n0, t)),   t = 1, ..., n

    i.e. alpha0 for t <= n0 and alpha0 * sqrt(n0/t) afterwards, which is the
    paper's alpha_t = alpha0 * (max{n0, t})^{-1/2} up to the constant sqrt(n0).
    """
    t = np.arange(1, n + 1, dtype=float)
    return alpha0 * np.sqrt(n0 / np.maximum(n0, t))


def averaged_errors(traj, x_star, n0, T):
    """|| xbar_{n0+t} - x^* || for t = 1..T, the averaged-iterate curves.

    ``traj`` is the (n, d, 1) last-iterate trajectory with ``traj[k] = x_{k+1}``
    (see the module docstring), ``n0`` the burn-in and ``T`` the number of
    averaged increments to report.  At most n - n0 - 1 points are produced.

    The running sum starts at ``traj[n0 + 1]``.  That is the convention the
    authors' notebooks use -- they accumulate the iterate *after* the update
    has been applied, and divide by ``k - n0`` in the same iteration -- and it
    is what the published averaged panels are drawn from.  It equals the
    definition xbar_{n0+t} = (1/t) sum_{j=n0+2}^{n0+t+1} x_j, i.e. the same
    average shifted by one iterate; see :func:`averaged_errors_shifted` for the
    variant whose sum starts at x_{n0+1}.
    """
    if not 0 <= n0 < traj.shape[0]:
        raise ValueError(f"n0={n0} is outside a trajectory of length "
                         f"{traj.shape[0]}")
    T = min(T, traj.shape[0] - n0 - 1)
    out = np.empty(T)
    run = np.zeros_like(x_star)
    for t in range(1, T + 1):
        run = run + traj[n0 + t]
        out[t - 1] = np.linalg.norm(run / t - x_star)
    return out


def averaged_errors_shifted(traj, x_star, n0, T):
    """|| xbar_{n0+t} - x^* || with the sum starting at x_{n0+1}.

    Same as :func:`averaged_errors` but for the literal reading of
    xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j: the running sum starts at
    ``traj[n0]`` and at most n - n0 points are produced.  Not used by the
    released figures, which follow the authors' notebooks; kept so that the two
    conventions can be compared without editing the shared helper.
    """
    if not 0 <= n0 < traj.shape[0]:
        raise ValueError(f"n0={n0} is outside a trajectory of length "
                         f"{traj.shape[0]}")
    T = min(T, traj.shape[0] - n0)
    out = np.empty(T)
    run = np.zeros_like(x_star)
    for t in range(1, T + 1):
        run = run + traj[n0 + t - 1]
        out[t - 1] = np.linalg.norm(run / t - x_star)
    return out


def last_iterate_errors(traj, x_star):
    """||x_t - x^*|| for every t (traj is (n, d, 1))."""
    return np.linalg.norm(traj - x_star[None, :, :], axis=(1, 2))


# --------------------------------------------------------------------------- #
# Helper quantities
# --------------------------------------------------------------------------- #
def adaptive_gamma(mu, alpha):
    """Adaptive momentum weight gamma = ((1-mu*alpha)/(1+mu*alpha))^2."""
    return ((1.0 - mu * alpha) / (1.0 + mu * alpha)) ** 2


def asymptotic_covariance(Ai, bi, x_star, A, batch_size, replace=True, N=None):
    """Covariance of the averaged SGD/SGDM limit (Corollary 3 / Eq. (Sigma)).

        Sigma^{-1} Omega Sigma^{-1},  with
        Omega = (1/B) * (1/N) sum_i (A_i x^* + b_i)(A_i x^* + b_i)^T.

    ``Omega / B`` is the covariance of the mean gradient when the mini-batch is
    drawn **with** replacement (``replace=True``), which is the setting of
    Corollary 3 and of Figure 8.  When the mini-batch is drawn **without**
    replacement -- ``make_batch_schedule(..., replace=False)``, used by the
    coverage sweep of Figure 12 -- the finite-population factor
    ``(N - B)/(N - 1)`` applies:

        Omega_B = (N - B)/(N - 1) * Omega_1 / B,

    which is 0.8 at the release's ``B = 0.2 N`` (so 0.894 times the sd).  Pass
    ``replace=False`` for such a sweep; ``N`` is only needed to keep the default
    (``len(Ai)``) explicit.
    """
    N = Ai.shape[0] if N is None else int(N)
    A_inv = np.linalg.inv(A)
    Omega = np.zeros_like(A)
    for i in range(N):
        delta = Ai[i] @ x_star + bi[i]
        Omega += (delta @ delta.T) / N
    if replace:
        Omega = Omega / batch_size
    else:
        if batch_size >= N:
            # a full pass has no sampling noise at all
            Omega = np.zeros_like(Omega)
        else:
            Omega = Omega / batch_size * (N - batch_size) / (N - 1)
    return A_inv @ Omega @ A_inv


def clt_projection(Ai, bi, x_star, initial, batch_size, schedule, alpha, n, n0,
                   cov, gamma=0.0, phi=None, seed=0, A_seq=None, b_seq=None):
    """One replication of the projected CLT statistic Z (Eq. (def:z)).

    Returns phi^T (xbar_{n0:n} - x^*) * sqrt(n - n0) / sqrt(phi^T Sigma phi),
    which is asymptotically N(0,1) by Corollary 3.

    ``A_seq``/``b_seq`` are forwarded to :func:`run_sgd` / :func:`run_sgdm`;
    supplying them lets a whole sweep over (gamma, alpha) share one pass over
    the data.
    """
    d = initial.shape[0]
    if phi is None:
        rng = np.random.RandomState(seed + 12345)
        phi = rng.rand(d, 1)
        phi = phi / np.linalg.norm(phi)
    if gamma == 0.0:
        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule, alpha, n,
                       A_seq=A_seq, b_seq=b_seq)
    else:
        traj = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                        alpha, gamma, n, A_seq=A_seq, b_seq=b_seq)
    x_age = traj[n0:].mean(axis=0)
    num = float((phi.T @ (x_age - x_star)).sum()) * np.sqrt(n - n0)
    den = np.sqrt(float((phi.T @ cov @ phi).sum()))
    return num / den, phi
