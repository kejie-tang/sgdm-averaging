"""
Figures EC.1-EC.3: SGD / SGDM / averaged SGD / averaged SGDM on the logistic loss.

This experiment is reported in the paper's electronic companion (Section EC.7.1).

Setting
-------
N = 20,000, d = 10, a_i ~ N(0, I_d), b_i ~ Bernoulli(sigmoid(theta0^T a_i)) with
theta0 = (1,...,1)/sqrt(d).  Batch size B = 0.2 N, learning rate alpha = 0.5,
gamma in {0, 0.3, 0.5, 0.7, 0.8, 0.9} plus the adaptive weight
gamma = ((1-mu*alpha)/(1+mu*alpha))^2 computed at the population Hessian
evaluated at the full-batch minimizer x^* (average adaptive weight ~ 0.75).

The features and the labels come from :func:`sgdm_core.generate_logistic_data`.

Panel (a) "small gamma": SGD, SGDM-0.3/0.5/0.7, SGDM-adap.
Panel (b) "large gamma": SGDM-adap, SGDM-0.8, SGDM-0.9 (no SGD curve, as in
the published panels).

The full-batch gradient descent used to locate x^* is the same procedure as in
the paper (gamma = 0, 10*n iterations, alpha = 0.5).

Update rule
-----------
The authors' recursion is

    m_t = gamma m_{t-1} + (1 - gamma) grad(x_t),   x_{t+1} = x_t - alpha m_t,

with the **first** momentum set to the raw mini-batch gradient, m_1 = grad(x_1),
i.e. *without* the (1 - gamma) factor.

Both panels start at t = 0 with the initial point x_1 = 0 (every curve of
Figure EC.1 begins from the same value, ||x_1 - x^*||^2 = 1.0: the quadratic
form of the minimizer is 1 up to the sampling noise).

Outputs (results/logistic/), one file per paper panel:
    fig_ec1_small.png / fig_ec1_large.png   -> Figure EC.1(a),(b)
    fig_ec2_small.png / fig_ec2_large.png   -> Figure EC.2(a),(b),  n0 = 10
    fig_ec3_small.png / fig_ec3_large.png   -> Figure EC.3(a),(b),  n0 = 40
    logistic_raw.npz

Run:  python src/logistic/logistic_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import generate_logistic_data, make_batch_schedule  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def gradient(theta, X, y):
    """Mean gradient of the negative log-likelihood; returns (d, 1)."""
    theta = theta.reshape(-1)
    n_obs = y.size
    h = sigmoid(X @ theta).reshape(-1)
    y = y.reshape(-1)
    return ((X.T @ (h - y)) / n_obs).reshape(-1, 1)


def hessian(theta, X, y):
    """Population Hessian of the negative log-likelihood; returns (d, d)."""
    n_obs, d = X.shape
    theta = theta.reshape(-1)
    h = sigmoid(X @ theta).reshape(-1)
    H = np.zeros((d, d))
    for i in range(n_obs):
        ai = X[i].reshape(d, 1)
        H += h[i] * (1 - h[i]) * (ai @ ai.T)
    return H / n_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000, help="training-set size N")
    ap.add_argument("--d", type=int, default=10, help="dimension d of x")
    ap.add_argument("--alpha", type=float, default=0.5, help="learning rate alpha")
    ap.add_argument("--num_seed", type=int, default=200, help="number of independent replications")
    ap.add_argument("--batch_frac", type=float, default=0.2, help="mini-batch size as a fraction of N")
    # the paper uses n0 = 10 for Figure EC.2 and n0 = 40 for Figure EC.3
    ap.add_argument("--n0_list", type=int, nargs="+", default=[10, 40], help="averaging burn-in n0, one value per panel")
    ap.add_argument("--ave_len", type=int, default=100, help="number of averaged points reported per n0")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "logistic"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d = args.N, args.d
    batch_size = int(round(args.batch_frac * N))
    ave_len = args.ave_len
    n = max(args.n0_list) + ave_len + 1

    # gamma_set: SGD (0), the fixed weights of panel (a), the adaptive weight
    # of the paper (None is replaced per replication), then panel (b)'s 0.8
    # and 0.9.  The ordering is the authors' and fixes the column layout of
    # err_last / err_ave.
    gamma_set = [0.0, 0.3, 0.5, 0.7, None, 0.8, 0.9]
    n_gammas = len(gamma_set)

    # last-iterate error (Figure EC.1) and averaged error per n0 (Figures EC.2-EC.3)
    # err_last[k] is ||x_{k+1} - x^*||, so entry 0 is the initial point x_1.
    err_last = np.zeros((args.num_seed, n, n_gammas))
    err_ave = {n0: np.zeros((args.num_seed, ave_len - 1, n_gammas))
               for n0 in args.n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[logistic] seed {seed}/{args.num_seed}", flush=True)
        # One unbroken random stream per replication, exactly as in the
        # authors' notebook: the features, the labels and the mini-batch
        # index sequence are drawn from the same ``np.random`` stream.
        rng = np.random.RandomState(seed)
        X, y, theta0 = generate_logistic_data(N, d, seed, rng=rng)
        schedule = make_batch_schedule(N, batch_size, n, seed, replace=False,
                                       rng=rng)

        # ---- full-batch GD to locate the minimizer x^* ----
        theta = np.zeros((d, 1))
        for _ in range(10 * n):
            theta = theta - args.alpha * gradient(theta, X, y)
        x_star = theta

        A = hessian(x_star, X, y)
        ev = np.linalg.eigvalsh(A)
        mu = float(np.min(ev))
        gamma = ((1 - mu * args.alpha) / (1 + mu * args.alpha)) ** 2
        gamma_mean[seed] = gamma

        # Slot 4 of gamma_set is ``None``: fill it with this replication's
        # adaptive weight, so that column 4 is SGDM-adap throughout.
        gammas = list(gamma_set)
        gammas[4] = gamma
        initial = np.zeros((d, 1))  # paper starts from theta = 0

        for j, gamma_j in enumerate(gammas):
            x = initial.copy()
            m = np.zeros((d, 1))
            runs = {n0: np.zeros((d, 1)) for n0 in args.n0_list}
            for k in range(n):
                idx = schedule[k * batch_size:(k + 1) * batch_size]
                xi, yi = X[idx], y[idx]
                # Record the current iterate x_{k+1} *before* updating it, so
                # that entry 0 is the initial point x_1 (whose error is the
                # common starting value of every curve in Figure EC.1).
                err_last[seed, k, j] = np.linalg.norm(x - x_star)
                g = gradient(x, xi, yi)
                # The authors' recursion, verbatim:
                #     m_t = gamma * m_{t-1} + (1 - gamma) * grad(x_t)
                # with the *first* momentum set to the raw mini-batch gradient,
                # m_1 = grad(x_1), without the (1 - gamma) factor.  The
                # special case is not cosmetic: for gamma = 0.9 it makes the first step
                # ten times larger, which is what excites the underdamped
                # oscillation that dominates Figure EC.1(b) (the gamma = 0.9
                # curve dips to ~1e-2 near t = 15 and then rises to a ~8e-2
                # hump near t = 28).  Dropping it flattens that hump by ~5x.
                if k == 0:
                    m = g
                else:
                    m = gamma_j * m + (1 - gamma_j) * g
                x = x - args.alpha * m
                for n0 in args.n0_list:
                    # ``x`` has already been updated, so at iteration k it
                    # holds x_{k+1}: the first averaged iterate is x_{n0+1},
                    # i.e. the k = n0 case, as in xbar_{n0+t} = (1/t) sum
                    # x_{n0+1..n0+t}.
                    if k >= n0:
                        runs[n0] = runs[n0] + x
                        if k - n0 < ave_len - 1:
                            err_ave[n0][seed, k - n0, j] = \
                                np.linalg.norm(runs[n0] / (k - n0 + 1) - x_star)

    print(f"[logistic] mean adaptive gamma = {gamma_mean.mean():.4f}")

    labels = ["SGD", "SGDM-0.3", "SGDM-0.5", "SGDM-0.7", "SGDM-adap",
              "SGDM-0.8", "SGDM-0.9"]

    # (colour, marker, linestyle) per curve, copied from the authors' plotting
    # cells.  Their ``markers`` list is ["o", "v", "s", "p", "P", "*", "+", "x"]
    # and their ``linestyles`` list is ['-.', '--', ':', '-.', '-']; the curves
    # that are not given an explicit colour pick up the default property cycle
    # in plotting order, i.e. C0/C1/C2/C3 for the first four such curves.
    SMALL_STYLE = [("#9467bd", "o", ":"),    # SGD        colors[4], ':.'
                   ("#1f77b4", "s", "--"),   # SGDM-0.3   C0, markers[2], ls[1]
                   ("#ff7f0e", "p", ":"),    # SGDM-0.5   C1, markers[3], ls[2]
                   ("#2ca02c", "P", "-."),   # SGDM-0.7   C2, markers[4], ls[3]
                   ("#d62728", "*", "-")]    # SGDM-adap  colors[3], markers[5], ls[4]
    LARGE_STYLE = [("#d62728", "*", "-"),    # SGDM-adap  '#d62728', markers[5], '-'
                   ("#1f77b4", "p", ":"),    # SGDM-0.8   C0, markers[3], ls[2]
                   ("#ff7f0e", "P", "-.")]   # SGDM-0.9   C1, markers[4], ls[3]

    def plot_single(fname, curve, idxs, styles, ylab, ylim):
        """curve: (num_seed, T, n_gammas); idxs: column indices, in legend order."""
        fig = plt.figure(figsize=(4, 4), dpi=150)
        # The geometry of the authors' plotting cells.  It matters visually:
        # a float ``markevery`` places markers every ``markevery``-th fraction
        # of the *axes bounding-box diagonal* measured **along the curve**, so
        # the marker positions depend on the axes aspect and on the data, and
        # reproduce the published panels only with this exact geometry.
        ax = fig.add_subplot(gridspec.GridSpec(
            1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
        for j, (color, marker, ls) in zip(idxs, styles):
            ax.plot((curve[:, :, j] ** 2).mean(axis=0), label=labels[j],
                    marker=marker, markevery=0.3, linestyle=ls, color=color)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim)
        ax.legend()
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # labels indices: 0=SGD,1=0.3,2=0.5,3=0.7,4=adap,5=0.8,6=0.9
    # Panel (a) "small gamma": SGD, SGDM-0.3/0.5/0.7, SGDM-adap.
    # Panel (b) "large gamma": SGDM-adap, SGDM-0.8, SGDM-0.9 -- the published
    # panels (EC.1(b), EC.2(b), EC.3(b)) contain no SGD reference curve.
    idx_small = [0, 1, 2, 3, 4]
    idx_large = [4, 5, 6]

    # All six published panels span t = 0, ..., 98 (99 points); the axis limits
    # below were read off the published panels' tick marks.
    T = ave_len - 1
    YLIM_EC1 = (3.3e-4, 1.42)      # both EC.1 panels
    YLIM_EC2 = (1e-4, 2e-1)        # both EC.2 panels (authors' notebook)
    YLIM_EC3 = (8e-5, 4e-2)        # both EC.3 panels (authors' notebook)

    # ---- Figure EC.1: last iterate, small/large gamma ----
    err_last = err_last[:, :T, :]
    plot_single("fig_ec1_small.png", err_last, idx_small, SMALL_STYLE,
                r"$||x_t-x^*||^2$", YLIM_EC1)
    plot_single("fig_ec1_large.png", err_last, idx_large, LARGE_STYLE,
                r"$||x_t-x^*||^2$", YLIM_EC1)

    # ---- Figures EC.2-EC.3: averaged, per n0, small/large gamma ----
    n0_first, n0_second = args.n0_list[0], args.n0_list[-1]
    plot_single("fig_ec2_small.png", err_ave[n0_first], idx_small, SMALL_STYLE,
                r"$||\bar{x}_{n_0+t}-x^*||^2$", YLIM_EC2)
    plot_single("fig_ec2_large.png", err_ave[n0_first], idx_large, LARGE_STYLE,
                r"$||\bar{x}_{n_0+t}-x^*||^2$", YLIM_EC2)
    plot_single("fig_ec3_small.png", err_ave[n0_second], idx_small,
                SMALL_STYLE, r"$||\bar{x}_{n_0+t}-x^*||^2$", YLIM_EC3)
    plot_single("fig_ec3_large.png", err_ave[n0_second], idx_large,
                LARGE_STYLE, r"$||\bar{x}_{n_0+t}-x^*||^2$", YLIM_EC3)

    np.savez_compressed(os.path.join(args.out, "logistic_raw.npz"),
                        err_last=err_last, gamma_mean=gamma_mean,
                        **{f"err_ave_n0_{n0}": err_ave[n0]
                           for n0 in args.n0_list})
    print(f"[logistic] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
