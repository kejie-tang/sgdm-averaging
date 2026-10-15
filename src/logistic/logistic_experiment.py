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

Panel (a) "small gamma": SGD, SGDM-0.3/0.5/0.7, SGDM-adap.
Panel (b) "large gamma": SGDM-adap, SGDM-0.8, SGDM-0.9 (no SGD curve, as in
the published panels).

The full-batch gradient descent used to locate x^* is the same procedure as in
the paper (beta = 0, 10*K iterations, alpha = 0.5).

Update rule
-----------
The authors' recursion is

    g_t = beta g_{t-1} + (1 - beta) grad(x_t),   x_{t+1} = x_t - alpha g_t,

with the **first** momentum set to the raw mini-batch gradient, g_1 = grad(x_1),
i.e. *without* the (1 - beta) factor.  Their notebook applies that special case
to every beta (``if k == 0: gt = gradient(...)``).  It is not cosmetic.  The
recursion is an AR(2) with roots of modulus sqrt(beta) once
alpha(1 - beta) L is large, so for beta = 0.9 the iterates are only lightly
damped; a ten-times-larger first step is what excites the oscillation that
dominates the published Figure EC.1(b).  With the (1 - beta) factor restored,
the gamma = 0.9 hump shrinks from ~8e-2 to ~1.8e-2 and moves from t = 28 to
t = 40, and the EC.2(b) panel comes out qualitatively wrong (gamma = 0.9 ends up
converging fastest instead of slowest).  This file therefore reproduces the
authors' first step exactly.

Both panels start at t = 0 with the initial point x_0 = 0 (every curve of
Figure EC.1 begins from the same value, ||x_0 - x^*||^2 = 1.0: the quadratic
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
    """Mean gradient of the negative log-likelihood; returns (p, 1)."""
    theta = theta.reshape(-1)
    m = y.size
    h = sigmoid(X @ theta).reshape(-1)
    y = y.reshape(-1)
    return ((X.T @ (h - y)) / m).reshape(-1, 1)


def hessian(theta, X, y):
    m, p = X.shape
    theta = theta.reshape(-1)
    h = sigmoid(X @ theta).reshape(-1)
    H = np.zeros((p, p))
    for i in range(m):
        ai = X[i].reshape(p, 1)
        H += h[i] * (1 - h[i]) * (ai @ ai.T)
    return H / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    # the paper uses n0 = 10 for Figure EC.2 and n0 = 40 for Figure EC.3
    ap.add_argument("--n0_list", type=int, nargs="+", default=[10, 40])
    ap.add_argument("--ave_len", type=int, default=100)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "logistic"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p = args.n, args.p
    batch_size = int(round(args.batch_frac * n))
    ave_len = args.ave_len
    K = max(args.n0_list) + ave_len + 1

    # beta_set: SGD(0), fixed weights, adaptive (index 4), 0.8, 0.9
    beta_set = [0.0, 0.3, 0.5, 0.7, None, 0.8, 0.9]
    n_beta = len(beta_set)
    beta_fixed_small = [0.3, 0.5, 0.7]
    beta_fixed_large = [0.8, 0.9]

    # last-iterate error (Figure EC.1) and averaged error per n0 (Figures EC.2-EC.3)
    # err_last[k] is ||x_k - x^*||, so entry 0 is the initial point x_0.
    err_last = np.zeros((args.num_seed, K, n_beta))
    err_ave = {n0: np.zeros((args.num_seed, ave_len - 1, n_beta))
               for n0 in args.n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[logistic] seed {seed}/{args.num_seed}", flush=True)
        # One unbroken random stream per replication, exactly as in the
        # authors' notebook: the features, the labels and the mini-batch
        # index sequence are drawn from the same ``np.random`` stream.
        rng = np.random.RandomState(seed)
        X, y, theta0 = generate_logistic_data(n, p, seed, rng=rng)
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False,
                                       rng=rng)

        # ---- full-batch GD to locate the minimizer x^* ----
        theta = np.zeros((p, 1))
        for _ in range(10 * K):
            theta = theta - args.alpha * gradient(theta, X, y)
        x_star = theta

        A = hessian(x_star, X, y)
        ev = np.linalg.eigvalsh(A)
        mu = float(np.min(ev))
        gamma = ((1 - mu * args.alpha) / (1 + mu * args.alpha)) ** 2
        gamma_mean[seed] = gamma

        betas = [gamma if b is None else b for b in beta_set]
        initial = np.zeros((p, 1))  # paper starts from theta = 0

        for j, beta in enumerate(betas):
            x = initial.copy()
            m = np.zeros((p, 1))
            runs = {n0: np.zeros((p, 1)) for n0 in args.n0_list}
            for k in range(K):
                idx = schedule[k * batch_size:(k + 1) * batch_size]
                xi, yi = X[idx], y[idx]
                # Record the current iterate x_k *before* updating it, so that
                # entry 0 is the initial point x_0 (whose error is the common
                # starting value of every curve in Figure EC.1).
                err_last[seed, k, j] = np.linalg.norm(x - x_star)
                g = gradient(x, xi, yi)
                # The authors' recursion, verbatim:
                #     g_t = beta * g_{t-1} + (1 - beta) * grad(x_t)
                # with the *first* momentum set to the raw mini-batch gradient,
                # g_1 = grad(x_1), without the (1 - beta) factor.  The special
                # case is not cosmetic: for beta = 0.9 it makes the first step
                # ten times larger, which is what excites the underdamped
                # oscillation that dominates Figure EC.1(b) (the gamma = 0.9
                # curve dips to ~1e-2 near t = 15 and then rises to a ~8e-2
                # hump near t = 28).  Dropping it flattens that hump by ~5x.
                if k == 0:
                    m = g
                else:
                    m = beta * m + (1 - beta) * g
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
        """curve: (num_seed, T, n_beta); idxs: column indices, in legend order."""
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
