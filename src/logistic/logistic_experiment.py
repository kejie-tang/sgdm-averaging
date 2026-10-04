"""
Figures 10-12: SGD / SGDM / averaged SGD / averaged SGDM on the logistic loss.

Setting
-------
N = 20,000, d = 10, a_i ~ N(0, I_d), b_i ~ Bernoulli(sigmoid(theta0^T a_i)) with
theta0 = (1,...,1)/sqrt(d).  Batch size B = 0.2 N, learning rate alpha = 0.5,
gamma in {0, 0.3, 0.5, 0.7, 0.8, 0.9} plus the adaptive weight
gamma = ((1-mu*alpha)/(1+mu*alpha))^2 computed at the population Hessian
evaluated at the full-batch minimizer x^* (average adaptive weight ~ 0.75).

Panel (a) "small gamma": SGD, SGDM-0.3/0.5/0.7, SGDM-adap.
Panel (b) "large gamma": SGD, SGDM-0.8/0.9, SGDM-adap.

The full-batch gradient descent used to locate x^* is the same procedure as in
the paper (beta = 0, 10*K iterations, alpha = 0.5).

Outputs (results/logistic/), one file per paper panel:
    fig_logistic_small.png / fig_logistic_large.png   -> Figure 10(a),(b)
    fig_logistic_ave_small.png / _large.png           -> Figure 11(a),(b),  n0=10
    fig_logistic_ave2_small.png / _large.png          -> Figure 12(a),(b),  n0=40
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
    # the paper uses n0 = 10 for Figure 11 and n0 = 40 for Figure 12
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

    # last-iterate error (Figure 7) and averaged error per n0 (Figures 8-9)
    err_last = np.zeros((args.num_seed, K - 1, n_beta))
    err_ave = {n0: np.zeros((args.num_seed, ave_len - 1, n_beta))
               for n0 in args.n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[logistic] seed {seed}/{args.num_seed}", flush=True)
        X, y, theta0 = generate_logistic_data(n, p, seed)
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)

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
                g = gradient(x, xi, yi)
                m = beta * m + (1 - beta) * g if k > 0 else g
                x = x - args.alpha * m
                if k > 0:
                    err_last[seed, k - 1, j] = np.linalg.norm(x - x_star)
                for n0 in args.n0_list:
                    if k > n0:
                        runs[n0] = runs[n0] + x
                        if k - n0 - 1 < ave_len - 1:
                            err_ave[n0][seed, k - n0 - 1, j] = \
                                np.linalg.norm(runs[n0] / (k - n0) - x_star)

    print(f"[logistic] mean adaptive gamma = {gamma_mean.mean():.4f}")

    labels = ["SGD", "SGDM-0.3", "SGDM-0.5", "SGDM-0.7", "SGDM-adap",
              "SGDM-0.8", "SGDM-0.9"]

    def plot_single(fname, curve, idxs, ylab, ylim=None):
        """curve: (num_seed, T, n_beta); idxs: keyed by label-order."""
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(gridspec.GridSpec(
            1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15,
            figure=fig)[0])
        for mk, j in enumerate(idxs):
            ax.plot(curve[:, :, j].mean(axis=0) ** 2, label=labels[j],
                    marker=MARKERS[mk], markevery=8,
                    linestyle=LINESTYLES[mk % 4])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # labels indices: 0=SGD,1=0.3,2=0.5,3=0.7,4=adap,5=0.8,6=0.9
    # Panel (a) "small gamma": SGD, SGDM-0.3/0.5/0.7, SGDM-adap.
    # Panel (b) "large gamma": SGD, SGDM-0.8/0.9, SGDM-adap  -- the paper keeps
    # SGD in this panel as the reference curve (see Figure 10(b),
    # Figure 11(b) and Figure 12(b)).
    idx_small = [0, 1, 2, 3, 4]
    idx_large = [0, 4, 5, 6]

    # ---- Figure 10: last iterate, small/large gamma ----
    plot_single("fig_logistic_small.png", err_last, idx_small,
                r"$||x_t-x^*||^2$")
    plot_single("fig_logistic_large.png", err_last, idx_large,
                r"$||x_t-x^*||^2$")

    # ---- Figures 8-9: averaged, per n0, small/large gamma ----
    n0_first, n0_second = args.n0_list[0], args.n0_list[-1]
    plot_single("fig_logistic_ave_small.png", err_ave[n0_first], idx_small,
                r"$||\bar{x}_{n_0+t}-x^*||^2$")
    plot_single("fig_logistic_ave_large.png", err_ave[n0_first], idx_large,
                r"$||\bar{x}_{n_0+t}-x^*||^2$")
    plot_single("fig_logistic_ave2_small.png", err_ave[n0_second], idx_small,
                r"$||\bar{x}_{n_0+t}-x^*||^2$")
    plot_single("fig_logistic_ave2_large.png", err_ave[n0_second], idx_large,
                r"$||\bar{x}_{n_0+t}-x^*||^2$")

    np.savez_compressed(os.path.join(args.out, "logistic_raw.npz"),
                        err_last=err_last, gamma_mean=gamma_mean,
                        **{f"err_ave_n0_{n0}": err_ave[n0]
                           for n0 in args.n0_list})
    print(f"[logistic] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
