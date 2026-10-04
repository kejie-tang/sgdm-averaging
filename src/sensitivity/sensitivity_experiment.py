"""
Figures 6-8: sensitivity of SGD and SGDM to the learning rate.

Setting
-------
N = 20,000, d = 10, A_i = (V_i^T V_i + 1.03 I)/1.04 so that the average
condition number is L/mu = 26/1.  Batch size B = 0.2 N.
Learning rates alpha in {2^1, 2^0, 2^-1, ..., 2^-9}.

Figure 6:
    final ||x_T - x^*||^2 at T = 500 for SGD and SGDM (gamma = 0.8, 0.9).
    SGD fails for alpha >= 2^-3; SGDM converges up to 2^-1 (gamma=0.8) / 2^0
    (gamma=0.9).
Figure 7:
    finite-sample error at T = 500 vs. the learning rate.
Figure 8:
    averaged SGD / SGDM with n0 = 500.

Outputs (results/sensitivity/):
    fig_dif_alpha.png, fig_sen_alpha.png, fig_ave_alpha.png, sensitivity_raw.npz

Run:  python src/sensitivity/sensitivity_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    generate_quadratic_data, make_batch_schedule, run_sgd, run_sgdm,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--n0", type=int, default=500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--exps", type=int, nargs="+",
                    default=[-1, -3, -5, -7, -9])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "sensitivity"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K = args.n, args.p, args.K
    batch_size = int(round(args.batch_frac * n))
    alphas = [2.0 ** e for e in args.exps]
    n_a = len(alphas)
    n_b = len(args.betas)
    if args.n0 >= K:
        raise ValueError(
            f"--n0={args.n0} must be smaller than --K={K} (the averaged curve "
            f"starts at n0).  Reduce --n0 or increase --K.")

    # final error at T=K, per replication
    err_sgd = np.zeros((args.num_seed, n_a))
    err_sgdm = np.zeros((args.num_seed, n_b, n_a))
    # averaged error curve (n0=500, up to K-n0 points)
    ave_len = K - args.n0
    ave_sgd = np.zeros((args.num_seed, ave_len, n_a))
    ave_sgdm = np.zeros((args.num_seed, ave_len, n_b, n_a))
    diverge = np.zeros((args.num_seed, n_a), dtype=bool)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[sensitivity] seed {seed}/{args.num_seed}", flush=True)
        data = generate_quadratic_data(n, p, seed, rho=1.0, shift=1.03,
                                       normalize=True)
        Ai, bi = data["Ai"], data["bi"]
        x_star, initial = data["x_star"], data["initial"]
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)

        for ai_, alpha in enumerate(alphas):
            traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                           alpha, K)
            err_sgd[seed, ai_] = np.linalg.norm(traj[-1] - x_star) ** 2
            diverge[seed, ai_] = (not np.isfinite(err_sgd[seed, ai_])) or \
                err_sgd[seed, ai_] > 1.0
            run = np.cumsum(traj[args.n0:], axis=0) / \
                np.arange(1, ave_len + 1)[:, None, None]
            sq = ((run - x_star) ** 2).sum(axis=(1, 2))
            ave_sgd[seed, :, ai_] = np.where(sq > 1e6, np.nan, sq)

            for bi_, beta in enumerate(args.betas):
                traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size,
                                  schedule, alpha, beta, K)
                err_sgdm[seed, bi_, ai_] = \
                    np.linalg.norm(traj_m[-1] - x_star) ** 2
                run = np.cumsum(traj_m[args.n0:], axis=0) / \
                    np.arange(1, ave_len + 1)[:, None, None]
                sq = ((run - x_star) ** 2).sum(axis=(1, 2))
                ave_sgdm[seed, :, bi_, ai_] = np.where(sq > 1e6, np.nan, sq)

    # ---- Figure 6: dif-alpha ----
    fig = plt.figure(figsize=(8, 4), dpi=150)
    spec = gridspec.GridSpec(1, 2, left=0.08, right=0.98, top=0.92,
                             bottom=0.15, wspace=0.3, figure=fig)
    for bi_, beta in enumerate(args.betas):
        ax = fig.add_subplot(spec[bi_])
        for ai_, alpha in enumerate(alphas):
            ax.plot(ave_sgdm[:, :, bi_, ai_].mean(axis=0),
                    label=r"SGDM($\gamma$={}),$\alpha=2^{{{}}}$".format(
                        beta, args.exps[ai_]),
                    marker=MARKERS[ai_], markevery=max(1, ave_len // 4),
                    linestyle=LINESTYLES[ai_ % 3])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||\bar{x}_{n_0+t}-x^*||^2$")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
    fig.savefig(os.path.join(args.out, "fig_dif_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 7: sen-alpha -- final error vs alpha ----
    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.15, right=0.97, top=0.95, bottom=0.15, figure=fig)[0])
    ax.plot(args.exps, err_sgd.mean(axis=0), label="SGD",
            marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0])
    for bi_, beta in enumerate(args.betas):
        ax.plot(args.exps, err_sgdm[:, bi_, :].mean(axis=0),
                label=rf"SGDM($\gamma$={beta})", marker=MARKERS[bi_ + 1],
                linestyle=LINESTYLES[bi_ + 1], color=COLORS[bi_ + 1])
    ax.set_xlabel(r"$\alpha=2^{x}$")
    ax.set_ylabel(r"$||x_T-x^*||^2$")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(os.path.join(args.out, "fig_sen_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 8: ave-alpha (gamma=0.8) ----
    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.13, right=0.97, top=0.95, bottom=0.15, figure=fig)[0])
    for ai_, alpha in enumerate(alphas):
        ax.plot(ave_sgd[:, :, ai_].mean(axis=0),
                label=r"SGD,$\alpha=2^{{{}}}$".format(args.exps[ai_]),
                marker=MARKERS[ai_], markevery=max(1, ave_len // 4),
                linestyle=LINESTYLES[0])
        ax.plot(ave_sgdm[:, :, 0, ai_].mean(axis=0),
                label=r"SGDM($\gamma$=0.8),$\alpha=2^{{{}}}$".format(
                    args.exps[ai_]),
                marker=MARKERS[ai_], markevery=max(1, ave_len // 4),
                linestyle=LINESTYLES[1])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$||\bar{x}_{n_0+t}-x^*||^2$")
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(os.path.join(args.out, "fig_ave_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "sensitivity_raw.npz"),
                        err_sgd=err_sgd, err_sgdm=err_sgdm,
                        ave_sgd=ave_sgd, ave_sgdm=ave_sgdm,
                        diverge=diverge, exps=np.array(args.exps),
                        betas=np.array(args.betas))
    print(f"[sensitivity] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
