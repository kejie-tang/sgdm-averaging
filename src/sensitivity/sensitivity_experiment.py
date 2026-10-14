"""
Figures 9-11: sensitivity of SGD and SGDM to the learning rate.

Setting
-------
N = 20,000, d = 10, A_i = (V_i^T V_i + 1.03 I)/1.04 (so the eigenvalues lie in
[0, 1]), batch size B = 0.2 N, b_i ~ U[0, 1]^d.  Learning rates alpha = 2^e.

Figure 9 (last iterate, ||x_t - x^*||^2):
    left panel  -- SGD with alpha in {2^-9, 2^-7, 2^-5};
    right panel -- SGDM with gamma = 0.8 and alpha in {2^-9, ..., 2^-1}.
    SGD fails to converge for alpha >= 2^-3; SGDM-0.8 still converges at 2^-1.
Figure 10 (final error ||x_T - x^*||^2 at T = 500 vs. alpha):
    SGD, SGDM-0.8, SGDM-0.9 over alpha = 2^-11, ..., 2^2.  The largest
    learning rates that still converge are 2^-4 (SGD), 2^-1 (SGDM-0.8) and
    2^0 (SGDM-0.9), matching alpha L < 2 for SGD and alpha L < 2(1+gamma)/(1-gamma)
    for SGDM.
Figure 11 (averaged iterate, ||xbar_{n0+t} - x^*||^2 with n0 = 500):
    same two panels as Figure 9 but for the Polyak-averaged iterates.

Because every (gamma, alpha) pair shares the same pre-sampled mini-batches, the
per-iteration batch gradients are computed once per replication and reused, so
the full grid costs about the same as a single run.

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
from sgdm_core import generate_quadratic_data, make_batch_schedule  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Panel composition of Figures 9 and 11 (paper).
SGD_CURVE_EXPS = (-9, -7, -5)
SGDM_CURVE_EXPS = (-9, -7, -5, -3, -1)
FINAL_EXPS = tuple(range(-11, 3))          # Figure 10 x-axis: 2^-11 ... 2^2

# Learning rates above the stability threshold make the iterates overflow; the
# paper's Figure 10 shows them as a plateau at 10^0, so inf/nan errors are
# replaced by 1.
DIVERGED_CAP = 1.0


def _cap(v):
    """Map a diverged (inf/nan or > 1) error onto the 10^0 plateau."""
    return v if np.isfinite(v) and v <= DIVERGED_CAP else DIVERGED_CAP


def _run_grid(A_seq, b_seq, x_star, initial, alphas, gammas, K, T, n0):
    """Run every (gamma, alpha) pair on a shared batch-gradient sequence.

    Returns ``last``, ``ave``, ``final`` dictionaries keyed by ``(gamma, alpha)``
    holding, respectively, the squared last-iterate error for t = 0..T, the
    squared averaged error for t = 1..K-n0, and the squared error at index T.
    """
    p = initial.shape[0]
    last, ave, final = {}, {}, {}
    for gamma in gammas:
        for alpha in alphas:
            x = initial.copy()
            m = np.zeros((p, 1))
            err_t = np.empty(T + 1)
            run = np.zeros((p, 1))
            cnt = 0
            ave_curve = []
            # Large learning rates make the iterates overflow before the
            # horizon; the resulting inf/nan are handled (and capped) below.
            for k in range(K):
                if k <= T:
                    d = x - x_star
                    err_t[k] = float(np.sum(d * d))
                if n0 <= k:                     # x_{k+1} is averaged from here
                    run = run + x
                    cnt += 1
                    d = run / cnt - x_star
                    ave_curve.append(float(np.sum(d * d)))
                g = A_seq[k] @ x + b_seq[k]
                if gamma == 0.0:
                    x = x - alpha * g
                else:
                    m = gamma * m + (1.0 - gamma) * g
                    x = x - alpha * m
            key = (gamma, alpha)
            last[key] = err_t
            ave[key] = np.asarray(ave_curve)
            final[key] = err_t[T]
    return last, ave, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=1000,
                    help="number of iterations per run (>= n0 + 500 for "
                         "Figure 11 and >= T + 1 for Figures 9-10)")
    ap.add_argument("--T", type=int, default=500,
                    help="reporting horizon for the last iterate (Figures 9-10)")
    ap.add_argument("--n0", type=int, default=500,
                    help="averaging burn-in for Figure 11")
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "sensitivity"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K, T, n0 = args.n, args.p, args.K, args.T, args.n0
    batch_size = int(round(args.batch_frac * n))
    final_alphas = [2.0 ** e for e in FINAL_EXPS]
    n_a = len(final_alphas)
    n_b = len(args.betas)
    if args.n0 + T > K:
        raise ValueError(
            f"--K={K} is too small: the averaged curve needs n0 + T = "
            f"{args.n0 + T} iterations.  Increase --K.")
    if args.T + 1 > K:
        raise ValueError(f"--T={T} must be smaller than --K={K}.")

    gamma_all = [0.0] + list(args.betas)
    betas0 = args.betas[0]

    # squared error at index T, per replication
    err_sgd = np.zeros((args.num_seed, n_a))
    err_sgdm = np.zeros((args.num_seed, n_b, n_a))
    # last-iterate and averaged curves, only for the panel exponents
    last_sgd = np.zeros((args.num_seed, T + 1, len(SGD_CURVE_EXPS)))
    last_sgdm = np.zeros((args.num_seed, T + 1, len(SGDM_CURVE_EXPS)))
    ave_sgd = np.zeros((args.num_seed, K - n0, len(SGD_CURVE_EXPS)))
    ave_sgdm = np.zeros((args.num_seed, K - n0, len(SGDM_CURVE_EXPS)))

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[sensitivity] seed {seed}/{args.num_seed}", flush=True)
        data = generate_quadratic_data(n, p, seed, rho=1.0, shift=1.03,
                                       norm_const=1.04)
        Ai, bi = data["Ai"], data["bi"]
        x_star, initial = data["x_star"], data["initial"]
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)

        # Pre-compute the per-iteration batch gradients once (shared by all
        # (gamma, alpha) pairs, since the mini-batches do not depend on them).
        A_seq = np.empty((K, p, p))
        b_seq = np.empty((K, p, 1))
        for k in range(K):
            idx = schedule[k * batch_size:(k + 1) * batch_size]
            A_seq[k] = Ai[idx].mean(axis=0)
            b_seq[k] = bi[idx].mean(axis=0)

        with np.errstate(over="ignore", invalid="ignore"):
            last, ave, final = _run_grid(A_seq, b_seq, x_star, initial,
                                         final_alphas, gamma_all, K, T, n0)

        for ai_, alpha in enumerate(final_alphas):
            err_sgd[seed, ai_] = _cap(final[(0.0, alpha)])
            for bi_, beta in enumerate(args.betas):
                err_sgdm[seed, bi_, ai_] = _cap(final[(beta, alpha)])

        for ai_, e in enumerate(SGD_CURVE_EXPS):
            alpha = 2.0 ** e
            last_sgd[seed, :, ai_] = last[(0.0, alpha)]
            ave_sgd[seed, :, ai_] = ave[(0.0, alpha)]
        for ai_, e in enumerate(SGDM_CURVE_EXPS):
            alpha = 2.0 ** e
            last_sgdm[seed, :, ai_] = last[(betas0, alpha)]
            ave_sgdm[seed, :, ai_] = ave[(betas0, alpha)]

    # ---- Figure 9: last iterate under different learning rates ----
    fig = plt.figure(figsize=(9, 4), dpi=150)
    spec = gridspec.GridSpec(1, 2, left=0.075, right=0.98, top=0.93,
                             bottom=0.15, wspace=0.25, figure=fig)

    ax = fig.add_subplot(spec[0])
    for j, e in enumerate(SGD_CURVE_EXPS):
        ax.plot(np.arange(T + 1), last_sgd[:, :, j].mean(axis=0),
                label=rf"SGD-$2^{{{e}}}$", marker=MARKERS[j],
                markevery=max(1, (T + 1) // 5),
                linestyle=LINESTYLES[j % 3], color=COLORS[j])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|x_t-x^*\|^2$")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax = fig.add_subplot(spec[1])
    for j, e in enumerate(SGDM_CURVE_EXPS):
        ax.plot(np.arange(T + 1), last_sgdm[:, :, j].mean(axis=0),
                label=rf"SGDM-$2^{{{e}}}$", marker=MARKERS[j],
                markevery=max(1, (T + 1) // 5),
                linestyle=LINESTYLES[j % 3], color=COLORS[j])
    ax.set_xlabel(r"$t$")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_dif_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 10: final error at T vs alpha ----
    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.15, right=0.97, top=0.95, bottom=0.15, figure=fig)[0])
    ax.plot(FINAL_EXPS, err_sgd.mean(axis=0), label="SGD",
            marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0])
    for bi_, beta in enumerate(args.betas):
        ax.plot(FINAL_EXPS, err_sgdm[:, bi_, :].mean(axis=0),
                label=rf"SGDM-{beta}", marker=MARKERS[bi_ + 1],
                linestyle=LINESTYLES[bi_ + 1], color=COLORS[bi_ + 1])
    ax.set_xlabel(r"$\alpha=2^{x}$")
    ax.set_ylabel(r"$\|x_T-x^*\|^2$")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(os.path.join(args.out, "fig_sen_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 11: averaged iterate under different learning rates ----
    t_ave = np.arange(1, K - n0 + 1)
    fig = plt.figure(figsize=(9, 4), dpi=150)
    spec = gridspec.GridSpec(1, 2, left=0.075, right=0.98, top=0.93,
                             bottom=0.15, wspace=0.25, figure=fig)
    ax = fig.add_subplot(spec[0])
    for j, e in enumerate(SGD_CURVE_EXPS):
        ax.plot(t_ave, ave_sgd[:, :, j].mean(axis=0),
                label=rf"SGD-$2^{{{e}}}$", marker=MARKERS[j],
                markevery=max(1, len(t_ave) // 4),
                linestyle=LINESTYLES[j % 3], color=COLORS[j])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|\bar{x}_{n_0+t}-x^*\|^2$")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax = fig.add_subplot(spec[1])
    for j, e in enumerate(SGDM_CURVE_EXPS):
        ax.plot(t_ave, ave_sgdm[:, :, j].mean(axis=0),
                label=rf"SGDM-$2^{{{e}}}$", marker=MARKERS[j],
                markevery=max(1, len(t_ave) // 4),
                linestyle=LINESTYLES[j % 3], color=COLORS[j])
    ax.set_xlabel(r"$t$")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(args.out, "fig_ave_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "sensitivity_raw.npz"),
                        err_sgd=err_sgd, err_sgdm=err_sgdm,
                        last_sgd=last_sgd, last_sgdm=last_sgdm,
                        ave_sgd=ave_sgd, ave_sgdm=ave_sgdm,
                        final_exps=np.array(FINAL_EXPS),
                        sgd_curve_exps=np.array(SGD_CURVE_EXPS),
                        sgdm_curve_exps=np.array(SGDM_CURVE_EXPS),
                        betas=np.array(args.betas), T=T, n0=n0, K=K)
    print(f"[sensitivity] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
