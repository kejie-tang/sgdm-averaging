"""
Figures 8 and 12: asymptotic normality of the averaged SGD / SGDM estimator.

Setting
-------
Quadratic loss, N = 20,000, d = 10, B = 0.2 N.
The statistic (Eq. (def:z))
    Z = sqrt(B) / (sigma * sqrt(omega^T Sigma^{-1} Omega Sigma^{-1} omega))
        * sum_{t=n0+1}^{n} omega^T (x_t - x^*) / sqrt(n - n0)
is asymptotically N(0, 1) by Corollary 3.

Figure 8 : frequency of Z for averaged SGD and averaged SGDM, gamma = 0.9,
           n0 = 1000, n = 2000, over 1000 trials.
Figure 12: empirical P(|Z| < 1.96) as a function of the learning rate for SGD
           and SGDM (gamma = 0.8, 0.9), over the grid alpha = 2^-11 ... 2^2,
           with a trajectory of n = 500 iterations, a burn-in of n0 = 200 and
           1000 trials per learning rate.

           ``--n_cov 500 --n0_cov 200`` are the settings for this panel: the
           transition of SGDM-0.9 falls between alpha = 2^-8 (~0.1) and
           alpha = 2^-7 (~0.9), and the plateau sits at ~0.98.  Note where the
           three curves fall back to zero: SGD between alpha = 2^-4 and 2^-3,
           SGDM-0.8 between 2^-1 and 2^0, SGDM-0.9 between 2^0 and 2^1.  Those
           are the linear stability limits of the recursions, alpha * L < 2 for
           SGD and alpha * L < 2 (1 + gamma) / (1 - gamma) for SGDM (the
           momentum term is damped by 1 - gamma), with L = 25.8 for this A --
           not a failure of the normal approximation.

Outputs (results/clt/):
    fig_clt_sgd.png, fig_clt_sgdm.png, fig_clt_alpha.png, clt_raw.npz

Run:  python src/clt/clt_experiment.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    asymptotic_covariance, batch_gradient_sequence, clt_projection,
    make_batch_schedule,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
# The published Figure 12 draws SGD dash-dot, SGDM-0.8 solid and SGDM-0.9
# dashed (the authors' plotting cell: ``linestyles = ['-.', '-', '--']``).
LINESTYLES = ["-.", "-", "--", ":"]


def hist_figure(z, gamma_label, fname, out_dir, color="#1f77b4"):
    # The authors' canvas: 4 x 4 in with the axes box at 0.1/0.9 by 0.2/0.9.
    # The published panels carry no axis labels, only the tick values.
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
    # The published panels draw the N(0,1) reference first, so that it heads
    # the legend.
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=2,
            label="N(0,1)")
    ax.hist(z, range=(-5, 5), bins=20, density=True, rwidth=0.8,
            color=color, label=gamma_label)
    ax.set_ylim([0, 0.45])
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--num_seed", type=int, default=1000)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    # histogram experiment (Figure 8)
    ap.add_argument("--n_hist", type=int, default=2000)
    ap.add_argument("--n0_hist", type=int, default=1000)
    ap.add_argument("--gamma_hist", type=float, default=0.9)
    # coverage experiment (Figure 12): an n = 500-iteration trajectory with a
    # burn-in of n0 = 200, averaged over 1000 trials per learning rate.
    ap.add_argument("--n_cov", type=int, default=500)
    ap.add_argument("--n0_cov", type=int, default=200)
    ap.add_argument("--num_seed_cov", type=int, default=1000)
    # learning rates of Figure 12: alpha = 2^2 down to 2^-11, as in the paper
    ap.add_argument("--exps", type=int, nargs="+",
                    default=list(range(2, -12, -1)))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "clt"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d = args.N, args.d
    batch_size = int(round(args.batch_frac * N))

    # --- shared problem instance for the histogram experiment ---
    # One seed-1 stream supplies the data, the starting point *and* the
    # projection direction, in the authors' order; the per-trial mini-batch
    # schedules are then drawn from their own per-trial streams below.
    rng = np.random.RandomState(1)
    V = rng.rand(N, d, d)
    Ai = np.empty((N, d, d))
    for i in range(N):
        Vi = V[i]
        Ai[i] = Vi @ Vi.T + 10.0 * np.eye(d)
    bi = rng.rand(N, d, 1)
    A = Ai.mean(axis=0)
    b = bi.mean(axis=0)
    x_star = np.linalg.solve(A, -b)
    noise = rng.rand(d, 1)
    noise /= np.linalg.norm(noise)
    initial = x_star + noise * 0.1
    phi = rng.rand(d, 1)
    phi /= np.linalg.norm(phi)
    ev = np.linalg.eigvalsh(A)
    mu = float(ev.min())
    alpha_hist = 0.001

    cov = asymptotic_covariance(Ai, bi, x_star, A, batch_size)

    z_sgd = np.zeros(args.num_seed)
    z_sgdm = np.zeros(args.num_seed)
    for seed in range(args.num_seed):
        if seed % 100 == 0:
            print(f"[clt-hist] seed {seed}/{args.num_seed}", flush=True)
        schedule = make_batch_schedule(N, batch_size, args.n_hist, seed,
                                       replace=True)
        # averaged SGD and averaged SGDM replay the same batches
        A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule,
                                               args.n_hist)
        z_sgd[seed], _ = clt_projection(
            Ai, bi, x_star, initial, batch_size, schedule, alpha_hist,
            args.n_hist, args.n0_hist, cov, gamma=0.0, phi=phi,
            A_seq=A_seq, b_seq=b_seq)
        z_sgdm[seed], _ = clt_projection(
            Ai, bi, x_star, initial, batch_size, schedule, alpha_hist,
            args.n_hist, args.n0_hist, cov, gamma=args.gamma_hist, phi=phi,
            A_seq=A_seq, b_seq=b_seq)

    print(f"[clt-hist] mean/std SGD  = {z_sgd.mean():.3f} / {z_sgd.std():.3f}")
    print(f"[clt-hist] mean/std SGDM = {z_sgdm.mean():.3f} / {z_sgdm.std():.3f}")

    hist_figure(z_sgd, "SGD", "fig_clt_sgd.png", args.out)
    hist_figure(z_sgdm, f"SGDM-{args.gamma_hist}",
                "fig_clt_sgdm.png", args.out, color="#ff7f0e")

    # --- coverage vs. learning rate (Figure 12) ---
    rng2 = np.random.RandomState(1)
    V2 = rng2.rand(N, d, d)
    Ai2 = np.empty((N, d, d))
    for i in range(N):
        Vi = V2[i]
        Ai2[i] = (Vi @ Vi.T + 1.03 * np.eye(d)) / 1.04
    bi2 = rng2.rand(N, d, 1)
    A2 = Ai2.mean(axis=0)
    b2 = bi2.mean(axis=0)
    x_star2 = np.linalg.solve(A2, -b2)
    noise2 = rng2.rand(d, 1)
    noise2 /= np.linalg.norm(noise2)
    initial2 = x_star2 + noise2 * 0.1
    cov2 = asymptotic_covariance(Ai2, bi2, x_star2, A2, batch_size)

    alphas = [2.0 ** e for e in args.exps]
    n_alphas, n_gammas = len(alphas), len(args.gammas)
    cover_sgd = np.zeros((args.num_seed_cov, n_alphas))
    cover_sgdm = np.zeros((args.num_seed_cov, n_gammas, n_alphas))
    z1_96 = 1.96

    for seed in range(args.num_seed_cov):
        if seed % 200 == 0:
            print(f"[clt-cov] seed {seed}/{args.num_seed_cov}", flush=True)
        schedule = make_batch_schedule(N, batch_size, args.n_cov, seed,
                                       replace=False)
        # Every (gamma, alpha) pair in the sweep replays the same batches, so
        # the batch gradients are formed once per trial.
        A_seq, b_seq = batch_gradient_sequence(Ai2, bi2, batch_size, schedule,
                                               args.n_cov)
        for ai_, alpha in enumerate(alphas):
            z, phi_ = clt_projection(
                Ai2, bi2, x_star2, initial2, batch_size, schedule, alpha,
                args.n_cov, args.n0_cov, cov2, gamma=0.0,
                phi=None, seed=seed, A_seq=A_seq, b_seq=b_seq)
            cover_sgd[seed, ai_] = 1.0 if abs(z) <= z1_96 else 0.0
            for gi_, gamma in enumerate(args.gammas):
                zb, _ = clt_projection(
                    Ai2, bi2, x_star2, initial2, batch_size, schedule, alpha,
                    args.n_cov, args.n0_cov, cov2, gamma=gamma,
                    phi=phi_, seed=seed, A_seq=A_seq, b_seq=b_seq)
                cover_sgdm[seed, gi_, ai_] = 1.0 if abs(zb) <= z1_96 else 0.0

    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
    ax.plot(args.exps, cover_sgd.mean(axis=0), label="SGD", marker=MARKERS[0],
            markevery=0.3, linestyle=LINESTYLES[0])
    for gi_, gamma in enumerate(args.gammas):
        ax.plot(args.exps, cover_sgdm[:, gi_, :].mean(axis=0),
                label=f"SGDM-{gamma}", marker=MARKERS[gi_ + 1],
                markevery=0.3, linestyle=LINESTYLES[gi_ + 1])
    ax.axhline(y=0.95, color="black", linestyle="--")
    ax.set_xlabel(r"$\alpha=2^{x}$")
    ax.set_ylabel(r"$P(|Z|<1.96)$")
    ax.set_xticks(list(range(-11, 3)))
    ax.set_xlim([-11.3, 2.8])
    # The authors' tick list carries an extra 0.95 (the nominal level), which
    # is also what makes matplotlib label the axis with two decimals.
    ax.set_yticks([i * 0.2 for i in range(6)] + [0.95])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "fig_clt_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "clt_raw.npz"),
                        z_sgd=z_sgd, z_sgdm=z_sgdm,
                        cover_sgd=cover_sgd, cover_sgdm=cover_sgdm,
                        exps=np.array(args.exps), gammas=np.array(args.gammas))
    print(f"[clt] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
