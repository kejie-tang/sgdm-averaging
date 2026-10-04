"""
Figures 5 and 9: asymptotic normality of the averaged SGD / SGDM estimator.

Setting
-------
Quadratic loss, N = 20,000, d = 10, B = 0.2 N.
The statistic (Eq. (def:z))
    Z = sqrt(B) / (sigma * sqrt(omega^T Sigma^{-1} Omega Sigma^{-1} omega))
        * sum_{t=n0+1}^n omega^T (x_t - x^*) / sqrt(n - n0)
is asymptotically N(0, 1) by Corollary 3.

Figure 5 : frequency of Y for averaged SGD and averaged SGDM, gamma = 0.9,
           n0 = 1000, n = 2000.
Figure 9 : empirical P(|Z| < 1.96) as a function of the learning rate for SGD
           and SGDM (gamma = 0.8, 0.9).

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
    asymptotic_covariance, clt_projection, generate_quadratic_data,
    make_batch_schedule,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]


def hist_figure(z, gamma_label, fname, out_dir):
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.15, right=0.97, top=0.95, bottom=0.15, figure=fig)[0])
    ax.hist(z, range=(-5, 5), bins=20, density=True, rwidth=0.8,
            label=gamma_label)
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=2,
            label="N(0,1)")
    ax.set_ylim([0, 0.45])
    ax.set_xlabel(r"$Z$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--num_seed", type=int, default=1000)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    # histogram experiment
    ap.add_argument("--K_hist", type=int, default=2000)
    ap.add_argument("--n0_hist", type=int, default=1000)
    ap.add_argument("--gamma_hist", type=float, default=0.9)
    # coverage experiment
    ap.add_argument("--K_cov", type=int, default=500)
    ap.add_argument("--n0_cov", type=int, default=200)
    ap.add_argument("--num_seed_cov", type=int, default=1000)
    ap.add_argument("--exps", type=int, nargs="+",
                    default=[-1, -3, -5, -7, -9])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "clt"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p = args.n, args.p
    batch_size = int(round(args.batch_frac * n))

    # --- shared problem instance for the histogram experiment ---
    rng = np.random.RandomState(1)
    V = rng.rand(n, p, p)
    Ai = np.empty((n, p, p))
    for i in range(n):
        Vi = V[i]
        Ai[i] = Vi @ Vi.T + 10.0 * np.eye(p)
    bi = rng.rand(n, p, 1)
    A = Ai.mean(axis=0)
    b = bi.mean(axis=0)
    x_star = np.linalg.solve(A, -b)
    noise = rng.rand(p, 1)
    noise /= np.linalg.norm(noise)
    initial = x_star + noise * 0.1
    ev = np.linalg.eigvalsh(A)
    mu = float(ev.min())
    alpha_hist = 0.001

    cov = asymptotic_covariance(Ai, bi, x_star, A, batch_size)
    rng_phi = np.random.RandomState(1)
    phi = rng_phi.rand(p, 1)
    phi /= np.linalg.norm(phi)

    z_sgd = np.zeros(args.num_seed)
    z_sgdm = np.zeros(args.num_seed)
    for seed in range(args.num_seed):
        if seed % 100 == 0:
            print(f"[clt-hist] seed {seed}/{args.num_seed}", flush=True)
        schedule = make_batch_schedule(n, batch_size, args.K_hist, seed,
                                       replace=True)
        z_sgd[seed], _ = clt_projection(
            Ai, bi, x_star, initial, batch_size, schedule, alpha_hist,
            args.K_hist, args.n0_hist, cov, gamma=0.0, phi=phi)
        z_sgdm[seed], _ = clt_projection(
            Ai, bi, x_star, initial, batch_size, schedule, alpha_hist,
            args.K_hist, args.n0_hist, cov, gamma=args.gamma_hist, phi=phi)

    print(f"[clt-hist] mean/std SGD  = {z_sgd.mean():.3f} / {z_sgd.std():.3f}")
    print(f"[clt-hist] mean/std SGDM = {z_sgdm.mean():.3f} / {z_sgdm.std():.3f}")

    hist_figure(z_sgd, "SGD", "fig_clt_sgd.png", args.out)
    hist_figure(z_sgdm, f"SGDM ($\\gamma$={args.gamma_hist})",
                "fig_clt_sgdm.png", args.out)

    # --- coverage vs. learning rate (Figure 9) ---
    rng2 = np.random.RandomState(1)
    V2 = rng2.rand(n, p, p)
    Ai2 = np.empty((n, p, p))
    for i in range(n):
        Vi = V2[i]
        Ai2[i] = (Vi @ Vi.T + 1.03 * np.eye(p)) / 1.04
    bi2 = rng2.rand(n, p, 1)
    A2 = Ai2.mean(axis=0)
    b2 = bi2.mean(axis=0)
    x_star2 = np.linalg.solve(A2, -b2)
    noise2 = rng2.rand(p, 1)
    noise2 /= np.linalg.norm(noise2)
    initial2 = x_star2 + noise2 * 0.1
    cov2 = asymptotic_covariance(Ai2, bi2, x_star2, A2, batch_size)

    alphas = [2.0 ** e for e in args.exps]
    n_a, n_b = len(alphas), len(args.betas)
    cover_sgd = np.zeros((args.num_seed_cov, n_a))
    cover_sgdm = np.zeros((args.num_seed_cov, n_b, n_a))
    z1_96 = 1.96

    for seed in range(args.num_seed_cov):
        if seed % 200 == 0:
            print(f"[clt-cov] seed {seed}/{args.num_seed_cov}", flush=True)
        schedule = make_batch_schedule(n, batch_size, args.K_cov, seed,
                                       replace=False)
        for ai_, alpha in enumerate(alphas):
            z, phi_ = clt_projection(
                Ai2, bi2, x_star2, initial2, batch_size, schedule, alpha,
                args.K_cov, args.n0_cov, cov2, gamma=0.0,
                phi=None, seed=seed)
            x_age = None  # coverage computed only on the aggregate Z
            cover_sgd[seed, ai_] = 1.0 if abs(z) <= z1_96 else 0.0
            for bi_, beta in enumerate(args.betas):
                zb, _ = clt_projection(
                    Ai2, bi2, x_star2, initial2, batch_size, schedule, alpha,
                    args.K_cov, args.n0_cov, cov2, gamma=beta,
                    phi=phi_, seed=seed)
                cover_sgdm[seed, bi_, ai_] = 1.0 if abs(zb) <= z1_96 else 0.0

    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.15, right=0.97, top=0.95, bottom=0.15, figure=fig)[0])
    ax.plot(args.exps, cover_sgd.mean(axis=0), label="SGD", marker=MARKERS[0],
            linestyle=LINESTYLES[0])
    for bi_, beta in enumerate(args.betas):
        ax.plot(args.exps, cover_sgdm[:, bi_, :].mean(axis=0),
                label=f"SGDM-{beta}", marker=MARKERS[bi_ + 1],
                linestyle=LINESTYLES[bi_ + 1])
    ax.axhline(y=0.95, color="black", linestyle="--")
    ax.set_xlabel(r"$\alpha=2^{x}$")
    ax.set_ylabel(r"$P(|Z|<1.96)$")
    ax.set_ylim([-0.01, 1.05])
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "fig_clt_alpha.png"),
                bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "clt_raw.npz"),
                        z_sgd=z_sgd, z_sgdm=z_sgdm,
                        cover_sgd=cover_sgd, cover_sgdm=cover_sgdm,
                        exps=np.array(args.exps), betas=np.array(args.betas))
    print(f"[clt] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
