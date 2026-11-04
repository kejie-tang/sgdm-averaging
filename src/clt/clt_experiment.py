"""
Figures 8 and 12: asymptotic normality of the averaged SGD / SGDM estimator.

Setting
-------
Quadratic loss, N = 20,000, d = 10, B = 0.2 N.
The statistic (Eq. (def:z))
    Z = sqrt(B) / (sigma * sqrt(omega^T Sigma^{-1} Omega Sigma^{-1} omega))
        * sum_{t=n0+1}^{n} omega^T (x_t - x^*) / sqrt(n - n0)
is asymptotically N(0, 1) by Corollary 3.

Figure 8 : full-data reference normalization, n=2000, n0=1000.
Figure 12: Remark 8 sampled-gradient covariance, n=1000, n0=500.
Both default to independent sampling WITH replacement, as in Corollary 3.
--sampling without-replacement applies the finite-population correction.
--covariance oracle provides the full-data reference comparison for Figure 12.

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
    make_batch_schedule, draw_quadratic_problem, build_quadratic_data,
    run_sgd, run_sgdm, online_projected_variances,
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



def coverage_trial(data, schedule, batch_size, alphas, gammas, n, n0,
                   covariance="online", replace=True):
    """One full Figure-12 sweep; no selection or dropping of failed trials."""
    Ai, bi = data["Ai"], data["bi"]
    xs, initial = data["x_star"], data["initial"]
    A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, n)
    phi = data.get("phi", np.ones_like(xs) / np.sqrt(len(xs)))
    trajectories = []
    with np.errstate(over="ignore", invalid="ignore"):
        for gamma in gammas:
            for alpha in alphas:
                trajectories.append(run_sgdm(
                    Ai, bi, xs, initial, batch_size, schedule, alpha, gamma, n,
                    A_seq=A_seq, b_seq=b_seq))
        trajectories = np.asarray(trajectories)
        numerator = np.sqrt(n - n0) * (
            (trajectories[:, n0:].mean(axis=1) - xs)[..., 0] @ phi[:, 0])
    if covariance == "online":
        # A_i >= (1.03/1.04) I for this experiment: this is a known
        # construction bound, not an oracle estimate of the empirical mu.
        variances = online_projected_variances(
            Ai, bi, trajectories, batch_size, schedule, phi, 1.03 / 1.04,
            replace=replace, A_seq=A_seq)
    elif covariance == "oracle":
        cov = asymptotic_covariance(
            Ai, bi, xs, data["A"], batch_size, replace=replace)
        variances = np.full(len(trajectories), float((phi.T @ cov @ phi).item()))
    else:
        raise ValueError(f"Unknown covariance method: {covariance}")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = numerator / np.sqrt(variances)
    invalid = ~np.isfinite(trajectories).all(axis=(1, 2, 3)) | ~np.isfinite(z)
    z[invalid] = np.nan
    return z.reshape(len(gammas), len(alphas))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000, help="training-set size N")
    ap.add_argument("--d", type=int, default=10, help="dimension d of x")
    ap.add_argument("--num_seed", type=int, default=1000, help="number of independent replications")
    ap.add_argument("--batch_frac", type=float, default=0.2, help="mini-batch size as a fraction of N")
    # histogram experiment (Figure 8)
    ap.add_argument("--n_hist", type=int, default=2000)
    ap.add_argument("--n0_hist", type=int, default=1000)
    ap.add_argument("--gamma_hist", type=float, default=0.9)
    ap.add_argument("--n_cov", type=int, default=1000)
    ap.add_argument("--n0_cov", type=int, default=500)
    ap.add_argument("--num_seed_cov", type=int, default=1000)
    ap.add_argument("--exps", type=int, nargs="+", default=list(range(2, -12, -1)))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--sampling", choices=["with-replacement", "without-replacement"],
                    default="with-replacement",
                    help="independent batches; default matches Corollary 3")
    ap.add_argument("--covariance", choices=["online", "oracle"], default="online",
                    help="Figure 12 covariance: Remark 8 estimator or reference")
    ap.add_argument("--projection", choices=["ones", "random"], default="ones",
                    help="Figure 12: historical ones/sqrt(d), or a fixed seed-1 random direction")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "clt"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d = args.N, args.d
    batch_size = int(round(args.batch_frac * N))

    if not (0 <= args.n0_hist < args.n_hist and 0 <= args.n0_cov < args.n_cov):
        ap.error("Require 0 <= burn-in < trajectory length")
    replace = args.sampling == "with-replacement"
    rng = np.random.RandomState(1)
    data = build_quadratic_data(*draw_quadratic_problem(N, d, rng))
    Ai, bi, A = data["Ai"], data["bi"], data["A"]
    x_star, initial = data["x_star"], data["initial"]
    phi = rng.rand(d, 1)
    phi /= np.linalg.norm(phi)
    alpha_hist = 0.001
    cov = asymptotic_covariance(
        Ai, bi, x_star, A, batch_size, replace=replace)

    z_sgd = np.zeros(args.num_seed)
    z_sgdm = np.zeros(args.num_seed)
    for seed in range(args.num_seed):
        if seed % 100 == 0:
            print(f"[clt-hist] seed {seed}/{args.num_seed}", flush=True)
        schedule = make_batch_schedule(N, batch_size, args.n_hist, seed,
                                       replace=replace)
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

    rng2 = np.random.RandomState(1)
    data2 = build_quadratic_data(
        *draw_quadratic_problem(N, d, rng2), shift=1.03, norm_const=1.04)
    phi_cov = (rng2.rand(d, 1) if args.projection == "random"
               else np.ones((d, 1)))
    phi_cov /= np.linalg.norm(phi_cov)
    data2["phi"] = phi_cov
    alphas = [2.0 ** e for e in args.exps]
    gamma_all = [0.0] + list(args.gammas)
    z_cov = np.full((args.num_seed_cov, len(gamma_all), len(alphas)), np.nan)
    for seed in range(args.num_seed_cov):
        if seed % 20 == 0:
            print(f"[clt-cov] seed {seed}/{args.num_seed_cov}", flush=True)
        schedule = make_batch_schedule(
            N, batch_size, args.n_cov, seed, replace=replace)
        z_cov[seed] = coverage_trial(
            data2, schedule, batch_size, alphas, gamma_all,
            args.n_cov, args.n0_cov, args.covariance, replace)
    covered = np.isfinite(z_cov) & (np.abs(z_cov) <= 1.96)
    cover_sgd = covered[:, 0, :].astype(float)
    cover_sgdm = covered[:, 1:, :].astype(float)

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
                        exps=np.array(args.exps), gammas=np.array(args.gammas), z_cov=z_cov,
                        valid_cov=np.isfinite(z_cov), n_hist=args.n_hist,
                        n0_hist=args.n0_hist, n_cov=args.n_cov, n0_cov=args.n0_cov,
                        sampling=args.sampling, covariance=args.covariance,
                        projection=args.projection, phi_cov=phi_cov,
                        phi_hist=phi, N=N, d=d, batch_size=batch_size,
                        alpha_hist=alpha_hist, gamma_hist=args.gamma_hist,
                        data_seed=1, lambda_lower=1.03 / 1.04)
    print(f"[clt] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
