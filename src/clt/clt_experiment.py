"""
Figures 8 and 12: asymptotic normality of the averaged SGD / SGDM estimator.

Setting
-------
Quadratic loss, N = 20,000, d = 10, B = 0.2 N.
Figure 8 uses A_i=V_i^T V_i+10I; Figure 12 uses A_i=V_i^T V_i+I.
The projected statistic (Eq. (def:z)) is
    Z = sqrt(B * (n - n0)) * omega^T (xbar - x^*) / sqrt(omega^T C omega).
Figure 8 uses the full-data reference C; Figure 12 substitutes the sampled
Remark 8 sandwich estimator C_hat = Sigma_tilde^{-1} S_hat Sigma_tilde^{-1}.
The simulations examine its finite-sample normal approximation.

Figure 8 uses the full-data reference covariance evaluated at x^*:
n=2000, n0=1000, independent batches sampled with replacement.
Figure 12 uses the sampled-gradient sandwich estimator of Remark 8:
n=500, n0=150, independent batches sampled with replacement, 1000 trials.
Both experiments use a fixed unit random projection, shared across trials,
rates and methods. Figure 12 uses every pre-update x_t, including burn-in,
to estimate S_hat and Sigma_hat; it does not evaluate them at x^*.
The known construction bound A_i >= I gives lambda_Sigma=1 for thresholding.
Failed or non-finite statistics count as non-coverage.

Outputs (results/clt/):
    fig_clt_sgd.png, fig_clt_sgdm.png, fig_clt_alpha.png, clt_raw.npz

Run:  python src/clt/clt_experiment.py --figure 12
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    asymptotic_covariance, batch_gradient_sequence, clt_projection,
    make_batch_schedule, draw_quadratic_problem, build_quadratic_data,
    run_sgdm, online_projected_variances,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "-", "--", ":"]


def hist_figure(z, gamma_label, fname, out_dir, color="#1f77b4"):
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=2,
            label="N(0,1)")
    ax.hist(z, range=(-5, 5), bins=20, density=True, rwidth=0.8,
            color=color, label=gamma_label)
    ax.set_ylim([0, 0.45])
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)



def coverage_figure(exps, gammas, cover_sgd, cover_sgdm, out_dir):
    """Figure 12 coverage with the nominal 95% reference level."""
    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
    ax.plot(exps, cover_sgd.mean(axis=0), label="SGD", marker=MARKERS[0],
            linestyle=LINESTYLES[0])
    for gi_, gamma in enumerate(gammas):
        ax.plot(exps, cover_sgdm[:, gi_, :].mean(axis=0),
                label=f"SGDM-{gamma}", marker=MARKERS[gi_ + 1],
                linestyle=LINESTYLES[gi_ + 1])
    ax.axhline(y=0.95, color="black", linestyle="--")
    ax.set_xlabel(r"$\alpha=2^{x}$")
    ax.set_ylabel(r"$P(|Z|<1.96)$")
    ax.set_xticks(list(range(-11, 3)))
    ax.set_xlim([-11.3, 2.8])
    ax.set_yticks([i * 0.2 for i in range(6)] + [0.95])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(fontsize=8, loc="lower center")
    fig.savefig(os.path.join(out_dir, "fig_clt_alpha.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_clt_alpha.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def coverage_trial(data, schedule, batch_size, alphas, gammas, n, n0,
                   return_variances=False):
    """Figure-12 statistics using the Remark 8 covariance, for one trial."""
    Ai, bi = data["Ai"], data["bi"]
    xs, initial = data["x_star"], data["initial"]
    A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, n)
    phi = data["phi"]
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
    variances = online_projected_variances(
        Ai, bi, trajectories, batch_size, schedule, phi,
        lambda_lower=1.0, A_seq=A_seq)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = numerator / np.sqrt(variances)
    invalid = ~np.isfinite(trajectories).all(axis=(1, 2, 3)) | ~np.isfinite(z)
    z[invalid] = np.nan
    z = z.reshape(len(gammas), len(alphas))
    if return_variances:
        return z, variances.reshape(z.shape), A_seq.mean(axis=0)
    return z


def _init_coverage_worker(N, d, batch_size, alphas, gammas, n, n0):
    global _COVERAGE_DATA, _COVERAGE_CONFIG
    rng = np.random.RandomState(1)
    _COVERAGE_DATA = build_quadratic_data(
        *draw_quadratic_problem(N, d, rng), shift=1.0)
    phi = rng.rand(d, 1)
    _COVERAGE_DATA["phi"] = phi / np.linalg.norm(phi)
    _COVERAGE_CONFIG = (N, batch_size, alphas, gammas, n, n0)


def _coverage_replication(seed):
    N, batch_size, alphas, gammas, n, n0 = _COVERAGE_CONFIG
    schedule = make_batch_schedule(N, batch_size, n, seed, replace=True)
    return seed, coverage_trial(
        _COVERAGE_DATA, schedule, batch_size, alphas, gammas, n, n0,
        return_variances=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=["all", "8", "12"], default="all")
    ap.add_argument("--N", type=int, default=20000, help="training-set size N")
    ap.add_argument("--d", type=int, default=10, help="dimension d of x")
    ap.add_argument("--num_seed", type=int, default=1000, help="number of independent replications")
    ap.add_argument("--batch_frac", type=float, default=0.2, help="mini-batch size as a fraction of N")
    # histogram experiment (Figure 8)
    ap.add_argument("--n_hist", type=int, default=2000)
    ap.add_argument("--n0_hist", type=int, default=1000)
    ap.add_argument("--gamma_hist", type=float, default=0.9)
    ap.add_argument("--n_cov", type=int, default=500)
    ap.add_argument("--n0_cov", type=int, default=150)
    ap.add_argument("--num_seed_cov", type=int, default=1000)
    ap.add_argument("--seed_start_cov", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exps", type=int, nargs="+", default=list(range(2, -12, -1)))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.8, 0.9])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "clt"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d = args.N, args.d
    batch_size = int(round(args.batch_frac * N))

    if not (0 <= args.n0_hist < args.n_hist and 0 <= args.n0_cov < args.n_cov):
        ap.error("Require 0 <= burn-in < trajectory length")
    if not (1 <= batch_size and args.workers >= 1):
        ap.error("Require positive batch size and worker count")
    sampling_hist = "with-replacement"
    sampling_cov = "with-replacement"
    rng = np.random.RandomState(1)
    data = build_quadratic_data(*draw_quadratic_problem(N, d, rng))
    Ai, bi, A = data["Ai"], data["bi"], data["A"]
    x_star, initial = data["x_star"], data["initial"]
    phi = rng.rand(d, 1)
    phi /= np.linalg.norm(phi)
    alpha_hist = 0.001
    z_sgd = z_sgdm = np.empty(0)
    if args.figure in ("all", "8"):
        cov = asymptotic_covariance(
            Ai, bi, x_star, A, batch_size, replace=True)
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

    if args.figure == "8":
        np.savez_compressed(os.path.join(args.out, "clt_hist_raw.npz"),
                            z_sgd=z_sgd, z_sgdm=z_sgdm,
                            n_hist=args.n_hist, n0_hist=args.n0_hist,
                            sampling_hist=sampling_hist, covariance_hist="oracle",
                            phi_hist=phi, N=N, d=d, batch_size=batch_size,
                            alpha_hist=alpha_hist, gamma_hist=args.gamma_hist,
                            data_seed=1)
        return
    rng2 = np.random.RandomState(1)
    data2 = build_quadratic_data(
        *draw_quadratic_problem(N, d, rng2), shift=1.0)
    phi_cov = rng2.rand(d, 1)
    phi_cov /= np.linalg.norm(phi_cov)
    data2["phi"] = phi_cov
    alphas = [2.0 ** e for e in args.exps]
    gamma_all = [0.0] + list(args.gammas)
    z_cov = np.full((args.num_seed_cov, len(gamma_all), len(alphas)), np.nan)
    variances_cov = np.full_like(z_cov, np.nan)
    hessians_cov = np.empty((args.num_seed_cov, d, d))
    seeds_cov = np.arange(args.seed_start_cov,
                          args.seed_start_cov + args.num_seed_cov)
    config = (N, d, batch_size, alphas, gamma_all, args.n_cov, args.n0_cov)
    _init_coverage_worker(*config)
    completed = 0
    def record(seed, result):
        nonlocal completed
        i = seed - args.seed_start_cov
        z_cov[i], variances_cov[i], hessians_cov[i] = result
        completed += 1
        if completed % 50 == 0 or completed == args.num_seed_cov:
            print(f"[clt-cov] completed {completed}/{args.num_seed_cov}", flush=True)
    if args.workers == 1:
        for seed in seeds_cov:
            seed, result = _coverage_replication(int(seed))
            record(seed, result)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_coverage_worker, initargs=config) as pool:
            jobs = [pool.submit(_coverage_replication, int(seed)) for seed in seeds_cov]
            for job in cf.as_completed(jobs):
                seed, result = job.result()
                record(seed, result)
    covered = np.isfinite(z_cov) & (np.abs(z_cov) <= 1.96)
    cover_sgd = covered[:, 0, :].astype(float)
    cover_sgdm = covered[:, 1:, :].astype(float)

    coverage_figure(args.exps, args.gammas, cover_sgd, cover_sgdm, args.out)

    np.savez_compressed(os.path.join(args.out, "clt_raw.npz"),
                        z_sgd=z_sgd, z_sgdm=z_sgdm,
                        cover_sgd=cover_sgd, cover_sgdm=cover_sgdm,
                        exps=np.array(args.exps), gammas=np.array(args.gammas), z_cov=z_cov,
                        valid_cov=np.isfinite(z_cov), n_hist=args.n_hist,
                        n0_hist=args.n0_hist, n_cov=args.n_cov, n0_cov=args.n0_cov,
                        sampling="per-figure",
                        sampling_hist=sampling_hist, sampling_cov=sampling_cov,
                        covariance="per-figure", covariance_hist="oracle",
                        covariance_cov="Remark 8 plug-in", projection="fixed",
                        phi_cov=phi_cov, projected_variances_cov=variances_cov,
                        hessian_estimates_cov=hessians_cov, seeds_cov=seeds_cov,
                        lambda_lower=1.0, covariance_uses_burn_in=True,
                        phi_hist=phi, N=N, d=d, batch_size=batch_size,
                        alpha_hist=alpha_hist, gamma_hist=args.gamma_hist,
                        data_seed=1, hessian_cov="V.T@V+I")
    print(f"[clt] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
