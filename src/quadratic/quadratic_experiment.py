"""
Figure 2 (fig:linear): performance of SGD and SGDM on the quadratic loss.
Figure 3 (fig:average) / Figure 4 (fig:average2): averaged SGD and SGDM.

Reproduces Section 6.1 of the paper.

Setting
-------
N = 20,000, d = 10, rho = 1 (so that the average condition number is 35/10),
batch size B = 0.2 N, learning rate alpha = 0.001, K = 1500 iterations,
200 independent replications.  The adaptive momentum weight is
gamma = ((1 - mu*alpha)/(1 + mu*alpha))^2, averaged value ~ 0.95.

The (a) "small gamma" panels use gamma in {0.5, 0.7, 0.9}; the (b) "large
gamma" panels use gamma in {0.97, 0.98, 0.99} (see Section 6.1 of the paper).

Outputs (written to results/quadratic/):
    fig_linear_small.png / fig_linear_large.png          -> Figure 2(a),(b)
    fig_average_{small,large}_n0_200.png                 -> Figure 3(a),(b)
    fig_average_{small,large}_n0_500.png                 -> Figure 4(a),(b)
    quadratic_raw.npz                                    -> raw trajectories

Run:  python src/quadratic/quadratic_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    adaptive_gamma, averaged_errors, generate_quadratic_data,
    last_iterate_errors, make_batch_schedule, run_sgd, run_sgdm,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", "-", ":"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=1500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--n0_list", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--ave_len", type=int, default=1000)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "quadratic"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K = args.n, args.p, args.K
    batch_size = int(round(args.batch_frac * n))
    # make sure every n0 has room for the averaged curve
    for n0 in args.n0_list:
        if n0 + args.ave_len > K:
            raise ValueError(
                f"ave_len={args.ave_len} plus n0={n0} exceeds K={K}; "
                f"increase --K or reduce --ave_len.")

    # beta_set: fixed small / large momentum weights + adaptive (last slot)
    # The paper's Figure 2/3/4(b) ("large gamma") uses gamma = 0.97, 0.98, 0.99
    # (not 0.7/0.9): with K = 1500 these are the weights that still oscillate
    # visibly, matching the published panels.
    beta_fixed_small = [0.5, 0.7, 0.9]
    beta_fixed_large = [0.97, 0.98, 0.99]

    # error[seed, K] for SGD; error_m[seed, K, n_beta] for SGDM
    err_sgd = np.zeros((args.num_seed, K))
    err_m_small = np.zeros((args.num_seed, K, len(beta_fixed_small) + 1))
    err_m_large = np.zeros((args.num_seed, K, len(beta_fixed_large) + 1))
    ave_sgd = {n0: np.zeros((args.num_seed, args.ave_len))
               for n0 in args.n0_list}
    ave_m_small = {n0: np.zeros((args.num_seed, args.ave_len,
                                 len(beta_fixed_small) + 1))
                   for n0 in args.n0_list}
    ave_m_large = {n0: np.zeros((args.num_seed, args.ave_len,
                                 len(beta_fixed_large) + 1))
                   for n0 in args.n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[quadratic] seed {seed}/{args.num_seed}", flush=True)
        data = generate_quadratic_data(n, p, seed, rho=1.0, shift=10.0)
        Ai, bi, A, b = data["Ai"], data["bi"], data["A"], data["b"]
        x_star, mu, initial = data["x_star"], data["mu"], data["initial"]
        gamma = adaptive_gamma(mu, args.alpha)
        gamma_mean[seed] = gamma

        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)

        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                       args.alpha, K)
        err_sgd[seed] = last_iterate_errors(traj, x_star)
        for n0 in args.n0_list:
            ave_sgd[n0][seed] = averaged_errors(traj, x_star, n0, args.ave_len)

        for grp, betas, err_store, ave_store in (
                ("small", beta_fixed_small, err_m_small, ave_m_small),
                ("large", beta_fixed_large, err_m_large, ave_m_large)):
            for j, beta in enumerate(betas + [gamma]):
                traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size,
                                  schedule, args.alpha, beta, K)
                err_store[seed, :, j] = last_iterate_errors(traj_m, x_star)
                for n0 in args.n0_list:
                    ave_store[n0][seed, :, j] = averaged_errors(
                        traj_m, x_star, n0, args.ave_len)

    print(f"[quadratic] mean adaptive gamma = {gamma_mean.mean():.4f}")

    # ------------------------------------------------------------------ #
    # Figure 2: last-iterate ||x_t - x^*||
    #
    # Legend order / colours follow the published panels:
    #   (a) small gamma : SGD, SGDM-0.5, SGDM-0.7, SGDM-0.9, SGDM-adap
    #   (b) large gamma : SGDM-adap, SGDM-0.97, SGDM-0.98, SGDM-0.99
    # ------------------------------------------------------------------ #
    def _legend_order(betas):
        """Return (order, is_adap) for each entry, paper-style.

        For the small-gamma panel SGD comes first and SGDM-adap last; for the
        large-gamma panel SGDM-adap comes first (matching q-large.png).
        """
        if betas is beta_fixed_large:
            return [("adap", True)] + [(b, False) for b in betas]
        return [("sgd", False)] + [(b, False) for b in betas] + [("adap", True)]

    def plot_last_iterate(betas, err_m, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.17, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        for j, (key, is_adap) in enumerate(_legend_order(betas)):
            if key == "sgd":
                ax.plot(err_sgd.mean(axis=0) ** 2, label="SGD",
                        marker=MARKERS[0], markevery=100,
                        linestyle=LINESTYLES[3], color=COLORS[4], zorder=5)
            elif is_adap:
                ax.plot(err_m[:, :, -1].mean(axis=0) ** 2, label="SGDM-adap",
                        marker=MARKERS[-1], markevery=100, linestyle="--",
                        color=COLORS[0], zorder=6)
            else:
                ax.plot(err_m[:, :, betas.index(key)].mean(axis=0) ** 2,
                        label=f"SGDM-{key}", marker=MARKERS[j],
                        markevery=100, linestyle=LINESTYLES[j % 3],
                        color=COLORS[j])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||x_t-x^*||^2$")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 1e-3])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    plot_last_iterate(beta_fixed_small, err_m_small, "fig_linear_small.png")
    plot_last_iterate(beta_fixed_large, err_m_large, "fig_linear_large.png")

    # ------------------------------------------------------------------ #
    # Figures 3 & 4: averaged SGD / SGDM, split into (a) small / (b) large
    # gamma panels exactly as in the paper.
    # ------------------------------------------------------------------ #
    def plot_averaged(n0, betas, ave_m, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.17, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        for j, (key, is_adap) in enumerate(_legend_order(betas)):
            if key == "sgd":
                ax.plot(ave_sgd[n0].mean(axis=0) ** 2, label="SGD",
                        marker=MARKERS[0], markevery=100,
                        linestyle=LINESTYLES[3], color=COLORS[4], zorder=5)
            elif is_adap:
                ax.plot(ave_m[n0][:, :, -1].mean(axis=0) ** 2,
                        label="SGDM-adap", marker=MARKERS[-1], markevery=100,
                        linestyle="--", color=COLORS[0], zorder=6)
            else:
                ax.plot(ave_m[n0][:, :, betas.index(key)].mean(axis=0) ** 2,
                        label=f"SGDM-{key}", marker=MARKERS[j], markevery=100,
                        linestyle=LINESTYLES[j % 3], color=COLORS[j])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||\bar{x}_{n_0+t}-x^*||^2$")
        ax.set_yscale("log")
        ax.set_ylim([1e-10, 1e-3])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # Figures 3 & 4 (fig:average, fig:average2): one pair of panels per n0.
    # By default --n0_list = 200 500, reproducing Fig. 3 (n0 = 200) and
    # Fig. 4 (n0 = 500), each as (a) small gamma and (b) large gamma.
    for n0 in args.n0_list:
        plot_averaged(n0, beta_fixed_small, ave_m_small,
                      f"fig_average_small_n0_{n0}.png")
        plot_averaged(n0, beta_fixed_large, ave_m_large,
                      f"fig_average_large_n0_{n0}.png")

    arrays = dict(
        err_sgd=err_sgd, err_m_small=err_m_small, err_m_large=err_m_large,
        gamma_mean=gamma_mean,
        beta_small=np.array(beta_fixed_small),
        beta_large=np.array(beta_fixed_large))
    for n0 in args.n0_list:
        arrays[f"ave_sgd_{n0}"] = ave_sgd[n0]
        arrays[f"ave_m_small_{n0}"] = ave_m_small[n0]
        arrays[f"ave_m_large_{n0}"] = ave_m_large[n0]
    np.savez_compressed(os.path.join(args.out, "quadratic_raw.npz"), **arrays)
    print(f"[quadratic] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
