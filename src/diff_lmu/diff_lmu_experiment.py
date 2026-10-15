"""
Figure 6: performance under different condition numbers L/mu.

Setting
-------
Quadratic loss with A_i = rho * V_i^T V_i + 10 I_d for rho in {1, 5, 10}.
The realised condition numbers L/mu are 3.3, 9.9 and 14.7; the printed labels
are the paper's 3.5, 9.9 and 14.7 (see ``KAPPA_LABELS`` and README.md,
section 7).  Otherwise the setup of Section 6.1: N = 20,000, d = 10, batch
size B = 0.2 N, learning rate
alpha = 0.001, K = 1500 iterations, 200 replications.  The momentum weight is
the *adaptive* weight gamma = ((1 - mu alpha)/(1 + mu alpha))^2 computed from
each replication's mu; its averages over the 200 replications are about
0.96, 0.95 and 0.93 for rho = 1, 5, 10.

The burn-in for the averaged iterates is n0 = 250.

Panel layout
------------
    (a) fig_lmu_sgd.png  -- SGD and Averaged SGD, one pair per L/mu
    (b) fig_lmu_sgdm.png -- SGDM-adap and Averaged SGDM-adap, one pair per L/mu

Outputs (results/diff_lmu/):
    fig_lmu_sgd.png / fig_lmu_sgdm.png
    lmu_raw.npz
    (the mean adaptive momentum weight per rho is printed to stdout)

Run:  python src/diff_lmu/diff_lmu_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    adaptive_gamma, averaged_errors, batch_gradient_sequence,
    generate_quadratic_data, last_iterate_errors, make_batch_schedule,
    run_sgd, run_sgdm,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']

# The legend labels of the published Figure 6.  The authors' plotting cell
# hardcodes them (``kappa_set = [3.5, 9.9, 14.7]``); they are not read off the
# data.  The realised values, which the script prints, are 3.3 / 9.9 / 14.7 --
# the rho = 5 and rho = 10 entries agree to one decimal, the rho = 1 entry does
# not (see README.md, section 7).  The labels below are what the paper's
# caption and legend say, so the released figure matches the published one.
KAPPA_LABELS = [3.5, 9.9, 14.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=1500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--n0", type=int, default=250)
    ap.add_argument("--rho_list", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "diff_lmu"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K = args.n, args.p, args.K
    n0 = args.n0
    if n0 >= K:
        raise ValueError(f"n0={n0} must be smaller than K={K}.")
    batch_size = int(round(args.batch_frac * n))
    rho_list = list(args.rho_list)
    ave_len = K - n0 - 1

    err_sgd = np.zeros((args.num_seed, K, len(rho_list)))
    err_sgdm = np.zeros((args.num_seed, K, len(rho_list)))
    ave_sgd = np.zeros((args.num_seed, ave_len, len(rho_list)))
    ave_sgdm = np.zeros((args.num_seed, ave_len, len(rho_list)))
    gamma_mean = np.zeros((args.num_seed, len(rho_list)))
    kappa_mean = np.zeros(len(rho_list))

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[diff-Lmu] seed {seed}/{args.num_seed}", flush=True)
        # The mini-batch schedule depends only on the seed, not on rho.
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)
        for j, rho in enumerate(rho_list):
            data = generate_quadratic_data(n, p, seed, rho=rho, shift=10.0)
            Ai, bi = data["Ai"], data["bi"]
            x_star, mu, initial = data["x_star"], data["mu"], data["initial"]
            if seed == 0:
                kappa_mean[j] = data["L"] / mu
            gamma = adaptive_gamma(mu, args.alpha)
            gamma_mean[seed, j] = gamma
            # SGD and SGDM share the batches, so the batch gradients are formed
            # once per (replication, rho) and replayed by both methods.
            A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size,
                                                   schedule, K)

            traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                           args.alpha, K, A_seq=A_seq, b_seq=b_seq)
            err_sgd[seed, :, j] = last_iterate_errors(traj, x_star)
            ave_sgd[seed, :, j] = averaged_errors(traj, x_star, n0, ave_len)

            traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                              args.alpha, gamma, K, A_seq=A_seq, b_seq=b_seq)
            err_sgdm[seed, :, j] = last_iterate_errors(traj_m, x_star)
            ave_sgdm[seed, :, j] = averaged_errors(traj_m, x_star, n0, ave_len)

    for j, rho in enumerate(rho_list):
        print(f"[diff-Lmu] rho={rho:g}: L/mu = {kappa_mean[j]:.2f}, "
              f"mean adaptive gamma = {gamma_mean[:, j].mean():.4f}")

    # ------------------------------------------------------------------ #
    # One panel per method; each rho contributes a dotted last-iterate curve
    # and a solid averaged curve (paper's Figure 6).  Panel (a) runs
    # blue-orange-green, panel (b) orange-green-red.
    # ------------------------------------------------------------------ #
    def _panel(fname, err, ave, last_label, ave_label, colors, marker):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.19, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        # The published panels list the last-iterate curves first and the
        # averaged ones after, so the two loops are separate.
        for j, rho in enumerate(rho_list):
            cond = KAPPA_LABELS[j] if j < len(KAPPA_LABELS) else kappa_mean[j]
            ax.plot(np.arange(K), (err[:, :, j] ** 2).mean(axis=0),
                    label=rf"{last_label}-$L/\mu={cond:.1f}$",
                    marker=marker, markevery=250,
                    linestyle=":", color=colors[j])
        for j, rho in enumerate(rho_list):
            cond = KAPPA_LABELS[j] if j < len(KAPPA_LABELS) else kappa_mean[j]
            ya = (ave[:, :, j] ** 2).mean(axis=0)
            ax.plot(np.arange(n0 + 1, n0 + 1 + ya.size), ya,
                    label=rf"{ave_label}-$L/\mu={cond:.1f}$",
                    marker=marker, markevery=250,
                    linestyle="-", color=colors[j], zorder=5)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_2$-error")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 1e-2])
        ax.legend(fontsize=6.5)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # Panel (a) uses circles, panel (b) triangles, as in the published figure.
    _panel("fig_lmu_sgd.png", err_sgd, ave_sgd, "SGD", "Ave-SGD",
           ["#1f77b4", "#ff7f0e", "#2ca02c"], "o")
    _panel("fig_lmu_sgdm.png", err_sgdm, ave_sgdm, "SGDM-adap", "Ave-SGDM-adap",
           ["#ff7f0e", "#2ca02c", "#d62728"], "v")

    np.savez_compressed(
        os.path.join(args.out, "lmu_raw.npz"),
        err_sgd=err_sgd, err_sgdm=err_sgdm,
        ave_sgd=ave_sgd, ave_sgdm=ave_sgdm,
        gamma_mean=gamma_mean, kappa_mean=kappa_mean,
        rho_list=np.array(rho_list), n0=n0)
    print(f"[diff-Lmu] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
