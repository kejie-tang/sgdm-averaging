"""
Figures 2-4: performance of SGD and SGDM (and their averaged versions) on the
quadratic loss.

Setting
-------
N = 20,000, d = 10, rho = 1 (so that the average condition number is 35/10),
batch size B = 0.2 N, learning rate alpha = 0.001, K = 1500 iterations,
200 independent replications.  The adaptive momentum weight is
gamma = ((1 - mu*alpha)/(1 + mu*alpha))^2, averaged value ~ 0.96.

The (a) "small gamma" panels use gamma in {0.5, 0.7, 0.9}; the (b) "large
gamma" panels use gamma in {0.97, 0.98, 0.99}.

Figures
-------
Figure 2  last iterate ||x_t - x^*||, small / large gamma.
Figure 3  averaged iterates ||xbar_{n0+t} - x^*|| with n0 = 500, small / large
          gamma (same momentum values as Figure 2).
Figure 4  SGD, SGDM (gamma = 0.96) and their averaged versions, with
          (a) n0 = 250 and (b) n0 = 500.

Outputs (written to results/quadratic/):
    fig_linear_{small,large}.png            -> Figure 2(a),(b)
    fig_average_{small,large}_n0_500.png    -> Figure 3(a),(b)
    fig_compare_n0_{250,500}.png            -> Figure 4(a),(b)
    quadratic_raw.npz                       -> raw trajectories

Run:  python src/quadratic/quadratic_experiment.py --num_seed 200
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
LINESTYLES = ["-.", "--", "-", ":"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Figure 3 uses a single burn-in (500); Figure 4 uses one panel per n0.
FIG3_N0 = 500
GAMMA_FIG4 = 0.96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=1500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--n0_list", type=int, nargs="+", default=[250, 500])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "quadratic"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K = args.n, args.p, args.K
    batch_size = int(round(args.batch_frac * n))
    n0_list = sorted(set(args.n0_list) | {FIG3_N0})
    for n0 in n0_list:
        if n0 >= K:
            raise ValueError(f"n0={n0} must be smaller than K={K}.")
    # The averaged curves run to the end of the trajectory, exactly as in the
    # published panels: xbar_{n0+t} for t = 1, ..., K - n0.
    ave_len = {n0: K - n0 for n0 in n0_list}

    # beta_set: fixed small / large momentum weights + adaptive (last slot).
    # The paper's "large gamma" panels use gamma = 0.97, 0.98, 0.99.
    beta_fixed_small = [0.5, 0.7, 0.9]
    beta_fixed_large = [0.97, 0.98, 0.99]

    err_sgd = np.zeros((args.num_seed, K))
    err_m_small = np.zeros((args.num_seed, K, len(beta_fixed_small) + 1))
    err_m_large = np.zeros((args.num_seed, K, len(beta_fixed_large) + 1))
    err_m_fig4 = np.zeros((args.num_seed, K))
    ave_sgd = {n0: np.zeros((args.num_seed, ave_len[n0]))
               for n0 in n0_list}
    ave_m_small = {n0: np.zeros((args.num_seed, ave_len[n0],
                                 len(beta_fixed_small) + 1))
                   for n0 in n0_list}
    ave_m_large = {n0: np.zeros((args.num_seed, ave_len[n0],
                                 len(beta_fixed_large) + 1))
                   for n0 in n0_list}
    ave_m_fig4 = {n0: np.zeros((args.num_seed, ave_len[n0]))
                  for n0 in n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[quadratic] seed {seed}/{args.num_seed}", flush=True)
        data = generate_quadratic_data(n, p, seed, rho=1.0, shift=10.0)
        Ai, bi, x_star, mu, initial = (data["Ai"], data["bi"], data["x_star"],
                                       data["mu"], data["initial"])
        gamma = adaptive_gamma(mu, args.alpha)
        gamma_mean[seed] = gamma

        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)
        # Every (gamma, alpha) pair replays the same mini-batches, so the batch
        # gradients are formed once per replication instead of once per sweep
        # entry.  The iterates are unchanged.
        A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, K)

        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                       args.alpha, K, A_seq=A_seq, b_seq=b_seq)
        err_sgd[seed] = last_iterate_errors(traj, x_star)
        for n0 in n0_list:
            ave_sgd[n0][seed] = averaged_errors(traj, x_star, n0, ave_len[n0])

        for betas, err_store, ave_store in (
                (beta_fixed_small, err_m_small, ave_m_small),
                (beta_fixed_large, err_m_large, ave_m_large)):
            for j, beta in enumerate(betas + [gamma]):
                traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size,
                                  schedule, args.alpha, beta, K,
                                  A_seq=A_seq, b_seq=b_seq)
                err_store[seed, :, j] = last_iterate_errors(traj_m, x_star)
                for n0 in n0_list:
                    ave_store[n0][seed, :, j] = averaged_errors(
                        traj_m, x_star, n0, ave_len[n0])

        # SGDM with the fixed weight 0.96 used by Figure 4
        traj_96 = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                           args.alpha, GAMMA_FIG4, K, A_seq=A_seq, b_seq=b_seq)
        err_m_fig4[seed] = last_iterate_errors(traj_96, x_star)
        for n0 in n0_list:
            ave_m_fig4[n0][seed] = averaged_errors(traj_96, x_star, n0,
                                                   ave_len[n0])

    print(f"[quadratic] mean adaptive gamma = {gamma_mean.mean():.4f}")

    # ------------------------------------------------------------------ #
    # Colours / markers / line styles are the ones used in the published
    # panels (read off Figures 2-4), so that a regenerated figure is
    # recognisable next to the paper's.
    # ------------------------------------------------------------------ #
    STYLE = {
        "SGD":           dict(color="#9467bd", marker="o", linestyle=":"),
        "SGDM-adap":     dict(color="#d62728", marker="P", linestyle="-"),
        "SGDM-0.5":      dict(color="#1f77b4", marker="v", linestyle="--"),
        "SGDM-0.7":      dict(color="#ff7f0e", marker="s", linestyle="--"),
        "SGDM-0.9":      dict(color="#2ca02c", marker="D", linestyle=":"),
        "SGDM-0.96":     dict(color="#ff7f0e", marker="v", linestyle="--"),
        "SGDM-0.97":     dict(color="#1f77b4", marker="v", linestyle="--"),
        "SGDM-0.98":     dict(color="#ff7f0e", marker="s", linestyle="--"),
        "SGDM-0.99":     dict(color="#2ca02c", marker="D", linestyle=":"),
        "Ave-SGD":       dict(color="#2ca02c", marker="o", linestyle="-"),
        "Ave-SGDM-0.96": dict(color="#d62728", marker="v", linestyle="-"),
    }

    # ------------------------------------------------------------------ #
    # Figure 2: last-iterate ||x_t - x^*||.
    #
    # Legend order follows the published panels:
    #   (a) small gamma : SGD, SGDM-0.5, SGDM-0.7, SGDM-0.9, SGDM-adap
    #   (b) large gamma : SGDM-adap, SGDM-0.97, SGDM-0.98, SGDM-0.99
    # ------------------------------------------------------------------ #
    def _legend_order(betas):
        if betas is beta_fixed_large:
            return [("adap", True)] + [(b, False) for b in betas]
        return [("sgd", False)] + [(b, False) for b in betas] + [("adap", True)]

    def plot_last_iterate(betas, err_m, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.17, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        for key, is_adap in _legend_order(betas):
            if key == "sgd":
                y, lab = err_sgd.mean(axis=0) ** 2, "SGD"
            elif is_adap:
                y = err_m[:, :, -1].mean(axis=0) ** 2
                lab = "SGDM-adap"
            else:
                y = err_m[:, :, betas.index(key)].mean(axis=0) ** 2
                lab = f"SGDM-{key}"
            ax.plot(np.arange(K), y, label=lab, markevery=200, **STYLE[lab])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||x_t-x^*||^2$")
        ax.set_yscale("log")
        ax.set_xlim([0, 800])
        ax.set_ylim([1e-9, 1e-1])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    plot_last_iterate(beta_fixed_small, err_m_small, "fig_linear_small.png")
    plot_last_iterate(beta_fixed_large, err_m_large, "fig_linear_large.png")

    # ------------------------------------------------------------------ #
    # Figure 3: averaged SGD / SGDM with n0 = 500 (one pair of panels).
    # Here the horizontal axis is the averaging count t, as in the paper.
    # ------------------------------------------------------------------ #
    def plot_averaged(n0, betas, ave_m, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.17, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        for key, is_adap in _legend_order(betas):
            if key == "sgd":
                y, lab = ave_sgd[n0].mean(axis=0) ** 2, "SGD"
            elif is_adap:
                y = ave_m[n0][:, :, -1].mean(axis=0) ** 2
                lab = "SGDM-adap"
            else:
                y = ave_m[n0][:, :, betas.index(key)].mean(axis=0) ** 2
                lab = f"SGDM-{key}"
            ax.plot(np.arange(1, ave_len[n0] + 1), y, label=lab, markevery=250,
                    **STYLE[lab])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||\bar{x}_{n_0+t}-x^*||^2$")
        ax.set_yscale("log")
        ax.set_ylim([1e-10, 1e-7])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    plot_averaged(FIG3_N0, beta_fixed_small, ave_m_small,
                  f"fig_average_small_n0_{FIG3_N0}.png")
    plot_averaged(FIG3_N0, beta_fixed_large, ave_m_large,
                  f"fig_average_large_n0_{FIG3_N0}.png")

    # ------------------------------------------------------------------ #
    # Figure 4: SGD, SGDM-0.96 and their averaged versions, one panel per n0.
    # The averaged curves are drawn on the absolute iteration axis, as in the
    # paper.
    # ------------------------------------------------------------------ #
    def plot_compare(n0, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.19, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        ax.plot(np.arange(K), err_sgd.mean(axis=0) ** 2, label="SGD",
                markevery=250, **STYLE["SGD"])
        ax.plot(np.arange(K), err_m_fig4.mean(axis=0) ** 2, label="SGDM-0.96",
                markevery=250, **STYLE["SGDM-0.96"])
        xa = np.arange(n0 + 1, K + 1)
        ax.plot(xa, ave_sgd[n0].mean(axis=0) ** 2, label="Ave-SGD",
                markevery=250, **STYLE["Ave-SGD"])
        ax.plot(xa, ave_m_fig4[n0].mean(axis=0) ** 2, label="Ave-SGDM-0.96",
                markevery=250, **STYLE["Ave-SGDM-0.96"])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_2$-error")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 1e-2])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    for n0 in sorted(args.n0_list):
        plot_compare(n0, f"fig_compare_n0_{n0}.png")

    arrays = dict(
        err_sgd=err_sgd, err_m_small=err_m_small, err_m_large=err_m_large,
        err_m_fig4=err_m_fig4, gamma_mean=gamma_mean,
        beta_small=np.array(beta_fixed_small),
        beta_large=np.array(beta_fixed_large))
    for n0 in n0_list:
        arrays[f"ave_sgd_{n0}"] = ave_sgd[n0]
        arrays[f"ave_m_small_{n0}"] = ave_m_small[n0]
        arrays[f"ave_m_large_{n0}"] = ave_m_large[n0]
        arrays[f"ave_m_fig4_{n0}"] = ave_m_fig4[n0]
    np.savez_compressed(os.path.join(args.out, "quadratic_raw.npz"), **arrays)
    print(f"[quadratic] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
