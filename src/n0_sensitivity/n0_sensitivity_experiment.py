"""
Figure 5: sensitivity of the averaged iterates to the burn-in n0.

Setting
-------
Quadratic loss (Section 6.1 of the paper): N = 20,000, d = 10,
A_i = V_i^T V_i + 10 I_d (rho = 1, so the average condition number is L/mu =
3.5), batch size B = 0.2 N, learning rate alpha = 0.001, K = 1500 iterations,
200 independent replications.  The momentum weight is fixed at gamma = 0.96
(the average of the adaptive weight over the replications).

The averaged iterate is xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j, and the
figure compares four burn-ins n0 in {250, 500, 750, 1000}.

Panel layout
------------
    (a) fig_n0_sgd.png  -- SGD and Averaged SGD for n0 = 250, 500, 750, 1000
    (b) fig_n0_sgdm.png -- SGDM-0.96 and Averaged SGDM for the same n0 values

Outputs (results/n0_sensitivity/):
    fig_n0_sgd.png / fig_n0_sgdm.png
    n0_raw.npz

Run:  python src/n0_sensitivity/n0_sensitivity_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    averaged_errors, batch_gradient_sequence, generate_quadratic_data,
    last_iterate_errors, make_batch_schedule, run_sgd, run_sgdm,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--K", type=int, default=1500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--n0_list", type=int, nargs="+",
                    default=[250, 500, 750, 1000])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results",
                                         "n0_sensitivity"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n, p, K = args.n, args.p, args.K
    batch_size = int(round(args.batch_frac * n))
    n0_list = sorted(args.n0_list)
    if n0_list[-1] >= K:
        raise ValueError(f"n0={n0_list[-1]} must be smaller than K={K}.")

    err_sgd = np.zeros((args.num_seed, K))
    err_sgdm = np.zeros((args.num_seed, K))
    # per n0 the averaged curve has length K - n0 (t = n0+1, ..., K)
    ave_sgd = {n0: np.zeros((args.num_seed, K - n0 - 1)) for n0 in n0_list}
    ave_sgdm = {n0: np.zeros((args.num_seed, K - n0 - 1)) for n0 in n0_list}

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[n0] seed {seed}/{args.num_seed}", flush=True)
        data = generate_quadratic_data(n, p, seed, rho=1.0, shift=10.0)
        Ai, bi = data["Ai"], data["bi"]
        x_star, initial = data["x_star"], data["initial"]
        schedule = make_batch_schedule(n, batch_size, K, seed, replace=False)
        # SGD and SGDM replay the same batches; form the batch gradients once.
        A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, K)

        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                       args.alpha, K, A_seq=A_seq, b_seq=b_seq)
        err_sgd[seed] = last_iterate_errors(traj, x_star)
        for n0 in n0_list:
            ave_sgd[n0][seed] = averaged_errors(traj, x_star, n0, K - n0)

        traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                          args.alpha, args.gamma, K, A_seq=A_seq, b_seq=b_seq)
        err_sgdm[seed] = last_iterate_errors(traj_m, x_star)
        for n0 in n0_list:
            ave_sgdm[n0][seed] = averaged_errors(traj_m, x_star, n0, K - n0)

    # ------------------------------------------------------------------ #
    # Plotting.  Panel (a) has no momentum, panel (b) is SGDM-0.96; both
    # show the last iterate plus one averaged curve per n0.  Colours and
    # markers follow the published Figure 5: the reference curve is blue
    # (SGD) / orange (SGDM-0.96) and the four averaged curves run
    # green-red-purple-brown in (a) and red-purple-brown-pink in (b).
    # ------------------------------------------------------------------ #
    def _panel(fname, err, ave, label_last, ave_label, ref_style, ave_colors):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.17, right=0.95, top=0.95,
                                 bottom=0.15, figure=fig)
        ax = fig.add_subplot(spec[0])
        ax.plot(np.arange(K), (err ** 2).mean(axis=0), label=label_last,
                markevery=250, **ref_style)
        for j, n0 in enumerate(n0_list):
            y = (ave[n0] ** 2).mean(axis=0)
            ax.plot(np.arange(n0 + 1, n0 + 1 + y.size), y,
                    label=rf"{ave_label}-$n_0$={n0}",
                    marker=ref_style["marker"], markevery=250,
                    linestyle="-", color=ave_colors[j])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_2$-error")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 1e-2])
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    _panel("fig_n0_sgd.png", err_sgd, ave_sgd, "SGD", "Ave-SGD",
           dict(color="#1f77b4", marker="o", linestyle=":"),
           ["#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    _panel("fig_n0_sgdm.png", err_sgdm, ave_sgdm, f"SGDM-{args.gamma}",
           "Ave-SGDM",
           dict(color="#ff7f0e", marker="v", linestyle="-."),
           ["#d62728", "#9467bd", "#8c564b", "#e377c2"])

    arrays = dict(err_sgd=err_sgd, err_sgdm=err_sgdm,
                  n0_list=np.array(n0_list), gamma=args.gamma)
    for n0 in n0_list:
        arrays[f"ave_sgd_{n0}"] = ave_sgd[n0]
        arrays[f"ave_sgdm_{n0}"] = ave_sgdm[n0]
    np.savez_compressed(os.path.join(args.out, "n0_raw.npz"), **arrays)
    print(f"[n0] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
