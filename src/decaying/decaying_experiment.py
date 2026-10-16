"""
Figure 7: decaying learning rates.

Setting
-------
Quadratic loss (Section 5.1): N = 20,000, d = 10, A_i = V_i^T V_i + 10 I_d,
batch size B = 0.2 N, n = 2000 iterations, 200 replications, momentum weight
gamma = 0.96.  The learning rate decays as

    alpha_t = alpha0 * sqrt(n0 / max(n0, t)),       alpha0 = 0.001,

i.e. alpha0 up to the burn-in and alpha0 * sqrt(n0/t) afterwards, which is the
paper's alpha_t = alpha0 (max{n0, t})^{-1/2} up to the constant sqrt(n0).
This schedule does not require knowing the horizon n in advance.

Panel layout
------------
    (a) fig_decay_n0_250.png -- n0 = 250
    (b) fig_decay_n0_500.png -- n0 = 500
Each panel shows SGD, SGDM-0.96, Averaged SGD and Averaged SGDM-0.96.

Outputs (results/decaying/):
    fig_decay_n0_250.png / fig_decay_n0_500.png
    decaying_raw.npz

Run:  python src/decaying/decaying_experiment.py --num_seed 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgdm_core import (  # noqa: E402
    averaged_errors, batch_gradient_sequence, build_quadratic_data,
    decaying_schedule, draw_quadratic_problem, last_iterate_errors,
    make_batch_schedule, run_sgd, run_sgdm,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# (label, colour, marker, linestyle) per curve, in the paper's order.
CURVES = [("SGD", 0, 0, ":"),
          ("SGDM-0.96", 1, 1, "--"),
          ("Ave-SGD", 2, 0, "-"),
          ("Ave-SGDM-0.96", 3, 1, "-")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha0", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--n0_list", type=int, nargs="+", default=[250, 500])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "decaying"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d, n = args.N, args.d, args.n
    batch_size = int(round(args.batch_frac * N))
    n0_list = sorted(args.n0_list)
    if n0_list[-1] >= n:
        raise ValueError(f"n0={n0_list[-1]} must be smaller than n={n}.")

    err_sgd = np.zeros((args.num_seed, n))
    err_sgdm = np.zeros((args.num_seed, n))
    ave_sgd = {n0: np.zeros((args.num_seed, n - n0 - 1)) for n0 in n0_list}
    ave_sgdm = {n0: np.zeros((args.num_seed, n - n0 - 1)) for n0 in n0_list}

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[decaying] seed {seed}/{args.num_seed}", flush=True)
        rng = np.random.RandomState(seed)
        V, bi_r, u = draw_quadratic_problem(N, d, rng)
        data = build_quadratic_data(V, bi_r, u, rho=1.0, shift=10.0)
        Ai, bi = data["Ai"], data["bi"]
        x_star, initial = data["x_star"], data["initial"]
        schedule = make_batch_schedule(N, batch_size, n, seed, replace=False,
                                       rng=rng)
        # The batches do not depend on n0, so the batch gradients are formed
        # once per replication and replayed by every panel / method.
        A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, n)

        # the schedule depends on n0, so each panel needs its own run
        for n0 in n0_list:
            alphas = decaying_schedule(n0, n, args.alpha0)

            traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                           alphas, n, A_seq=A_seq, b_seq=b_seq)
            ave_sgd[n0][seed] = averaged_errors(traj, x_star, n0, n - n0)
            if n0 == n0_list[0]:
                err_sgd[seed] = last_iterate_errors(traj, x_star)

            traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                              alphas, args.gamma, n, A_seq=A_seq, b_seq=b_seq)
            ave_sgdm[n0][seed] = averaged_errors(traj_m, x_star, n0, n - n0)
            if n0 == n0_list[0]:
                err_sgdm[seed] = last_iterate_errors(traj_m, x_star)

    # ------------------------------------------------------------------ #
    # One panel per n0.  The last-iterate curves are the same across panels
    # in the paper's layout; here they are recomputed per panel so that the
    # stored arrays are self-consistent.
    # ------------------------------------------------------------------ #
    def _panel(n0, fname):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        spec = gridspec.GridSpec(1, 1, left=0.1, right=0.9, top=0.9,
                                 bottom=0.2, figure=fig)
        ax = fig.add_subplot(spec[0])
        for label, ci, mi, ls in CURVES:
            if label == "SGD":
                y, x = (err_sgd ** 2).mean(axis=0), np.arange(n)
            elif label == "SGDM-0.96":
                y, x = (err_sgdm ** 2).mean(axis=0), np.arange(n)
            elif label == "Ave-SGD":
                y = (ave_sgd[n0] ** 2).mean(axis=0)
                x = np.arange(n0 + 1, n0 + 1 + y.size)
            else:
                y = (ave_sgdm[n0] ** 2).mean(axis=0)
                x = np.arange(n0 + 1, n0 + 1 + y.size)
            ax.plot(x, y, label=label, marker=MARKERS[mi], markevery=0.3,
                    linestyle=ls, color=COLORS[ci])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_2$-error")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 2e-2])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    for n0 in n0_list:
        _panel(n0, f"fig_decay_n0_{n0}.png")

    arrays = dict(err_sgd=err_sgd, err_sgdm=err_sgdm,
                  n0_list=np.array(n0_list), gamma=args.gamma,
                  alpha0=args.alpha0)
    for n0 in n0_list:
        arrays[f"ave_sgd_{n0}"] = ave_sgd[n0]
        arrays[f"ave_sgdm_{n0}"] = ave_sgdm[n0]
    np.savez_compressed(os.path.join(args.out, "decaying_raw.npz"), **arrays)
    print(f"[decaying] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
