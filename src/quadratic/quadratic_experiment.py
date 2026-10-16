"""
Figures 2-4: performance of SGD and SGDM (and their averaged versions) on the
quadratic loss.

Setting
-------
N = 20,000, d = 10, rho = 1 (so that the average condition number is 35/10),
batch size B = 0.2 N, learning rate alpha = 0.001, n = 1500 iterations,
200 independent replications.  The adaptive momentum weight is
gamma = ((1 - mu*alpha)/(1 + mu*alpha))^2, averaged value ~ 0.96.

The (a) "small gamma" panels use gamma in {0.5, 0.7, 0.9}; the (b) "large
gamma" panels use gamma in {0.97, 0.98, 0.99}.

Figures
-------
Figure 2  last iterate ||x_t - x^*||, small / large gamma.
Figure 3  averaged iterates ||xbar_{n0+t} - x^*|| with n0 = 500, small / large
          gamma (same momentum values as Figure 2).  The published (b) panel
          shows the gamma in {0.5, 0.7, 0.9} curves under the labels
          SGDM-0.97/0.98/0.99; see the note above ``plot_averaged``.
Figure 4  SGD, SGDM (gamma = 0.96) and their averaged versions, with
          (a) n0 = 250 and (b) n0 = 500.

Every curve is the mean over the replications of the *squared* error,
``mean_seed ||x - x^*||^2``, which is the quantity the authors' plotting
helpers compute (``np.mean(arr ** 2, axis=0)``).

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
    build_quadratic_data, draw_quadratic_problem, last_iterate_errors,
    make_batch_schedule, run_sgd, run_sgdm,
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

# The canvas of the published panels: 4 x 4 inches with the axes box spanning
# left=0.1 ... right=0.9 and bottom=0.2 ... top=0.9, i.e. a 3.2 x 2.8 inch box
# -- the geometry the authors' plotting cells use.  The figures are then saved
# with ``bbox_inches="tight"``, as the published panels were: measured on the
# paper, the axes box occupies about 0.79 of the image width and 0.82 of its
# height, which a raw 4x4 canvas (0.80 / 0.70) cannot produce.
FIGSIZE = (4, 4)
GRIDSPEC = dict(left=0.1, right=0.9, top=0.9, bottom=0.2)


def _canvas(figsize=FIGSIZE):
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(1, 1, figure=fig, **GRIDSPEC)[0])
    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20000)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--num_seed", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--batch_frac", type=float, default=0.2)
    ap.add_argument("--n0_list", type=int, nargs="+", default=[250, 500])
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "quadratic"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N, d, n = args.N, args.d, args.n
    batch_size = int(round(args.batch_frac * N))
    n0_list = sorted(set(args.n0_list) | {FIG3_N0})
    for n0 in n0_list:
        if n0 + 1 >= n:
            raise ValueError(f"n0={n0} must be smaller than n={n}.")
    # The averaged curves run to the end of the trajectory, exactly as in the
    # published panels: xbar_{n0+t} for t = 1, ..., n - n0 - 1 (the last
    # iterate that can be averaged in is x_{n-1}).
    ave_len = {n0: n - n0 - 1 for n0 in n0_list}

    # The two fixed momentum weight sets of Figure 2, plus the adaptive
    # weight gamma = ((1 - mu alpha)/(1 + mu alpha))^2 in the last slot.
    # The paper's "large gamma" panels use gamma = 0.97, 0.98, 0.99.
    gamma_fixed_small = [0.5, 0.7, 0.9]
    gamma_fixed_large = [0.97, 0.98, 0.99]

    err_sgd = np.zeros((args.num_seed, n))
    err_m_small = np.zeros((args.num_seed, n, len(gamma_fixed_small) + 1))
    err_m_large = np.zeros((args.num_seed, n, len(gamma_fixed_large) + 1))
    err_m_fig4 = np.zeros((args.num_seed, n))
    ave_sgd = {n0: np.zeros((args.num_seed, ave_len[n0]))
               for n0 in n0_list}
    ave_m_small = {n0: np.zeros((args.num_seed, ave_len[n0],
                                 len(gamma_fixed_small) + 1))
                   for n0 in n0_list}
    ave_m_large = {n0: np.zeros((args.num_seed, ave_len[n0],
                                 len(gamma_fixed_large) + 1))
                   for n0 in n0_list}
    ave_m_fig4 = {n0: np.zeros((args.num_seed, ave_len[n0]))
                  for n0 in n0_list}
    gamma_mean = np.zeros(args.num_seed)

    for seed in range(args.num_seed):
        if seed % 20 == 0:
            print(f"[quadratic] seed {seed}/{args.num_seed}", flush=True)
        # One unbroken random stream per replication: the data, the starting
        # point and the mini-batch indices are drawn from it in that order,
        # exactly as the authors' notebooks do.  (Drawing the schedule from a
        # freshly seeded stream instead -- as an earlier version of this script
        # did -- gives a statistically equivalent but different realisation.)
        rng = np.random.RandomState(seed)
        V, bi_r, u = draw_quadratic_problem(N, d, rng)
        data = build_quadratic_data(V, bi_r, u, rho=1.0, shift=10.0)
        Ai, bi, x_star, mu, initial = (data["Ai"], data["bi"], data["x_star"],
                                       data["mu"], data["initial"])
        gamma = adaptive_gamma(mu, args.alpha)
        gamma_mean[seed] = gamma

        schedule = make_batch_schedule(N, batch_size, n, seed, replace=False,
                                       rng=rng)
        # Every (gamma, alpha) pair replays the same mini-batches, so the batch
        # gradients are formed once per replication instead of once per sweep
        # entry.  The iterates are unchanged.
        A_seq, b_seq = batch_gradient_sequence(Ai, bi, batch_size, schedule, n)

        traj = run_sgd(Ai, bi, x_star, initial, batch_size, schedule,
                       args.alpha, n, A_seq=A_seq, b_seq=b_seq)
        err_sgd[seed] = last_iterate_errors(traj, x_star)
        for n0 in n0_list:
            ave_sgd[n0][seed] = averaged_errors(traj, x_star, n0, ave_len[n0])

        for gammas_fixed, err_store, ave_store in (
                (gamma_fixed_small, err_m_small, ave_m_small),
                (gamma_fixed_large, err_m_large, ave_m_large)):
            for j, gamma_fixed in enumerate(gammas_fixed + [gamma]):
                traj_m = run_sgdm(Ai, bi, x_star, initial, batch_size,
                                  schedule, args.alpha, gamma_fixed, n,
                                  A_seq=A_seq, b_seq=b_seq)
                err_store[seed, :, j] = last_iterate_errors(traj_m, x_star)
                for n0 in n0_list:
                    ave_store[n0][seed, :, j] = averaged_errors(
                        traj_m, x_star, n0, ave_len[n0])

        # SGDM with the fixed weight 0.96 used by Figure 4
        traj_96 = run_sgdm(Ai, bi, x_star, initial, batch_size, schedule,
                           args.alpha, GAMMA_FIG4, n, A_seq=A_seq, b_seq=b_seq)
        err_m_fig4[seed] = last_iterate_errors(traj_96, x_star)
        for n0 in n0_list:
            ave_m_fig4[n0][seed] = averaged_errors(traj_96, x_star, n0,
                                                   ave_len[n0])

    print(f"[quadratic] mean adaptive gamma = {gamma_mean.mean():.4f}")

    # ------------------------------------------------------------------ #
    # Colours / markers / line styles are the ones used in the published
    # panels (read off Figures 2-4), so that a regenerated figure is
    # recognisable next to the paper's.  The canvas is the authors' as well:
    # ``figsize=(4, 4)`` with the axes box spanning left=0.1 ... right=0.9
    # and bottom=0.2 ... top=0.9, and ``markevery=0.3``.  The float
    # ``markevery`` matters -- matplotlib spaces such markers by *arc length*
    # along the curve, so the marker positions follow from the axes-box size
    # (3.2 x 2.8 in) and are only reproducible with those parameters.
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
    def _legend_order(gammas_fixed):
        if gammas_fixed is gamma_fixed_large:
            return [("adap", True)] + [(g, False) for g in gammas_fixed]
        return [("sgd", False)] + [(g, False) for g in gammas_fixed] \
            + [("adap", True)]

    def plot_last_iterate(gammas_fixed, err_m, fname):
        fig, ax = _canvas()
        for key, is_adap in _legend_order(gammas_fixed):
            if key == "sgd":
                y, lab = (err_sgd ** 2).mean(axis=0), "SGD"
            elif is_adap:
                y = (err_m[:, :, -1] ** 2).mean(axis=0)
                lab = "SGDM-adap"
            else:
                y = (err_m[:, :, gammas_fixed.index(key)] ** 2).mean(axis=0)
                lab = f"SGDM-{key}"
            ax.plot(np.arange(n), y, label=lab, markevery=0.3, **STYLE[lab])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||x_t-x^*||^2$")
        ax.set_yscale("log")
        ax.set_xlim([0, 800])
        ax.set_ylim([1e-9, 1e-1])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    plot_last_iterate(gamma_fixed_small, err_m_small, "fig_linear_small.png")
    plot_last_iterate(gamma_fixed_large, err_m_large, "fig_linear_large.png")

    # ------------------------------------------------------------------ #
    # Figure 3: averaged SGD / SGDM with n0 = 500 (one pair of panels).
    # Here the horizontal axis is the averaging count t, as in the paper.
    #
    # NOTE on the (b) panel.  The published Figure 3(b) does *not* show the
    # gamma = 0.97/0.98/0.99 runs: it replays the gamma = 0.5/0.7/0.9 curves
    # of Figure 3(a) under those legend labels (the authors' plotting cell
    # indexes the momentum array with ``i`` instead of ``i + 4``).  The
    # published panel is reproduced faithfully here so that the figure matches
    # the paper; set FIG3B_HONEST = True to draw the curves the labels claim.
    # ------------------------------------------------------------------ #
    FIG3B_HONEST = False

    def plot_averaged(n0, spec, fname):
        """Draw averaged curves.  ``spec`` is a list of (label, getter) pairs
        in legend order; ``getter()`` returns the (num_seed, T) error norms."""
        fig, ax = _canvas()
        for lab, getter in spec:
            y = (getter() ** 2).mean(axis=0)      # mean over replications
            ax.plot(np.arange(1, len(y) + 1), y, label=lab, markevery=0.3,
                    **STYLE[lab])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$||\bar{x}_{n_0+t}-x^*||^2$")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 1e-7])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    n0 = FIG3_N0
    plot_averaged(n0, [
        ("SGD", lambda: ave_sgd[n0]),
        ("SGDM-0.5", lambda: ave_m_small[n0][:, :, 0]),
        ("SGDM-0.7", lambda: ave_m_small[n0][:, :, 1]),
        ("SGDM-0.9", lambda: ave_m_small[n0][:, :, 2]),
        ("SGDM-adap", lambda: ave_m_small[n0][:, :, 3]),
    ], f"fig_average_small_n0_{n0}.png")

    if FIG3B_HONEST:
        plot_averaged(n0, [
            ("SGDM-adap", lambda: ave_m_large[n0][:, :, 3]),
            ("SGDM-0.97", lambda: ave_m_large[n0][:, :, 0]),
            ("SGDM-0.98", lambda: ave_m_large[n0][:, :, 1]),
            ("SGDM-0.99", lambda: ave_m_large[n0][:, :, 2]),
        ], f"fig_average_large_n0_{n0}.png")
    else:
        plot_averaged(n0, [
            ("SGDM-adap", lambda: ave_m_small[n0][:, :, 3]),
            ("SGDM-0.97", lambda: ave_m_small[n0][:, :, 0]),
            ("SGDM-0.98", lambda: ave_m_small[n0][:, :, 1]),
            ("SGDM-0.99", lambda: ave_m_small[n0][:, :, 2]),
        ], f"fig_average_large_n0_{n0}.png")

    # ------------------------------------------------------------------ #
    # Figure 4: SGD, SGDM-0.96 and their averaged versions, one panel per n0.
    # The averaged curves are drawn on the absolute iteration axis, as in the
    # paper.
    # ------------------------------------------------------------------ #
    def plot_compare(n0, fname):
        fig, ax = _canvas()
        # Figure 4 draws SGD in blue (Figures 2-3 use the purple of STYLE).
        ax.plot(np.arange(n), (err_sgd ** 2).mean(axis=0), label="SGD",
                markevery=0.3, **dict(STYLE["SGD"], color="#1f77b4"))
        ax.plot(np.arange(n), (err_m_fig4 ** 2).mean(axis=0), label="SGDM-0.96",
                markevery=0.3, **STYLE["SGDM-0.96"])
        xa = np.arange(n0 + 1, n0 + 1 + ave_sgd[n0].shape[1])
        ax.plot(xa, (ave_sgd[n0] ** 2).mean(axis=0), label="Ave-SGD",
                markevery=0.3, **STYLE["Ave-SGD"])
        ax.plot(xa, (ave_m_fig4[n0] ** 2).mean(axis=0), label="Ave-SGDM-0.96",
                markevery=0.3, **STYLE["Ave-SGDM-0.96"])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$L_2$-error")
        ax.set_yscale("log")
        ax.set_ylim([1e-9, 2e-2])
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    for n0 in sorted(args.n0_list):
        plot_compare(n0, f"fig_compare_n0_{n0}.png")

    arrays = dict(
        err_sgd=err_sgd, err_m_small=err_m_small, err_m_large=err_m_large,
        err_m_fig4=err_m_fig4, gamma_mean=gamma_mean,
        gamma_small=np.array(gamma_fixed_small),
        gamma_large=np.array(gamma_fixed_large))
    for n0 in n0_list:
        arrays[f"ave_sgd_{n0}"] = ave_sgd[n0]
        arrays[f"ave_m_small_{n0}"] = ave_m_small[n0]
        arrays[f"ave_m_large_{n0}"] = ave_m_large[n0]
        arrays[f"ave_m_fig4_{n0}"] = ave_m_fig4[n0]
    np.savez_compressed(os.path.join(args.out, "quadratic_raw.npz"), **arrays)
    print(f"[quadratic] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
