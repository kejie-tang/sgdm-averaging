"""
Figures 13-15: neural networks on MNIST (tanh activation).

Model
-----
Two-layer perceptron with a tanh hidden layer, 784 -> 500 -> 10, trained with
the multi-class cross-entropy loss (see ``nn_models.py`` for how this maps onto
the paper's f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)).  This is a *non-convex*
problem and complements the multinomial logistic regression of Figure EC.4.

Setting
-------
MNIST, N = 60,000 images of size 28x28 reshaped to 784-dim vectors, inputs in
[0, 1].  Batch size B = 256, learning rate alpha = 1.0, 5 epochs, seeds 1, 2, 3,
and momentum weights gamma in {0.0 (SGD), 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}.  As
in the paper, the momentum weight beta and the learning rate alpha are realised
with ``SGD(lr = alpha * (1 - beta), momentum = beta)``.

Reporting convention (important)
--------------------------------
The curves are *not* moving averages with a window of one epoch.  Following the
authors' plotting routine, the value reported at iteration t is the **running
mean of the per-iteration training loss over iterations 1..t**, averaged over
the three seeds.  For the averaged algorithms (Figures 14-15) the sequence fed
to that running mean is the loss of the Polyak-averaged parameters
xbar_{n0+t} = (1/t) Sum_{j=n0+1}^{n0+t} x_j, and before n0 it is the ordinary
mini-batch loss.  Figures 14 and 15 plot this sequence over t = n0, ..., 1000
for n0 = 234 = floor(N/B) and n0 = 468 = 2 floor(N/B).

Panels
------
    fig_tanh_small.png / fig_tanh_large.png              -> Figure 13(a),(b)
    fig_avetanh_n0_234_{small,large}.png                 -> Figure 14(a),(b)
    fig_avetanh_n0_468_{small,large}.png                 -> Figure 15(a),(b)
The "small gamma" panels use SGD, SGDM-0.3, SGDM-0.5; the "large gamma" panels
use SGDM-0.5, 0.7, 0.9, 0.99 and contain no SGD curve.

Data
----
MNIST is read directly from the standard IDX files (no torchvision needed):
put them under ``data/MNIST/raw/`` or pass ``--download``.

Outputs (results/mnist_tanh/):
    the six PNGs above, plus mnist_tanh_losses.npz

Run:  python src/mnist_tanh/mnist_tanh_experiment.py [--device cuda] [--download]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "mnist"))
from mnist_experiment import load_mnist  # noqa: E402
from nn_models import Net  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

# Styling of the published panels (Figures 13-15), which is the same for the
# last-iterate and the averaged panels.
SMALL = dict(colors=['#1f77b4', '#2ca02c', '#d62728'],
             markers=['*', 'P', '*'], linestyles=['-', '--', '-'])
LARGE = dict(colors=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'],
             markers=['P', 'p', 'P', '*'], linestyles=['-', ':', '--', '-'])


def cumulative_mean(a):
    """Running mean: out[t] = mean(a[0], ..., a[t])."""
    a = np.asarray(a, dtype=float)
    return np.cumsum(a) / np.arange(1, len(a) + 1)


def run_one(seed, gamma, lr, num_epochs, batch_size, X, y, n0_list, device,
            max_iter=None):
    """Train one (seed, gamma) pair.

    Returns ``(last, ave)`` where ``last[t]`` is the mini-batch loss at
    iteration t and ``ave[n0][t]`` is the loss of the averaged parameters for
    t >= n0 (and the ordinary mini-batch loss before that), matching the
    authors' ``train_losses`` / ``train_losses_ave`` arrays.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = X.shape[0]
    num_batches = n // batch_size
    Xt = torch.from_numpy(X.astype(np.float32) / 255.0)
    yt = torch.from_numpy(y)

    net = Net(784, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr * (1 - gamma),
                          momentum=gamma)

    running = {n0: {k: torch.zeros_like(v)
                    for k, v in net.named_parameters()} for n0 in n0_list}
    counts = {n0: 0 for n0 in n0_list}
    net_ave = {n0: Net(784, 10).to(device) for n0 in n0_list}

    last = []
    ave_tail = {n0: [] for n0 in n0_list}   # averaged loss for t >= n0
    t = 0
    for _ in range(1, num_epochs + 1):
        for _ in range(num_batches):
            choice = np.random.choice(n, batch_size, replace=False)
            inputs = Xt[choice].to(device)
            targets = yt[choice].to(device)

            # Averaged parameters (built from the *pre-update* weights) and the
            # loss they achieve on this iteration's mini-batch.
            with torch.no_grad():
                for n0 in n0_list:
                    if t >= n0:
                        c = counts[n0]
                        for name, v in net.named_parameters():
                            running[n0][name] = (
                                (c / (c + 1)) * running[n0][name]
                                + v.detach() / (c + 1))
                        counts[n0] = c + 1
                        for name, v in net_ave[n0].named_parameters():
                            v.copy_(running[n0][name])
                        ave_tail[n0].append(
                            criterion(net_ave[n0](inputs), targets).item())

            optimizer.zero_grad()
            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()
            last.append(loss.item())
            t += 1
            if max_iter is not None and t >= max_iter:
                break
        if max_iter is not None and t >= max_iter:
            break

    # Before n0 the averaged curve is the ordinary mini-batch loss
    # (the authors' "else" branch); from n0 on it is ave_tail.
    ave = {n0: np.asarray(list(last[:n0]) + ave_tail[n0], dtype=float)
           for n0 in n0_list}
    return np.asarray(last), ave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n0_list", type=int, nargs="+", default=[234, 468])
    ap.add_argument("--max_iter", type=int, default=1000,
                    help="number of iterations that are plotted along t")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "..", "results",
                                         "mnist_tanh"))
    args = ap.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(_HERE, "..", "..", "data", "MNIST")
    raw_dir = os.path.join(args.data_root, "raw")
    os.makedirs(args.out, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mnist-tanh] device = {device}")

    X, y = load_mnist(raw_dir, download=args.download)
    print(f"[mnist-tanh] loaded {X.shape[0]} training images")

    gamma_all = [0.0] + list(args.gamma)
    last_curves, ave_curves = {}, {}
    for gamma in gamma_all:
        lasts, aves = [], {n0: [] for n0 in args.n0_list}
        for seed in args.seeds:
            print(f"[mnist-tanh] gamma={gamma} seed={seed}", flush=True)
            last, ave = run_one(seed, gamma, args.lr, args.epochs,
                                args.batch_size, X, y, args.n0_list, device,
                                max_iter=args.max_iter)
            lasts.append(last)
            for n0 in args.n0_list:
                aves[n0].append(ave[n0])
        L = min(len(a) for a in lasts)
        last_curves[gamma] = cumulative_mean(
            np.mean([a[:L] for a in lasts], axis=0))
        ave_curves[gamma] = {}
        for n0 in args.n0_list:
            La = min(len(a) for a in aves[n0])
            ave_curves[gamma][n0] = cumulative_mean(
                np.mean([a[:La] for a in aves[n0]], axis=0))

    T = min(len(c) for c in last_curves.values())

    def draw(fname, betas, getter, lo, hi, ylim, include_sgd, style,
             prefix=""):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(gridspec.GridSpec(
            1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15,
            figure=fig)[0])
        j = 0
        for beta in ([0.0] if include_sgd else []) + list(betas):
            y = getter(beta)
            if y is None:
                continue
            if beta == 0.0:
                label = "SGD" if not prefix else "Ave-SGD (Out of Plot)"
            else:
                label = f"{prefix}SGDM-{beta}"
            x = np.arange(lo, hi)
            ax.plot(x, y[lo:hi], label=label, marker=style["markers"][j],
                    markevery=max(1, len(x) // 8), markersize=6,
                    linestyle=style["linestyles"][j],
                    color=style["colors"][j])
            j += 1
        ax.set_xlabel(r"$t$" if lo == 0 else r"$n_0+t$")
        ax.set_ylabel("Training Loss")
        ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # Figure 13: last iterate, t = 0 .. 999.
    draw("fig_tanh_small.png", [0.3, 0.5], last_curves.get, 0, T,
         (0.2, 2.6), True, SMALL)
    draw("fig_tanh_large.png", [0.5, 0.7, 0.9, 0.99], last_curves.get, 0, T,
         (0.2, 2.6), False, LARGE)

    # Figures 14 and 15: averaged parameters, plotted from t = n0.
    for n0 in args.n0_list:
        def ave_getter(beta, _n0=n0):
            return ave_curves.get(beta, {}).get(_n0)

        draw(f"fig_avetanh_n0_{n0}_small.png", [0.3, 0.5], ave_getter, n0, T,
             (0.20, 0.65), True, SMALL, prefix="Ave-")
        draw(f"fig_avetanh_n0_{n0}_large.png", [0.5, 0.7, 0.9, 0.99],
             ave_getter, n0, T, (0.20, 0.65), False, LARGE, prefix="Ave-")

    arrays = {}
    for g in gamma_all:
        arrays[f"last_{g}"] = last_curves[g]
        for n0 in args.n0_list:
            arrays[f"ave_{n0}_{g}"] = ave_curves[g][n0]
    np.savez_compressed(os.path.join(args.out, "mnist_tanh_losses.npz"),
                        **arrays)
    print(f"[mnist-tanh] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
