"""
Figure 16: SGD and SGDM on CIFAR-10 with a VGG-11 network.

Setting
-------
CIFAR-10 (Krizhevsky et al. 2009): N = 50,000 training images of size 32x32
with three colour channels, 10 classes.  The loss is f_i(X) = H(VGG(X), b_i),
the multi-class cross-entropy.  Learning rate alpha = 0.5 and momentum weights
gamma in {0.0 (SGD), 0.7, 0.9, 0.99}.  As elsewhere, alpha and gamma are
realised with ``SGD(lr = alpha * (1 - gamma), momentum = gamma)``.

The stored logs behind Figure 16 run 50 epochs with 391 mini-batches per epoch,
i.e. 19,550 iterations with a batch of 128; those are the defaults here.  (The
paper's text quotes B = 256 for this experiment, which would give 195 batches
per epoch.)

This is a non-convex problem: SGD performs well early but eventually diverges,
whereas SGDM with gamma in {0.7, 0.9, 0.99} converges; unusually, the largest
momentum (0.99) accelerates convergence.

Reporting convention
--------------------
Following the authors' plotting routine, the curve drawn at iteration t is the
**running mean of the per-iteration training losses over iterations 1..t**,
averaged over the three seeds, with diverged runs (non-finite losses) replaced
by 1000 before averaging and the result clipped to [-10, 100].

Note
----
The authors' original CIFAR training script was not preserved, so this script
(and ``vgg.py``) is a reconstruction from the paper's description and the
stored training logs (learning rate, epochs, batches per epoch, momentum
weights and the reporting rule).

It reproduces the paper's *qualitative* conclusion -- SGD is by far the least
stable method (per-iteration loss spikes to ~39, against ~10 for SGDM-0.9) and
SGDM with gamma = 0.99 converges fastest -- but **not** the published SGD
blow-up: in the paper SGD runs away in the last few hundred iterations, whereas
here it keeps descending. The remaining difference is a recipe detail that the
lost script would have pinned down (initialisation, weight decay, whether the
crop/flip augmentation is used, or the exact learning-rate convention).

Batch normalisation is required for this experiment: without it, SGD at
alpha = 0.5 makes no progress at all (the loss stays at 2.303), which does not
match the paper's "SGD performs well at the early stage".

Data
----
CIFAR-10 is downloaded (python batches, ~170 MB) on first use with
``--download`` and cached under ``data/CIFAR10/``.

Outputs (results/cifar/):
    fig_cifar.png
    cifar_losses.npz

Run:  python src/cifar/cifar_experiment.py --download [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from vgg import VGG11  # noqa: E402

# Canonical source is the Toronto page; the BOS mirror is a fast byte-identical
# copy that is much quicker outside North America.  Tried in order.
CIFAR_URLS = [
    "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

# Figure 16 styling (SGD, SGDM-0.7, SGDM-0.9, SGDM-0.99).
MARKERS = ["P", "o", "v", "s"]
LINESTYLES = ["-.", "-", "--", ":"]
COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

DIVERGED = 1000.0        # non-finite loss -> this value before averaging
CLIP = (-10.0, 100.0)    # the authors' clip on the plotted curve


def load_cifar10(root, download=False):
    """Load the CIFAR-10 training set as (X, y) uint8/int64, without torchvision."""
    base = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(base):
        os.makedirs(root, exist_ok=True)
        if not download:
            raise FileNotFoundError(
                f"{base} not found. Pass --download or place the extracted "
                f"CIFAR-10 python batches under {root}.")
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            last = None
            for url in CIFAR_URLS:
                try:
                    print(f"[cifar] downloading {url}")
                    urllib.request.urlretrieve(url, tgz)
                    last = None
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    print(f"[cifar]   failed: {e}")
            if last is not None:
                raise RuntimeError(f"could not download CIFAR-10: {last}")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(root)

    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(base, f"data_batch_{i}"), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.asarray(d[b"labels"]))
    X = np.concatenate(xs).astype(np.uint8)
    y = np.concatenate(ys).astype(np.int64)
    return X, y


def _augment_batch(batch_u8, rng):
    """Random crop with 4-pixel padding + random horizontal flip (uint8)."""
    b, c, h, w = batch_u8.shape
    padded = np.pad(batch_u8, ((0, 0), (0, 0), (4, 4), (4, 4)))
    out = np.empty_like(batch_u8)
    offs_x = rng.integers(0, 9, size=b)
    offs_y = rng.integers(0, 9, size=b)
    for i in range(b):
        out[i] = padded[i, :, offs_x[i]:offs_x[i] + h, offs_y[i]:offs_y[i] + w]
    flip = rng.random(b) < 0.5
    out[flip] = out[flip][:, :, :, ::-1]
    return out


def cumulative_mean(a):
    """Running mean: out[t] = mean(a[0], ..., a[t])."""
    a = np.asarray(a, dtype=float)
    return np.cumsum(a) / np.arange(1, len(a) + 1)


def run_one(seed, gamma, lr, epochs, batch_size, X, y, device, max_iter=None):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    num_batches = int(round(n / batch_size))
    mean = torch.from_numpy(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(CIFAR_STD).view(1, 3, 1, 1)

    net = VGG11(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr * (1 - gamma), momentum=gamma)

    losses = []
    t = 0
    for _ in range(epochs):
        for _ in range(num_batches):
            choice = rng.integers(0, n, size=batch_size)
            xb = _augment_batch(X[choice], rng)
            inputs = (torch.from_numpy(xb).float() / 255.0 - mean) / std
            targets = torch.from_numpy(y[choice])
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            t += 1
            if max_iter is not None and t >= max_iter:
                break
        if max_iter is not None and t >= max_iter:
            break
    return np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.7, 0.9, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128,
                    help="391 mini-batches per epoch with B = 128, which is "
                         "what the logs behind Figure 16 use")
    ap.add_argument("--max_iter", type=int, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "..", "results", "cifar"))
    args = ap.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(_HERE, "..", "..", "data", "CIFAR10")
    os.makedirs(args.out, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cifar] device = {device}")

    X, y = load_cifar10(args.data_root, download=args.download)
    print(f"[cifar] loaded {X.shape[0]} training images")

    gamma_all = [0.0] + list(args.gamma)
    curves = {}
    for gamma in gamma_all:
        runs = []
        for seed in args.seeds:
            print(f"[cifar] gamma={gamma} seed={seed}", flush=True)
            r = run_one(seed, gamma, args.lr, args.epochs, args.batch_size,
                        X, y, device, max_iter=args.max_iter)
            runs.append(np.where(np.isnan(r), DIVERGED, r))
        L = min(len(r) for r in runs)
        # Diverged replications are replaced by 1000 before averaging, then the
        # running mean is clipped -- the authors' plotting recipe.
        mean_loss = np.mean([r[:L] for r in runs], axis=0)
        run_mean = cumulative_mean(mean_loss)
        curves[gamma] = np.clip(np.where(np.isnan(run_mean), CLIP[1], run_mean),
                                CLIP[0], CLIP[1])

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15, figure=fig)[0])
    for j, gamma in enumerate(gamma_all):
        label = "SGD" if gamma == 0.0 else f"SGDM-{gamma}"
        yy = curves[gamma]
        ax.plot(np.arange(len(yy)), yy, label=label, marker=MARKERS[j],
                markevery=max(1, len(yy) // 6),
                linestyle=LINESTYLES[j % 4], color=COLORS[j])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Training Loss")
    ax.set_ylim((0.2, 2.6))
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "fig_cifar.png"), bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "cifar_losses.npz"),
                        **{f"gamma_{g}": curves[g] for g in gamma_all})
    print(f"[cifar] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
