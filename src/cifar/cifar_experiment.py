"""
Figure 16: SGD and SGDM on CIFAR-10 with a VGG-11 network.

Setting
-------
CIFAR-10 (Krizhevsky et al. 2009): N = 50,000 training images of size 32x32
with three colour channels, 10 classes.  The loss is f_i(X) = H(VGG(X), b_i),
the multi-class cross-entropy.  Batch size B = 256, learning rate alpha = 0.5,
and momentum weights gamma in {0.0 (SGD), 0.7, 0.9, 0.99} (the momentum values
used in the paper's figure).  As elsewhere, alpha and gamma are realised with
``SGD(lr = alpha * (1 - gamma), momentum = gamma)``.

This is a non-convex problem: SGD performs well early but eventually diverges,
whereas SGDM with gamma in {0.7, 0.9, 0.99} converges; unusually, the largest
momentum (0.99) accelerates convergence.

The reported training loss is the moving average of the past 2*floor(N/B)
mini-batch losses, matching the smoothing used in the authors' CIFAR logs.

Note
----
The authors' original CIFAR training script was not preserved, so this script
(and ``vgg.py``) is a faithful reconstruction from the paper's description and
the stored training logs.  The logs contain 9,800 iterations per run (about 50
epochs); that is the default here.

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
sys.path.insert(0, os.path.join(_HERE, "..", "mnist"))
from mnist_experiment import moving_average  # noqa: E402
from vgg import VGG11  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Canonical source is the Toronto page; the BOS mirror is a fast byte-identical
# copy that is much quicker outside North America.  Tried in order.
CIFAR_URLS = [
    "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


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
    num_batches = n // batch_size
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
    ap.add_argument("--batch_size", type=int, default=256)
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
    k = 2 * (X.shape[0] // args.batch_size)  # smoothing window used in the logs

    gamma_all = [0.0] + list(args.gamma)
    curves = {}
    for gamma in gamma_all:
        runs = []
        for seed in args.seeds:
            print(f"[cifar] gamma={gamma} seed={seed}", flush=True)
            runs.append(run_one(seed, gamma, args.lr, args.epochs,
                                args.batch_size, X, y, device,
                                max_iter=args.max_iter))
        L = min(len(r) for r in runs)
        curves[gamma] = np.mean([moving_average(r[:L], k) for r in runs],
                                axis=0)

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15, figure=fig)[0])
    for j, gamma in enumerate(gamma_all):
        label = "SGD" if gamma == 0.0 else f"SGDM-{gamma}"
        yy = curves[gamma]
        ax.plot(np.arange(len(yy)), yy, label=label, marker=MARKERS[j],
                markevery=max(1, len(yy) // 8),
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
