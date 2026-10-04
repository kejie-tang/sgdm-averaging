"""
Figure 10 (fig:mnist): multinomial logistic regression on MNIST.

Reproduces Section 6.6 of the paper.

Setting
-------
MNIST, N = 60,000 images of size 28x28 reshaped to 784-dim vectors.  The model
is a single linear layer (multinomial logistic regression) trained with the
cross-entropy loss.  Batch size B = 256, learning rate alpha = 1.0,
momentum weights gamma in {0.1, 0.3, 0.5, 0.7, 0.9, 0.99}, seeds 1, 2, 3.
The reported training loss at iteration t is a moving average of the past
floor(N/B) mini-batch losses (as stated in the paper).

Panel layout of Figure 10 (fig:mnist):
    (a) "Small gamma": SGD, SGDM-0.1, SGDM-0.3, SGDM-0.5
    (b) "Large gamma": SGDM-0.5, SGDM-0.7, SGDM-0.9, SGDM-0.99
        (the paper's panel (b) contains no SGD curve)

Data
----
MNIST is read directly from the standard IDX files; the script does NOT need
torchvision.  Put the four raw files under ``data/MNIST/raw/``:
    train-images-idx3-ubyte  train-labels-idx1-ubyte
    t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte
(also accept the .gz versions).  If the directory does not exist and
``--download`` is passed, they are fetched from the public mirrors.

Requires: torch only (see requirements.txt).

Outputs (results/mnist/):
    fig_mnist_small.png / fig_mnist_large.png  -> Figure 10(a),(b)
    mnist_losses.npz

Run:  python src/mnist/mnist_experiment.py [--device cuda] [--download]
"""

from __future__ import annotations

import argparse
import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

MARKERS = ["o", "v", "s", "p", "P", "*", "+", "x"]
LINESTYLES = ["-.", "--", ":", "-"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2']

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _read_idx(path):
    """Read an IDX file (gzip optional). Returns a numpy array (uint8)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    magic, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8)
    return data.reshape(dims, order="C")


def load_mnist(raw_dir, download=False):
    """Load train images/labels from IDX files under ``raw_dir``."""
    names = {
        "train_img": "train-images-idx3-ubyte",
        "train_lbl": "train-labels-idx1-ubyte",
    }
    os.makedirs(raw_dir, exist_ok=True)

    def resolve(base):
        for cand in (base, base + ".gz"):
            p = os.path.join(raw_dir, cand)
            if os.path.exists(p):
                return p
        if download:
            for mirror in MIRRORS:
                for suffix in (base, base + ".gz"):
                    try:
                        url = mirror + suffix
                        dst = os.path.join(raw_dir, suffix)
                        print(f"[mnist] downloading {url}")
                        urllib.request.urlretrieve(url, dst)
                        return dst
                    except Exception as e:  # noqa: BLE001
                        print(f"[mnist]   failed: {e}")
        raise FileNotFoundError(
            f"{base} not found in {raw_dir}. Place the MNIST IDX files there "
            f"or pass --download.")

    X = _read_idx(resolve(names["train_img"])).reshape(-1, 784).astype(np.uint8)
    y = _read_idx(resolve(names["train_lbl"])).astype(np.int64)
    return X, y


def moving_average(arr, k):
    """Average of the past k mini-batch losses (the paper's reporting rule)."""
    n = len(arr)
    ma = np.zeros(n, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    for idx in range(n):
        lo = max(0, idx - k + 1)
        ma[idx] = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    return ma


def run_one(seed, gamma, lr, num_epochs, batch_size, X, y, device):
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

    net = nn.Sequential(nn.Linear(784, 10)).to(device)
    criterion = nn.CrossEntropyLoss()
    # The paper reports momentum weight beta and learning rate alpha; the
    # equivalent PyTorch SGD uses lr = alpha * (1 - beta), momentum = beta.
    optimizer = optim.SGD(net.parameters(), lr=lr * (1 - gamma),
                          momentum=gamma)

    losses = []
    for epoch in range(1, num_epochs + 1):
        net.train()
        for _ in range(num_batches):
            choice = np.random.choice(n, batch_size, replace=False)
            inputs = Xt[choice].to(device)
            targets = yt[choice].to(device)
            optimizer.zero_grad()
            loss = criterion(net(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "mnist"))
    args = ap.parse_args()

    if args.data_root is None:
        args.data_root = os.path.join(os.path.dirname(__file__),
                                      "..", "..", "data", "MNIST")
    raw_dir = os.path.join(args.data_root, "raw")
    os.makedirs(args.out, exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mnist] device = {device}")

    X, y = load_mnist(raw_dir, download=args.download)
    print(f"[mnist] loaded {X.shape[0]} training images")

    gamma_all = [0.0] + list(args.gamma)
    curves = {}
    for gamma in gamma_all:
        runs = []
        for seed in args.seeds:
            print(f"[mnist] gamma={gamma} seed={seed}", flush=True)
            runs.append(run_one(seed, gamma, args.lr, args.epochs,
                                args.batch_size, X, y, device))
        L = min(len(r) for r in runs)
        # The paper reports a moving average over the past floor(N/B)
        # mini-batch losses; apply it to each run before averaging the seeds.
        k = args.batch_size and (X.shape[0] // args.batch_size)
        ma_runs = [moving_average(r[:L], k) for r in runs]
        curves[gamma] = np.mean(ma_runs, axis=0)

    def plot(fname, betas, ylim, include_sgd):
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(gridspec.GridSpec(
            1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15, figure=fig)[0])
        j = 0
        wanted = ([0.0] if include_sgd else []) + list(betas)
        for beta in wanted:
            if beta not in curves:
                continue
            label = "SGD" if beta == 0.0 else f"SGDM-{beta}"
            ax.plot(curves[beta], label=label, marker=MARKERS[j],
                    markevery=max(1, len(curves[beta]) // 8),
                    linestyle=LINESTYLES[j % 4], color=COLORS[j])
            j += 1
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Training Loss")
        ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # Figure 10(a): SGD, SGDM-0.1, 0.3, 0.5
    plot("fig_mnist_small.png", [0.1, 0.3, 0.5], (0.2, 2.6), include_sgd=True)
    # Figure 10(b): SGDM-0.5, 0.7, 0.9, 0.99  (no SGD in the paper's panel)
    plot("fig_mnist_large.png", [0.5, 0.7, 0.9, 0.99], (0.2, 2.6),
         include_sgd=False)

    np.savez_compressed(os.path.join(args.out, "mnist_losses.npz"),
                        **{f"gamma_{g}": curves[g] for g in curves})
    print(f"[mnist] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
