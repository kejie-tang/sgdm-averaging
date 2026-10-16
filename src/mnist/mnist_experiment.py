"""
Figure EC.4: multinomial logistic regression on MNIST.

This experiment is reported in the paper's electronic companion (Section EC.7.2);
the tanh network of Section 5.3 is in ../mnist_tanh/.

Setting
-------
MNIST, N = 60,000 images of size 28x28 reshaped to 784-dim vectors.  The model
is a single linear layer (multinomial logistic regression) trained with the
cross-entropy loss.  Batch size B = 256, learning rate alpha = 1.0,
momentum weights gamma in {0.1, 0.3, 0.5, 0.7, 0.9, 0.99}, seeds 1, 2, 3.

Reporting convention: the curve drawn at iteration t is the **running mean of
the per-iteration mini-batch losses over iterations 1..t**, averaged over the
seeds.  (The electronic companion describes this as an average over the past
floor(N/B) batches; the running mean is what the published curves use, and the
two agree for t <= floor(N/B).)

The published panel stops at t = 1000 although the authors' run trains 20
epochs (20 x 234 = 4680 mini-batches), so the curves are cut at ``--max_iter``
(default 1000).  Cutting them matters for more than the axis range: the marker
positions follow the *arc length* of the drawn path, so the markers land
elsewhere if the tail is left in.

Panel layout:
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
    fig_mnist_small.png / fig_mnist_large.png  -> Figure EC.4(a),(b)
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

# Styling of the published panels (Figure EC.4).
SMALL = dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
             markers=['*', 'p', 'P', '*'], linestyles=['-', '-.', '--', '-'])
LARGE = dict(colors=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'],
             markers=['P', 'p', 'P', '*'], linestyles=['-', ':', '--', '-'])

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


def cumulative_mean(arr):
    """Running mean: out[t] = mean(arr[0], ..., arr[t])."""
    arr = np.asarray(arr, dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def moving_average(arr, k):
    """Average of the past k mini-batch losses (right-aligned window)."""
    arr = np.asarray(arr, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    out = np.empty(len(arr))
    for idx in range(len(arr)):
        lo = max(0, idx - k + 1)
        out[idx] = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
    return out


def run_one(seed, gamma, lr, num_epochs, batch_size, X, y, device,
            max_iter=None):
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
            if max_iter is not None and len(losses) >= max_iter:
                return np.array(losses)
    return np.array(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_iter", type=int, default=1000,
                    help="iterations that are plotted along t.  The authors' "
                         "run trains 20 epochs (4680 mini-batches) but the "
                         "published panel stops at t = 1000, so the curves are "
                         "cut there; this also fixes the marker positions, "
                         "which follow the arc length of the drawn path.")
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
                                args.batch_size, X, y, device,
                                max_iter=args.max_iter))
        L = min(len(r) for r in runs)
        # Running mean of the loss over iterations 1..t (see the module
        # docstring); applied per run before averaging over the seeds.
        ma_runs = [cumulative_mean(r[:L]) for r in runs]
        curves[gamma] = np.mean(ma_runs, axis=0)

    def plot(fname, betas, ylim, include_sgd, style):
        # The authors' canvas: 4 x 4 in, axes box at 0.1/0.9 by 0.2/0.9,
        # markevery=0.3, saved with bbox_inches="tight".
        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(gridspec.GridSpec(
            1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
        j = 0
        wanted = ([0.0] if include_sgd else []) + list(betas)
        for beta in wanted:
            if beta not in curves:
                continue
            label = "SGD" if beta == 0.0 else f"SGDM-{beta}"
            ax.plot(curves[beta], label=label, marker=style["markers"][j],
                    markevery=0.3, markersize=6,
                    linestyle=style["linestyles"][j],
                    color=style["colors"][j])
            j += 1
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Training Loss")
        ax.set_ylim(ylim)
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(args.out, fname), bbox_inches="tight")
        plt.close(fig)

    # Figure EC.4(a): SGD, SGDM-0.1, 0.3, 0.5
    plot("fig_mnist_small.png", [0.1, 0.3, 0.5], (0.2, 2.6), True, SMALL)
    # Figure EC.4(b): SGDM-0.5, 0.7, 0.9, 0.99  (no SGD in the paper's panel)
    plot("fig_mnist_large.png", [0.5, 0.7, 0.9, 0.99], (0.2, 2.6), False, LARGE)

    np.savez_compressed(os.path.join(args.out, "mnist_losses.npz"),
                        **{f"gamma_{g}": curves[g] for g in curves})
    print(f"[mnist] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
