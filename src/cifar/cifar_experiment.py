"""
Figure 16: SGD and SGDM on CIFAR-10 with VGG-11.

Setting: N=50,000, B=128, alpha=0.5, gamma in {0, 0.7, 0.9, 0.99},
50 epochs (19,550 iterations), three seeds. The model has no batch
normalization and a 512 -> 4096 -> 4096 -> 10 classifier with dropout.
Training uses SGD(lr=alpha*(1-gamma), momentum=gamma) and cross-entropy loss.

--mode plot (default) reads the archived Figure 16 training losses from
data/cifar_logs/ and requires NumPy and Matplotlib. --mode train uses a
reconstructed trainer and requires PyTorch and the CIFAR-10 data.

For t >= 1, the plotted value is the cumulative mean over losses[0:t],
averaged across seeds; at t=0 it is the first loss. Non-finite losses are
replaced by 1000 before averaging, and the final curve is clipped to [-10,100].

Outputs (results/cifar/): fig_cifar.png, cifar_losses.npz.

Run: python src/cifar/cifar_experiment.py
     python src/cifar/cifar_experiment.py --mode train --download [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tarfile

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from sgdm_core import validate_experiment_args  # noqa: E402
from data_io import download_file  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

# Figure 16 styling (SGD, SGDM-0.7, SGDM-0.9, SGDM-0.99).
MARKERS = ["P", "o", "v", "s"]
LINESTYLES = ["-.", "-", "--", ":"]
COLORS = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']
GAMMAS = [0.0, 0.7, 0.9, 0.99]

DIVERGED = 1000.0        # non-finite loss -> this value before averaging
CLIP = (-10.0, 100.0)    # limits for the plotted cumulative loss
N_ITER = 19550           # 50 epochs x 391 mini-batches


# ---------------------------------------------------------------------------
# Plotting the archived logs (the default)
# ---------------------------------------------------------------------------

def moving_ave2(arr, k=390):
    """Cumulative mean: out[t] = mean(arr[:t]) for t >= 1.

    out[0] = arr[0]; k pads the workspace and does not affect the result.
    """
    arr = np.asarray(arr, dtype=float)
    n_iter = len(arr)          # the paper's ``n``: number of iterations
    ar = np.zeros([n_iter + k])
    ar[0:k] = arr[0]
    ar[k:(n_iter + k)] = arr
    ma = arr.copy()
    for idx in range(1, n_iter):
        ma[idx] = np.mean(ar[k:(idx + k)])
    return ma


def curve_from_logs(csv_path):
    """Load one archived log file and return its Figure-16 curve."""
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)   # iteration + 3 seeds
    seeds = table[:, 1:]                                      # (n_iter, 3)
    seeds = np.nan_to_num(seeds, nan=DIVERGED, posinf=DIVERGED,
                          neginf=DIVERGED)
    mean_loss = seeds.mean(axis=1)                            # average over seeds
    return np.clip(np.nan_to_num(moving_ave2(mean_loss), nan=CLIP[1]),
                   CLIP[0], CLIP[1])


def load_curves(log_dir):
    curves = {}
    for gamma in GAMMAS:
        path = os.path.join(log_dir, f"lr0.5_gamma{gamma}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. The Figure-16 logs ship with the release "
                f"under data/cifar_logs/.")
        curves[gamma] = curve_from_logs(path)
        if len(curves[gamma]) != N_ITER:
            print(f"[cifar] warning: {path} has {len(curves[gamma])} rows, "
                  f"expected {N_ITER}")
    return curves


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
                    download_file(url, tgz)
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

    from vgg import VGG11

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    N = X.shape[0]
    num_batches = int(round(N / batch_size))
    mean = torch.from_numpy(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(CIFAR_STD).view(1, 3, 1, 1)

    net = VGG11(10).to(device)
    criterion = nn.CrossEntropyLoss()
    # lr=alpha*(1-gamma) implements m=gamma*m+(1-gamma)*g, x-=alpha*m.
    optimizer = optim.SGD(net.parameters(), lr=lr * (1.0 - gamma),
                          momentum=gamma)

    losses = []
    t = 0
    for _ in range(epochs):
        for _ in range(num_batches):
            choice = rng.integers(0, N, size=batch_size)
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


def train_curves(args, device, gammas):
    X, y = load_cifar10(args.data_root, download=args.download)
    print(f"[cifar] loaded {X.shape[0]} training images")

    curves = {}
    for gamma in gammas:
        runs = []
        for seed in args.seeds:
            print(f"[cifar] gamma={gamma} seed={seed}", flush=True)
            r = run_one(seed, gamma, args.lr, args.epochs, args.batch_size,
                        X, y, device, max_iter=args.max_iter)
            runs.append(np.where(np.isfinite(r), r, DIVERGED))
        n_iter = min(len(r) for r in runs)
        mean_loss = np.mean([r[:n_iter] for r in runs], axis=0)
        curves[gamma] = np.clip(np.nan_to_num(
            moving_ave2(mean_loss), nan=CLIP[1]), CLIP[0], CLIP[1])
    return curves


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(curves, out_png):
    # Use a 4 x 4 inch canvas with the axes at 0.1/0.9 by 0.2/0.9.
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.1, right=0.9, top=0.9, bottom=0.2, figure=fig)[0])
    for j, gamma in enumerate(curves):
        label = "SGD" if gamma == 0.0 else f"SGDM-{gamma}"
        yy = curves[gamma]
        ax.plot(np.arange(len(yy)), yy, label=label, marker=MARKERS[j % 4],
                markevery=0.3,
                linestyle=LINESTYLES[j % 4], color=COLORS[j % 4])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Training Loss")
    ax.set_ylim((0.2, 2.6))
    ax.set_xlim((0, 20000))
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plot", "train"], default="plot",
                    help="plot (default): draw from the archived logs; "
                         "train: train the network from scratch")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="directory with the archived Figure-16 logs")
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.7, 0.9, 0.99],
                    help="train mode only (SGD/gamma=0 is always included)")
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
    validate_experiment_args(ap, args)

    if args.log_dir is None:
        args.log_dir = os.path.join(_HERE, "..", "..", "data", "cifar_logs")
    if args.data_root is None:
        args.data_root = os.path.join(_HERE, "..", "..", "data", "CIFAR10")
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "plot":
        print(f"[cifar] plotting from {os.path.abspath(args.log_dir)}")
        curves = load_curves(args.log_dir)
    else:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[cifar] device = {device}")
        # SGD (gamma = 0) is always included; the requested weights follow it.
        gammas = [0.0] + [g for g in args.gamma if g != 0.0]
        curves = train_curves(args, device, gammas)

    draw(curves, os.path.join(args.out, "fig_cifar.png"))
    np.savez_compressed(os.path.join(args.out, "cifar_losses.npz"),
                        **{f"gamma_{g}": curves[g] for g in curves})
    print(f"[cifar] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
