"""
Figure 16: SGD and SGDM on CIFAR-10 with a VGG-11 network.

Two modes:

``--mode plot`` (default)
    Draws the published panel from ``data/cifar_logs/``: the per-iteration
    training losses recorded by the authors' own runs, three seeds per momentum
    weight, 19,550 iterations each (50 epochs x 391 mini-batches of size 128).

``--mode train``
    Re-runs the experiment from scratch with the released trainer
    (``src/cifar/vgg.py``, VGG-11 without batch normalisation).  Needs PyTorch
    and the CIFAR-10 data (``--download``).

Setting
-------
CIFAR-10 (Krizhevsky et al. 2009): N = 50,000 training images of size 32x32
with three colour channels, 10 classes.  The loss is f_i(X) = H(VGG(X), b_i),
the multi-class cross-entropy.  Learning rate alpha = 0.5 and momentum weights
gamma in {0.0 (SGD), 0.7, 0.9, 0.99}.  The step is
``SGD(lr = alpha*(1 - gamma), momentum = gamma)``: ``torch.optim.SGD`` scales
its momentum buffer by ``lr``, so that pair realises the paper's recursion
``m = gamma*m + (1-gamma)*g, x -= alpha*m`` (README section 6.2 and
``docs/momentum_convention.md``, which also derives the correspondence).
Training with the unscaled ``lr = alpha`` makes the effective step
``alpha/(1-gamma)`` -- ten times too large at gamma = 0.9 -- and the SGDM runs
then sit at chance level.

There is no batch normalisation: the paper cites the original VGG-11 of
Simonyan & Zisserman (2014), and ``vgg.py`` uses that network's three-layer
classifier head (4096 -> 4096 -> 10 with dropout), sized for a 1x1 feature map
because CIFAR-10 images are 32x32 rather than 224x224.

Reporting convention
--------------------
Following the authors' plotting routine (their ``moving_ave2``), the curve
drawn at iteration t is the running mean of the per-iteration training losses
over iterations 0..t-1, taken over the three seeds.  A non-finite loss is
replaced by 1000 before averaging (so that a diverged seed contributes a large
but finite value rather than an NaN curve), and the final curve is clipped to
[-10, 100].

Data
----
``--mode plot`` reads ``data/cifar_logs/*.csv`` and needs nothing else.
``--mode train`` downloads CIFAR-10 (python batches, ~170 MB) on first use with
``--download`` and caches it under ``data/CIFAR10/``.

Outputs (results/cifar/):
    fig_cifar.png
    cifar_losses.npz

Run:  python src/cifar/cifar_experiment.py
      python src/cifar/cifar_experiment.py --mode train --download [--device cuda]
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
CLIP = (-10.0, 100.0)    # the authors' clip on the plotted curve
N_ITER = 19550           # 50 epochs x 391 mini-batches


# ---------------------------------------------------------------------------
# Plotting the archived logs (the default)
# ---------------------------------------------------------------------------

def moving_ave2(arr, k=390):
    """The authors' plotting helper, reproduced verbatim.

    Despite the argument ``k``, this is a *cumulative* mean: ``out[t]`` is the
    mean of ``arr[0..t-1]`` (and ``out[0] = arr[0]``).  ``k`` only pads the
    left of the working array and never changes the result.
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
# Training the reconstruction
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
    # torch.optim.SGD scales its momentum buffer by lr, so lr = alpha*(1-gamma)
    # realises the paper's recursion m = gamma*m + (1-gamma)*g, x -= alpha*m.
    # With the unscaled lr = alpha the effective step is alpha/(1-gamma), ten
    # times too large at gamma = 0.9, and the SGDM runs stay at chance level.
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
    # The authors' canvas: 4 x 4 in with the axes box at 0.1/0.9 by 0.2/0.9,
    # markevery=0.3, saved with bbox_inches="tight" (the notebooks' savefig
    # line, which the published panels were produced with).
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
                         "train: re-run the reconstruction from scratch")
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
