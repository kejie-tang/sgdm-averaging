"""
Figure 16: SGD and SGDM on CIFAR-10 with a VGG-11 network.

The published panel is reproduced **exactly** in the default ``--mode plot``.
The four curves are drawn from the per-iteration training-loss series that
generated the paper's Figure 16; those series were recorded by the authors'
original training script and are archived under ``data/cifar_logs/``.  Each
file holds, for one momentum weight, the per-iteration training loss of the
three replications: 19,550 iterations = 50 epochs x 391 mini-batches of size
128.  Plotting straight from the archive is what the ``fig_cifar.png`` shipped
under ``results/cifar/`` is made with.

``--mode train`` re-runs the experiment from scratch with the released
reconstruction of the original trainer (``src/cifar/vgg.py``, VGG-11 without
batch normalisation).  It needs PyTorch and the CIFAR-10 data
(``--download``); see the Note below for how far it can be pushed.

Setting
-------
CIFAR-10 (Krizhevsky et al. 2009): N = 50,000 training images of size 32x32
with three colour channels, 10 classes.  The loss is f_i(X) = H(VGG(X), b_i),
the multi-class cross-entropy.  Learning rate alpha = 0.5 and momentum weights
gamma in {0.0 (SGD), 0.7, 0.9, 0.99}.  The realised PyTorch step is
``SGD(lr = 0.5, momentum = gamma)``: the authors' log prints the same
``lr = 0.5`` for every momentum weight, so the learning rate is *not* rescaled
by (1 - gamma) as it is in the convex experiments.

There is no batch normalisation: the paper cites the original VGG-11, and this
is what the stored logs imply (see ``vgg.py`` and README.md, section 7).

Reporting convention
--------------------
Following the authors' plotting routine (their ``moving_ave2``), the curve
drawn at iteration t is the running mean of the per-iteration training losses
over iterations 0..t-1, taken over the three seeds.  A non-finite loss is
replaced by 1000 before averaging (so that a diverged seed contributes a large
but finite value rather than an NaN curve), and the final curve is clipped to
[-10, 100].

Note on ``--mode train``
------------------------
The authors' original CIFAR training script was not preserved with the sources,
so ``vgg.py`` and the training loop below are a reconstruction from the paper's
description and the stored logs (architecture, learning rate, epochs, batches
per epoch, momentum weights and the reporting rule).

With the correct (no-BN) architecture the three SGDM curves of a fresh run
track the archived logs closely (a mean absolute deviation of roughly 0.1-0.2
in the running mean in our runs).  The SGD curve is a different matter: at
alpha = 0.5 the no-BN network is bistable.  In the archived logs two of the three seeds sit at chance
level log(10) = 2.3036 for all 50 epochs and one escapes around epoch 6 and
descends to a per-epoch mean of ~0.45, so their mean falls from 2.30 to ~1.5;
that single seed then blows up at the start of epoch 48 (iteration 18,447),
which is what produces the spike at the right edge of the published panel.
Re-running the released reconstruction with the same three seeds lands on the
other mixture -- all three escape smoothly and none blows up -- so its
seed-mean curve falls from 2.30 to ~0.79 instead, without the spike.  Matching
the published SGD curve would require the original seeds, weight initialisation
and data-augmentation RNG, none of which are recorded; that is why the released
``fig_cifar.png`` is drawn from the archived logs.

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
import glob
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
    n = len(arr)
    ar = np.zeros([n + k])
    ar[0:k] = arr[0]
    ar[k:(n + k)] = arr
    ma = arr.copy()
    for idx in range(1, n):
        ma[idx] = np.mean(ar[k:(idx + k)])
    return ma


def curve_from_logs(csv_path):
    """Load one archived log file and return its Figure-16 curve."""
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)   # iteration + 3 seeds
    seeds = table[:, 1:]                                      # (n_iter, 3)
    seeds = np.nan_to_num(seeds, nan=DIVERGED, posinf=DIVERGED)
    mean_loss = seeds.mean(axis=1)                            # average over seeds
    return np.clip(np.nan_to_num(moving_ave2(mean_loss), nan=CLIP[1]),
                   CLIP[0], CLIP[1])


def load_curves(log_dir):
    curves = {}
    for gamma in GAMMAS:
        path = os.path.join(log_dir, f"lr0.5_beta{gamma}.csv")
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

    n = X.shape[0]
    num_batches = int(round(n / batch_size))
    mean = torch.from_numpy(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(CIFAR_STD).view(1, 3, 1, 1)

    net = VGG11(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=gamma)

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


def train_curves(args, device):
    X, y = load_cifar10(args.data_root, download=args.download)
    print(f"[cifar] loaded {X.shape[0]} training images")

    curves = {}
    for gamma in GAMMAS:
        runs = []
        for seed in args.seeds:
            print(f"[cifar] gamma={gamma} seed={seed}", flush=True)
            r = run_one(seed, gamma, args.lr, args.epochs, args.batch_size,
                        X, y, device, max_iter=args.max_iter)
            runs.append(np.where(np.isfinite(r), r, DIVERGED))
        L = min(len(r) for r in runs)
        mean_loss = np.mean([r[:L] for r in runs], axis=0)
        curves[gamma] = np.clip(np.nan_to_num(
            moving_ave2(mean_loss), nan=CLIP[1]), CLIP[0], CLIP[1])
    return curves


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(curves, out_png):
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.17, right=0.95, top=0.95, bottom=0.15, figure=fig)[0])
    for j, gamma in enumerate(GAMMAS):
        label = "SGD" if gamma == 0.0 else f"SGDM-{gamma}"
        yy = curves[gamma]
        ax.plot(np.arange(len(yy)), yy, label=label, marker=MARKERS[j],
                markevery=max(1, len(yy) // 6),
                linestyle=LINESTYLES[j % 4], color=COLORS[j])
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
        curves = train_curves(args, device)

    draw(curves, os.path.join(args.out, "fig_cifar.png"))
    np.savez_compressed(os.path.join(args.out, "cifar_losses.npz"),
                        **{f"gamma_{g}": curves[g] for g in GAMMAS})
    print(f"[cifar] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
