# Data

Everything needed to reproduce the paper is either generated in-code,
downloaded from its public source, or — for Figure 16 only — shipped as the
archived training logs described at the end of this file. No dataset copies are
distributed.

## Simulations (Sections 5.1, 5.2 and EC.7.1)

The quadratic-loss and logistic-loss experiments use **synthetic data generated
in-code** from the seeds and parameter values given in the paper. Nothing needs
to be downloaded; the generators live in `src/sgdm_core.py`
(`draw_quadratic_problem` / `build_quadratic_data`, `generate_logistic_data`).
Each replication draws its data *and* its mini-batch index sequence from one
unbroken random stream, in the order the authors' notebooks use.

## MNIST (Figures 13-15 and EC.4)

The MNIST experiments — the tanh network of Section 5.3 (Figures 13-15) and the
logistic regression of Section EC.7.2 (Figure EC.4) — use the public MNIST
dataset (Yann LeCun, Corinna Cortes, Christopher Burges).

* **Source / download page:** <http://yann.lecun.com/exdb/mnist/>
* **Mirror used by the scripts:** <https://ossci-datasets.s3.amazonaws.com/mnist/>

The scripts fetch the four standard IDX files automatically:

```bash
python src/mnist_tanh/mnist_tanh_experiment.py --download   # Figures 13-15
python src/mnist/mnist_experiment.py --download             # Figure EC.4
```

They are placed (and expected) under

```
data/MNIST/raw/
    train-images-idx3-ubyte(.gz)
    train-labels-idx1-ubyte(.gz)
    t10k-images-idx3-ubyte(.gz)
    t10k-labels-idx1-ubyte(.gz)
```

If your machine has no internet access, download those four files from the
page above and drop them into `data/MNIST/raw/` by hand; the scripts also read
the `.gz` versions directly.

## CIFAR-10 (Figure 16)

The CIFAR-10 experiment uses the public CIFAR-10 dataset
(Alex Krizhevsky, Geoffrey Hinton).

* **Source / download page:** <https://www.cs.toronto.edu/~kriz/cifar.html>
* **Archive used by the script:**
  <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>
* **Mirror tried first** (a byte-identical copy that is much faster outside
  North America): <https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz>

The dataset is only needed by `--mode train`, which downloads and extracts the
python batches (~170 MB) automatically:

```bash
python src/cifar/cifar_experiment.py --mode train --download
```

The default `--mode plot` draws the published panel from `data/cifar_logs/`
instead and needs neither PyTorch nor the dataset.

They are placed under

```
data/CIFAR10/cifar-10-batches-py/
    data_batch_1 ... data_batch_5
    test_batch
    batches.meta
```

The batches are the standard 32x32 RGB images; the script reads them with
`pickle` directly and does **not** require torchvision.

## Archived Figure-16 training logs (`data/cifar_logs/`)

These are the per-iteration training losses that produced the paper's
Figure 16. They are **data**, not figures: `src/cifar/cifar_experiment.py`
plots the published panel straight from them (default `--mode plot`), which is
what makes Figure 16 reproducible exactly rather than approximately.

```
data/cifar_logs/
    lr0.5_gamma0.0.csv     SGD,      seed 1/2/3
    lr0.5_gamma0.7.csv     SGDM-0.7, seed 1/2/3
    lr0.5_gamma0.9.csv     SGDM-0.9, seed 1/2/3
    lr0.5_gamma0.99.csv    SGDM-0.99, seed 1/2/3
```

Each file has a header `iteration,seed1,seed2,seed3` followed by 19,550 rows —
one per mini-batch, 50 epochs × 391 batches of size 128. A run that diverged
has `nan` from the point it blew up (one of the three SGD seeds does, at
iteration 18,447); this is what produces the spike at the right edge of the
published panel.

## Terms

The public datasets are distributed under their own terms and are **not**
re-licensed by this repository. The downloaded archives and extracted files are
git-ignored (see `.gitignore`) so that no dataset copies are committed.
