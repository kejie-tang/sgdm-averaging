# Data

**No data files are distributed with this repository.** Everything needed to
reproduce the paper is either generated in-code or downloaded from its public
source, as described below.

## Simulations (Sections 6.1-6.5)

The quadratic-loss and logistic-loss experiments use **synthetic data generated
in-code** from the seeds and parameter values given in the paper. Nothing needs
to be downloaded; the generators live in `src/sgdm_core.py`
(`generate_quadratic_data`, `generate_logistic_data`).

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
* **Mirror used by the script:**
  <https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz>

The script downloads and extracts the python batches (~170 MB) automatically:

```bash
python src/cifar/cifar_experiment.py --download
```

They are placed under

```
data/CIFAR10/cifar-10-batches-py/
    data_batch_1 ... data_batch_5
    test_batch
    batches.meta
```

The batches are the standard 32x32 RGB images; the script reads them with
`pickle` directly and does **not** require torchvision.

## Terms

Both datasets are distributed under their own terms and are **not** re-licensed
by this repository. The downloaded archives and extracted files are git-ignored
(see `.gitignore`) so that no dataset copies are committed.
