# Data

**No data files are distributed with this repository.** Everything needed to
reproduce the paper is either generated in-code or downloaded from its public
source, as described below.

## Simulations (Sections 6.1-6.5)

The quadratic-loss and logistic-loss experiments use **synthetic data generated
in-code** from the seeds and parameter values given in the paper. Nothing needs
to be downloaded; the generators live in `src/sgdm_core.py`
(`generate_quadratic_data`, `generate_logistic_data`).

## MNIST (Section 6.6)

The MNIST experiment uses the public MNIST dataset
(Yann LeCun, Corinna Cortes, Christopher Burges).

* **Source / download page:** <http://yann.lecun.com/exdb/mnist/>
* **Mirror used by the scripts:** <https://ossci-datasets.s3.amazonaws.com/mnist/>

The script fetches the four standard IDX files automatically:

```bash
python src/mnist/mnist_experiment.py --download
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
page above and drop them into `data/MNIST/raw/` by hand; the script also reads
the `.gz` versions directly.

MNIST is distributed under its own terms and is **not** re-licensed by this
repository. The `.gz` archives are git-ignored (see `.gitignore`) so that no
dataset copies are committed.
