# Data

## Simulations

The quadratic-loss and logistic-loss experiments (Sections 6.1-6.5 of the
paper) use **synthetic data generated in-code** from the seeds and parameter
values described in the paper. No data files are needed; see
`src/sgdm_core.py` (`generate_quadratic_data`, `generate_logistic_data`).

## MNIST (Section 6.6)

The MNIST experiment reads the four standard IDX files from

```
data/MNIST/raw/
    train-images-idx3-ubyte(.gz)
    train-labels-idx1-ubyte(.gz)
    t10k-images-idx3-ubyte(.gz)
    t10k-labels-idx1-ubyte(.gz)
```

These files are publicly available and are downloaded automatically by

```bash
python src/mnist/mnist_experiment.py --download
```

if `--download` is passed (or you may place them here yourself). MNIST is
distributed under its own terms (Yann LeCun, Corinna Cortes, Christopher
Burges) and is not re-licensed by this repository.
