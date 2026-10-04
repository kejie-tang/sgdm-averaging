# `src/` — source code

One sub-directory per group of figures. All scripts import the shared module
`src/sgdm_core.py`; run them from the repository root.

| Directory | Figures | Command |
|---|---|---|
| `theory/` | 1 | `python src/theory/lambda_heatmap.py` |
| `quadratic/` | 2, 3, 4 | `python src/quadratic/quadratic_experiment.py` |
| `clt/` | 5, 9 | `python src/clt/clt_experiment.py` |
| `sensitivity/` | 6, 7, 8 | `python src/sensitivity/sensitivity_experiment.py` |
| `logistic/` | 10, 11, 12 | `python src/logistic/logistic_experiment.py` |
| `mnist/` | 13 | `python src/mnist/mnist_experiment.py` |

`../scripts/run_all.sh` runs everything (`FAST=1` for a reduced-fidelity pass).

## Relationship to the original experiment code

The scripts re-implement the experiments from notebook prototypes and make them
runnable as plain Python. The numerical settings (dimension, batch size,
learning rates, momentum weights, number of replications, seeds) follow those
that produced the published figures. Momentum weights per panel are documented
in `../README.md` (Section 5).

* quadratic loss  <- `sgd_and_sgdm_repeat2.ipynb` / `...-save*.ipynb`
* sensitivity     <- `sgd_and_sgdm_repeat-save*.ipynb`, `clt.ipynb`
* CLT             <- `CLT.ipynb`, `clt.ipynb`
* logistic        <- `logistic_repeat2_average.ipynb`
* MNIST           <- `logistic_mnist_fixed_alpha_beta.py`, `average.ipynb`
* lambda heat map <- `plot的副本3.ipynb`
