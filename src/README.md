# `src/` — source code

Each sub-directory reproduces one group of figures. All scripts import the
shared module `src/sgdm_core.py` (run them from the repository root so the
relative import path resolves).

| Directory | Paper element | Command |
|---|---|---|
| `theory/` | Fig. 1 `fig:lambda` | `python src/theory/lambda_heatmap.py` |
| `quadratic/` | Figs. 2-4 | `python src/quadratic/quadratic_experiment.py` |
| `sensitivity/` | Figs. 5-7 | `python src/sensitivity/sensitivity_experiment.py` |
| `clt/` | `fig:clt`, `fig:clt-alpha` | `python src/clt/clt_experiment.py` |
| `logistic/` | Figs. 7-9 | `python src/logistic/logistic_experiment.py` |
| `mnist/` | Fig. 10 | `python src/mnist/mnist_experiment.py` |

`../scripts/run_all.sh` runs everything (set `FAST=1` for a reduced-fidelity
pass).

## Relationship to the original experiment code

The scripts here re-implement the experiments from the authors' original
notebooks and clean them into runnable form. The numerical settings
(dimension, batch size, learning rates, momentum weights, number of
replications, seeds) are taken verbatim from the notebooks that produced the
paper's figures:

* quadratic loss  <- `sgd_and_sgdm_repeat2.ipynb` / `...-save*.ipynb`
* sensitivity     <- `sgd_and_sgdm_repeat-save*.ipynb`, `clt.ipynb`
* CLT             <- `CLT.ipynb`, `clt.ipynb`
* logistic        <- `logistic_repeat2_average.ipynb`
* MNIST           <- `logistic_mnist_fixed_alpha_beta.py`, `average.ipynb`
* lambda heat map <- `plot的副本3.ipynb`
