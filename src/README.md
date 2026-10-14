# `src/` — source code

One sub-directory per group of figures. All scripts import the shared module
`src/sgdm_core.py`; run them from the repository root.

| Directory | Figures | Command |
|---|---|---|
| `theory/` | 1 | `python src/theory/lambda_heatmap.py` |
| `quadratic/` | 2, 3, 4 | `python src/quadratic/quadratic_experiment.py` |
| `n0_sensitivity/` | 5 | `python src/n0_sensitivity/n0_sensitivity_experiment.py` |
| `diff_lmu/` | 6 | `python src/diff_lmu/diff_lmu_experiment.py` |
| `decaying/` | 7 | `python src/decaying/decaying_experiment.py` |
| `clt/` | 8, 12 | `python src/clt/clt_experiment.py` |
| `sensitivity/` | 9, 10, 11 | `python src/sensitivity/sensitivity_experiment.py` |
| `mnist_tanh/` | 13, 14, 15 | `python src/mnist_tanh/mnist_tanh_experiment.py --download` |
| `cifar/` | 16 | `python src/cifar/cifar_experiment.py --download` |
| `logistic/` | EC.1, EC.2, EC.3 | `python src/logistic/logistic_experiment.py` |
| `mnist/` | EC.4 | `python src/mnist/mnist_experiment.py --download` |

`../scripts/run_all.sh` runs everything (`FAST=1` for a reduced-fidelity pass).

## Relationship to the original experiment code

The scripts re-implement the experiments from notebook prototypes and make them
runnable as plain Python. The numerical settings (dimension, batch size,
learning rates, momentum weights, number of replications, seeds) follow those
that produced the published figures. Momentum weights per panel are documented
in `../README.md` (Section 5).

* theory (Fig 1)          <- `plot的副本3.ipynb`
* quadratic (Figs 2-4)    <- `sgd_and_sgdm_repeat2.ipynb` / `...-save*.ipynb`
* n0 sensitivity (Fig 5)  <- `sgd_and_sgdm_decaying.ipynb`
* diff L/mu (Fig 6)       <- `sgd_and_sgdm_decaying2.ipynb`
* decaying (Fig 7)        <- `sgd_and_sgdm_decaying2的副本.ipynb`
* CLT (Figs 8, 12)        <- `CLT.ipynb`, `clt.ipynb`
* sensitivity (Figs 9-11) <- `多线.py` (learning-rate grid), `sgd_and_sgdm_repeat-save*.ipynb`
* MNIST tanh (Figs 13-15) <- `nn_models.py`, `logistic_mnist_fixed_alpha_beta.py`, `average.ipynb`
* CIFAR (Fig 16)          <- `cifar05.ipynb` (original training script lost)
* logistic (Figs EC.1-3)  <- `logistic_repeat2_average.ipynb`
* MNIST linear (Fig EC.4) <- `logistic_mnist_fixed_alpha_beta.py`, `average.ipynb`

## The `Net` model (Figures 13-15)

`mnist_tanh/nn_models.py` defines the two-layer tanh network used for MNIST:
`784 -> 500 (tanh) -> 10`. Its module docstring spells out the correspondence
with the paper's `f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)`, including the two
points on which the paper's notation is easy to misread (the dimension of `X`
and the fact that `a_2i` is an output-layer *weight matrix*, not per-sample
data).
