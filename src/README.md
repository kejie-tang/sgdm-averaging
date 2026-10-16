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
| `cifar/` | 16 | `python src/cifar/cifar_experiment.py` (from `data/cifar_logs/`; `--mode train --download` to re-train) |
| `logistic/` | EC.1, EC.2, EC.3 | `python src/logistic/logistic_experiment.py` |
| `mnist/` | EC.4 | `python src/mnist/mnist_experiment.py --download` |

`../scripts/run_all.sh` runs everything (`FAST=1` for a reduced-fidelity pass).

## The shared module `sgdm_core.py`

Every script imports the data generators, the iterates and the derived
quantities from `sgdm_core.py`, so the conventions cannot drift between
experiments:

| Function | Purpose |
|---|---|
| `draw_quadratic_problem` | the raw draws `V = rand(n, p, p)`, `b_i = rand(n, p, 1)`, `u = rand(p, 1)` from a caller-owned `rng` |
| `build_quadratic_data` | `A_i = rho V_i^T V_i + shift I_d`, `b_i`, `x*`, `L`, `mu`, the initial point |
| `generate_quadratic_data` | `draw_quadratic_problem` + `build_quadratic_data` in one call (kept for convenience; `rng=` passthrough supported) |
| `generate_logistic_data` | `a_i ~ N(0, I_d)`, labels from `sigmoid(theta0^T a_i)`; accepts a caller-owned `rng` so the features, labels and batches can be drawn from one stream |
| `make_batch_schedule` | the pre-sampled index sequence shared by SGD and SGDM; accepts the same optional `rng` |
| `batch_gradient_sequence` | caches the per-iteration mini-batch `(A, b)` means |
| `run_sgd` / `run_sgdm` | last-iterate trajectories (scalar or decaying `alpha`) |
| `averaged_errors` / `last_iterate_errors` | the two error curves of the figures; `averaged_errors` starts at `x_{n0+1}`, matching `xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j` |
| `adaptive_gamma`, `asymptotic_covariance`, `clt_projection` | derived quantities |

`batch_gradient_sequence` is a pure speed-up: a learning-rate / momentum sweep
replays the *same* mini-batches, so the batch gradients are formed once per
replication and handed to every configuration through the optional
`A_seq` / `b_seq` arguments of `run_sgd`, `run_sgdm` and `clt_projection`. The
numbers are bit-for-bit what recomputing them would give; only the run time
changes (the quadratic experiment, for instance, drops from ~27 s to ~4 s per
replication).

## Relationship to the original experiment code

The scripts re-implement the experiments from notebook prototypes and make them
runnable as plain Python. The numerical settings (dimension, batch size,
learning rates, momentum weights, number of replications, seeds) follow those
that produced the published figures. Momentum weights per panel are documented
in `../README.md` (Section 5).

The error curves are aggregated over replications by **squaring first and then
averaging** — `mean_seed ||x - x*||^2`, i.e. `np.mean(err ** 2, axis=0)` — which
is what the authors' plotting helpers compute.

Each replication is driven by **one unbroken random stream**: the problem data
and the mini-batch index sequence are drawn from it in the authors' order
(draw the data, then draw the schedule). This matters — seeding the schedule
separately yields a statistically equivalent but different realisation, and the
released panel then no longer coincides with the published one.

## Plotting geometry

The panels are drawn on the authors' canvas, which is what makes a regenerated
figure line up with the published one:

* `figsize = (4, 4)` (and `(8, 4)` / `(5, 4)` for the two-panel and sweep
  figures), with the axes box spanning `left=0.1, right=0.9, top=0.9,
  bottom=0.2` — a 3.2 x 2.8 inch box.
* `markevery = 0.3`. A **float** `markevery` is interpreted by matplotlib as a
  fraction of the axes-box diagonal and spaces the markers by arc length along
  the curve, so the marker positions follow from the axes-box size, not from
  the data alone. Reproducing the published marker positions requires copying
  the canvas above as well as `markevery`.
* `fig.savefig(path, bbox_inches="tight")` — the published panels are
  tight-cropped, which is measurable: on the paper the axes box occupies about
  0.79 of the image width and 0.82 of its height, whereas a raw 4 x 4 in canvas
  gives 0.80 / 0.70. Cropping with `tight` is also what keeps the longer
  y-labels (e.g. Figure 16's `Training Loss`) from being cut off.
* `yscale("log")` with explicit `set_ylim(...)` per panel (values listed in the
  script docstrings).

* theory (Fig 1)          <- `plot的副本3.ipynb`
* quadratic (Figs 2-4)    <- `sgd_and_sgdm_repeat2.ipynb` / `...-save*.ipynb`
* n0 sensitivity (Fig 5)  <- `sgd_and_sgdm_decaying.ipynb`
* diff L/mu (Fig 6)       <- `sgd_and_sgdm_decaying2.ipynb`
* decaying (Fig 7)        <- `sgd_and_sgdm_decaying2的副本.ipynb`
* CLT (Figs 8, 12)        <- `CLT.ipynb`, `clt.ipynb`
* sensitivity (Figs 9-11) <- `多线.py` (learning-rate grid), `sgd_and_sgdm_repeat-save*.ipynb`
* MNIST tanh (Figs 13-15) <- `nn_models.py`, `logistic_mnist_fixed_alpha_beta.py`,
  `average.ipynb`, `logs/`
* CIFAR (Fig 16)          <- `cifar05.ipynb` and
  `cifar/logistic_repeat2_average的副本.ipynb` (plotting; both load
  `cifar/logs 2/`), `cifar/logs 2/` (training logs, incl. `num.log`, which
  records the optimizer call). The four series behind the published panel ship
  as `data/cifar_logs/`, which is what `cifar_experiment.py` plots by default.
  The training script itself was lost; the architecture was recovered from the
  logs (see `cifar/vgg.py`) and re-implemented behind `--mode train`.
* logistic (Figs EC.1-3)  <- `logistic_repeat2_average.ipynb` (averaging,
  moments) and `...的副本.ipynb` (n₀ = 20 variant). The scripts reproduce the
  authors' own recursion, including the special first momentum `g_1 = grad(x_1)`
  without the `(1 - gamma)` factor. Re-running the notebook's cell verbatim and
  comparing with `results/logistic/logistic_raw.npz` agrees to 1.1e-15.
* MNIST linear (Fig EC.4) <- `logistic_mnist_fixed_alpha_beta.py`, `average.ipynb`,
  `logs-save-logistic/`

The MNIST / CIFAR scripts reproduce the loss curves with the authors' reporting
rule (`moving_ave2` in the notebooks): the value at iteration t is the running
mean of the per-iteration losses over iterations 1..t, averaged over seeds.

## The `Net` model (Figures 13-15)

`mnist_tanh/nn_models.py` defines the two-layer tanh network used for MNIST:
`784 -> 500 (tanh) -> 10`. Its module docstring spells out the correspondence
with the paper's `f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)`, including the two
points on which the paper's notation is easy to misread (the dimension of `X`
and the fact that `a_2i` is an output-layer *weight matrix*, not per-sample
data).
