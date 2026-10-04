# Code and Data for "Acceleration of Stochastic Gradient Descent with Momentum by Averaging: Finite-Sample Rates and Asymptotic Normality"

Kejie Tang, Weidong Liu, Yichen Zhang, Xi Chen

This repository contains all code and data needed to reproduce the numerical
experiments in the paper. It follows the directory conventions recommended by
the INFORMS/Operations Research code-and-data guidelines.

---

## 1. What is in this repository

```
.
├── README.md                     <- this file
├── LICENSE                       <- MIT
├── AUTHORS                       <- list of contributing authors
├── requirements.txt
├── src/                          <- source code
│   ├── sgdm_core.py              <- shared library (data generation, SGD/SGDM
│   │                                iterates, adaptive weight, CLT helpers)
│   ├── theory/
│   │   └── lambda_heatmap.py     <- Figure 1  (fig:lambda)
│   ├── quadratic/
│   │   └── quadratic_experiment.py  <- Figures 2, 3, 4 (fig:linear,
│   │                                    fig:average, fig:average2)
│   ├── sensitivity/
│   │   └── sensitivity_experiment.py <- Figures 5, 6, 7 (fig:dif-alpha,
│   │                                    fig:sen-alpha, fig:ave-alpha)
│   ├── clt/
│   │   └── clt_experiment.py     <- fig:clt, fig:clt-alpha
│   ├── logistic/
│   │   └── logistic_experiment.py <- Figures 7, 8, 9 (fig:logistic,
│   │                                   fig:logistic_ave, fig:logistic_ave2)
│   └── mnist/
│       └── mnist_experiment.py   <- Figure 10 (fig:mnist)
├── scripts/
│   └── run_all.sh                <- reproduces every figure in one command
├── data/
│   └── MNIST/                    <- MNIST IDX files (see Section 3)
├── docs/
│   └── STATEMENTS.md             <- data/code availability statements
└── results/                      <- output figures and raw arrays
```

## 2. Environment, hardware and software requirements

### Hardware

The reference results in the paper were produced on a **CPU-only** machine for
all simulation experiments; the MNIST experiment benefits from a GPU.

| Experiment | CPU | Memory | GPU |
|---|---|---|---|
| `theory` | any (1 core) | < 1 GB | not needed |
| `quadratic` | 1 core | ~ 2 GB | not needed |
| `sensitivity` | 1 core | ~ 2 GB | not needed |
| `clt` | 1 core | ~ 2 GB | not needed |
| `logistic` | 1 core | ~ 2 GB | not needed |
| `mnist` | 1 core | ~ 2 GB | optional (CUDA); runs on CPU too |

Reference machine used for the paper's experiments: a single-socket Linux
workstation (x86-64) with 16 GB RAM, no GPU for the simulations; the MNIST
experiment was run on a single NVIDIA GPU. All simulation scripts are
single-threaded and memory-light, so they run on any modern laptop.

### Software

* **Python >= 3.9** with **numpy** and **matplotlib** — required by all
  simulation scripts (`theory`, `quadratic`, `sensitivity`, `clt`, `logistic`)
  and by `mnist`.
* **PyTorch >= 1.10** — required only by `src/mnist/mnist_experiment.py`
  (CPU or CUDA build).
* No other dependencies. Tested with Python 3.11, numpy 2.5, matplotlib 3.11,
  PyTorch 2.11.

```bash
pip install -r requirements.txt
```

All scripts are self-contained and use a fixed random seed per replication, so
results are reproducible. Nothing is downloaded at run time except
(optionally) MNIST; all simulation data are generated on the fly from the
seeds described in the paper.

## 3. Data

**No data files are shipped with this repository;** everything is either
generated in-code or downloaded from its public source.

* **Simulations** (quadratic and logistic): synthetic, generated in-code from
  `N = 20,000`, `d = 10` as described in Sections 6.1 and 6.4 of the paper.
  No external data files are required.
* **MNIST** (Section 6.6): public dataset (LeCun et al.).
  Download page: <http://yann.lecun.com/exdb/mnist/>
  (mirror used by the scripts:
  <https://ossci-datasets.s3.amazonaws.com/mnist/>). The script fetches the
  four standard IDX files automatically with `--download`; if the machine is
  offline, download them from the page above and place them under
  `data/MNIST/raw/` by hand. MNIST is not re-licensed by this repository.

See `data/README.md` for details.

## 4. Reproducing each figure

Run every command from the repository root (the folder containing this file).
`bash scripts/run_all.sh` reproduces everything (set `FAST=1` for a
reduced-fidelity pass).

### Figure 1 — spectral radius heat map (`fig:lambda`)
```bash
python src/theory/lambda_heatmap.py --kappa 5
```
Produces `results/theory/fig_lambda.png`. The minimum is attained near
`alpha*sqrt(mu L) = 1`, `gamma = 0.146`, `lambda = 0.382`, matching the text.

### Figures 2-4 — quadratic loss (`fig:linear`, `fig:average`, `fig:average2`)
```bash
python src/quadratic/quadratic_experiment.py --num_seed 200
```
Sections 6.1. 200 replications, `N = 20,000`, `d = 10`, `B = 0.2N`,
`alpha = 0.001`, `K = 1500` iterations. Mean adaptive momentum weight ~ 0.96
(paper: 0.95-0.96). Panel (a) uses `gamma in {0.5, 0.7, 0.9}`, panel (b) uses
`gamma in {0.97, 0.98, 0.99}`. Outputs one PNG per paper panel:
`results/quadratic/fig_linear_{small,large}.png` (Fig. 2(a),(b)),
`fig_average_{small,large}_n0_200.png` (Fig. 3(a),(b)) and
`fig_average_{small,large}_n0_500.png` (Fig. 4(a),(b)), plus
`quadratic_raw.npz`.

### Figures 5-7 — sensitivity to the learning rate (`fig:dif-alpha`, `fig:sen-alpha`, `fig:ave-alpha`)
```bash
python src/sensitivity/sensitivity_experiment.py --num_seed 200
```
Section 6.2. `A_i = (V_i^T V_i + 1.03 I)/1.04` (condition number 26/1),
`alpha in {2^-1, 2^-3, 2^-5, 2^-7, 2^-9}`, `gamma in {0.8, 0.9}`.

### CLT — `fig:clt` and `fig:clt-alpha`
```bash
python src/clt/clt_experiment.py
```
Histogram of the projected statistic `Z` over 1000 replications
(`n0 = 1000`, `n = 2000`, `gamma = 0.9`), and empirical `P(|Z| < 1.96)` as a
function of the learning rate (1000 replications). To speed up a first look:
```bash
python src/clt/clt_experiment.py --num_seed 200 --num_seed_cov 200
```

### Figures 7-9 — logistic loss (`fig:logistic`, `fig:logistic_ave`, `fig:logistic_ave2`)
```bash
python src/logistic/logistic_experiment.py --num_seed 200
```
Section 6.4. `N = 20,000`, `d = 10`, `B = 0.2N`, `alpha = 0.5`,
`gamma in {0.3,0.5,0.7,0.8,0.9}` plus adaptive. Panel (a) uses
`gamma in {0.3,0.5,0.7}`, panel (b) uses `gamma in {0.8,0.9}` (both panels
keep SGD as the reference curve). Mean adaptive weight ~ 0.75. Outputs one PNG
per panel:
`fig_logistic_{small,large}.png` (Fig. 7(a),(b)),
`fig_logistic_ave_{small,large}.png` (Fig. 8, `n0=10`) and
`fig_logistic_ave2_{small,large}.png` (Fig. 9, `n0=40`).

### Figure 10 — MNIST (`fig:mnist`)
```bash
python src/mnist/mnist_experiment.py --download   # or place files by hand
```
Section 6.6. `B = 256`, `alpha = 1.0`, `gamma in {0.1,0.3,0.5,0.7,0.9,0.99}`,
seeds 1,2,3. The reported loss is a moving average of the past `floor(N/B)`
mini-batch losses. Panel (a) draws SGD, SGDM-0.1/0.3/0.5; panel (b) draws
SGDM-0.5/0.7/0.9/0.99 (no SGD curve, matching the paper). The IDX files are
read directly (no `torchvision` needed); pass `--data-root data/MNIST` to point
at a local copy, or `--device cpu` to force CPU.

## 5. Figure -> code map

| Paper element | Script | Key parameters |
|---|---|---|
| Fig. 1 `fig:lambda` | `src/theory/lambda_heatmap.py` | `kappa=5` |
| Fig. 2 `fig:linear` | `src/quadratic/quadratic_experiment.py` | `alpha=0.001`, `B=0.2N`, `gamma={0.5,0.7,0.9}` (a) / `{0.97,0.98,0.99}` (b) |
| Fig. 3 `fig:average` | `src/quadratic/quadratic_experiment.py` | `n0=200` |
| Fig. 4 `fig:average2` | `src/quadratic/quadratic_experiment.py` | `n0=500` |
| Fig. 5 `fig:dif-alpha` | `src/sensitivity/sensitivity_experiment.py` | `betas={0.8,0.9}` |
| Fig. 6 `fig:sen-alpha` | `src/sensitivity/sensitivity_experiment.py` | `T=500` |
| Fig. 7 `fig:ave-alpha` | `src/sensitivity/sensitivity_experiment.py` | `n0=500` |
| `fig:clt` | `src/clt/clt_experiment.py` | `n0=1000`, `n=2000`, `gamma=0.9` |
| `fig:clt-alpha` | `src/clt/clt_experiment.py` | `n=1000`, `n0=500` |
| Fig. 7 `fig:logistic` | `src/logistic/logistic_experiment.py` | `alpha=0.5`, `gamma={0.3,0.5,0.7}` (a) / `{0.8,0.9}` (b) |
| Fig. 8 `fig:logistic_ave` | `src/logistic/logistic_experiment.py` | `n0=10` |
| Fig. 9 `fig:logistic_ave2` | `src/logistic/logistic_experiment.py` | `n0=40` |
| Fig. 10 `fig:mnist` | `src/mnist/mnist_experiment.py` | `B=256`, `alpha=1.0`, `{0.1,0.3,0.5}` (a) / `{0.5,0.7,0.9,0.99}` (b) |

Notes on the panel split: figures `fig:linear`, `fig:average`, `fig:average2`,
`fig:logistic`, `fig:logistic_ave`, `fig:logistic_ave2` and `fig:mnist` are
each drawn as two panels in the paper, (a) *small* `gamma` and (b) *large*
`gamma`. The scripts emit one PNG per panel, named `*_small.png` and
`*_large.png`. The exact curve sets of each panel follow the published
figures:

| Figure (panel) | Curves drawn |
|---|---|
| `fig:linear`(a), `fig:average`(a), `fig:average2`(a) | SGD, SGDM-0.5, SGDM-0.7, SGDM-0.9, SGDM-adap |
| `fig:linear`(b), `fig:average`(b), `fig:average2`(b) | SGDM-adap, SGDM-0.97, SGDM-0.98, SGDM-0.99 |
| `fig:logistic`(a), `fig:logistic_ave`(a), `fig:logistic_ave2`(a) | SGD, SGDM-0.3, SGDM-0.5, SGDM-0.7, SGDM-adap |
| `fig:logistic`(b), `fig:logistic_ave`(b), `fig:logistic_ave2`(b) | SGD, SGDM-0.8, SGDM-0.9, SGDM-adap |
| `fig:mnist`(a) | SGD, SGDM-0.1, SGDM-0.3, SGDM-0.5 |
| `fig:mnist`(b) | SGDM-0.5, SGDM-0.7, SGDM-0.9, SGDM-0.99 (no SGD) |

The `2-*.png`, `8-*.png`, `simulation1.png`, `alpha0001-*.png`,
`clt-*.png` and `average_mnist.png` files that also live in the paper's
`figures/` directory are **commented out** in the manuscript source and are
therefore not reproduced here.

## 6. Notes on conventions

* **SGDM update.** `m_t = gamma*m_{t-1} + (1-gamma)*g_t`,
  `x_{t+1} = x_t - alpha*m_t`, and SGD is the `gamma = 0` case. In the MNIST
  script this is mapped to PyTorch's `SGD(lr = alpha*(1-gamma),
  momentum = gamma)`.
* **Adaptive momentum weight.** `gamma = ((1 - mu*alpha)/(1 + mu*alpha))^2`
  where `mu` is the smallest eigenvalue of the population Hessian.
* **Averaged iterate.** `xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j`.
* All simulations use the **same pre-sampled mini-batch sequence** for SGD and
  SGDM within each replication, to reduce plot noise.

## 7. Runtime, experiment logs, and reduced-fidelity runs

Rough single-core timings (200 replications, `N = 20,000`, `d = 10`):

| Experiment | Full run | Reduced run |
|---|---|---|
| `theory` | < 1 min | < 1 min |
| `quadratic` | ~ 20-40 min | `--num_seed 10 --n 3000 --K 400 --n0_list 50 100 --ave_len 200` ~ 1 min |
| `sensitivity` | ~ 30-60 min | `--num_seed 10 --n 3000 --K 300 --n0 100` ~ 2 min |
| `clt` | several hours (1000+1000 repl.) | `--num_seed 100 --num_seed_cov 100` ~ 20 min |
| `logistic` | ~ 1-2 h | `--num_seed 10 --n 5000` ~ 2 min |
| `mnist` | GPU: a few min; CPU: ~ 1 h | `--epochs 1 --seeds 1 --device cpu` ~ 3 min |

Every script exposes `--num_seed` (and `--n`, `--K`, `--epochs` where relevant)
so a quick reduced-fidelity pass is easy; the table above lists a concrete,
verified reduced-run command for each group. **To keep the reviewer's job
light, the experiments are split per figure group** (one script per group), so
each can be run and checked on its own within a reasonable time.

### Logs of the authors' runs

The `results/` directory stores the figures and the raw numeric arrays
(`*.npz`) produced by the authors' runs; the arrays are exactly what the
plotting code consumes, so the reported curves can be re-plotted without
re-running the experiments. Representative console logs from a full
reproduction are kept in `results/logs/` (created by
`scripts/run_all.sh > results/logs/full_run.log 2>&1`). If a full reproduction
is impractical, comparing the reviewer's `*.npz` against the stored arrays in
`results/` verifies the results directly.
