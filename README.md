# Code and Data for "Acceleration of Stochastic Gradient Descent with Momentum by Averaging: Finite-Sample Rates and Asymptotic Normality"

Code and data to reproduce every numerical result in the paper.

Repository: <https://github.com/kejie-tang/sgdm-averaging>

---

## 1. Layout

```
.
├── README.md
├── LICENSE            MIT
├── AUTHORS
├── requirements.txt
├── src/               source code (one script per figure group)
├── scripts/           run_all.sh — reproduce everything in one command
├── data/              where the public MNIST files come from (none committed)
├── docs/              submission statements
└── results/           output figures and raw arrays
```

## 2. Requirements

* **Python >= 3.9**, **numpy**, **matplotlib** — all simulation scripts.
* **PyTorch >= 1.10** — only `src/mnist/mnist_experiment.py`.

```bash
pip install -r requirements.txt
```

Everything runs on a modern laptop. The simulations are single-threaded and
need < 2 GB RAM and no GPU; the MNIST experiment uses a GPU if available.

## 3. Data

No data files are shipped with this repository.

* **Simulations** — synthetic, generated in-code; nothing to download.
* **MNIST** — public dataset, <http://yann.lecun.com/exdb/mnist/>. The script
  fetches it automatically with `--download`, or you can place the four IDX
  files under `data/MNIST/raw/` by hand. See `data/README.md`.

## 4. Reproducing the figures

Run from the repository root. `bash scripts/run_all.sh` does everything
(`FAST=1` for a quick reduced-fidelity pass).

**Figure 1** — spectral radius of Γ w.r.t. γ and α, where L/μ = 5/1.
```bash
python src/theory/lambda_heatmap.py --kappa 5
```

**Figures 2-4** — performance of SGD / SGDM and their averaged versions on the
quadratic loss. 200 replications, N = 20000, d = 10, B = 0.2N, α = 0.001,
K = 1500; mean adaptive momentum weight 0.96.
```bash
python src/quadratic/quadratic_experiment.py --num_seed 200
```
* Figure 2 — Performance of SGD and SGDM on the quadratic loss.
* Figure 3 — Performance of averaged SGD and averaged SGDM on the quadratic
  loss, n₀ = 200.
* Figure 4 — Same, n₀ = 500.

**Figures 5-8** — sensitivity of SGD / SGDM to the learning rate.
```bash
python src/sensitivity/sensitivity_experiment.py --num_seed 200
```
* Figure 5 — Frequency of Y for averaged SGD and averaged SGDM, γ = 0.9.
* Figure 6 — SGD and SGDM (γ = 0.8) under different learning rates; SGD fails
  to converge at α = 2⁻¹, 2⁻³.
* Figure 7 — SGD and SGDM (γ = 0.8, 0.9) under different learning rates, T = 500.
* Figure 8 — Averaged SGD and averaged SGDM (γ = 0.8) under different learning
  rates.

**Figure 9** — probability P(|Z| < 1.96) for SGD and SGDM (γ = 0.8, 0.9) under
different learning rates.
```bash
python src/clt/clt_experiment.py
```

**Figures 10-12** — performance of SGD / SGDM and their averaged versions on the
logistic loss. N = 20000, d = 10, B = 0.2N, α = 0.5; mean adaptive momentum
weight 0.75.
```bash
python src/logistic/logistic_experiment.py --num_seed 200
```
* Figure 10 — Performance of SGD and SGDM on the logistic loss.
* Figure 11 — Performance of averaged SGD and averaged SGDM on the logistic
  loss, n₀ = 10.
* Figure 12 — Same, n₀ = 40.

**Figure 13** — performance of SGD and SGDM on MNIST. B = 256, α = 1.0,
γ ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 0.99}, seeds 1-3.
```bash
python src/mnist/mnist_experiment.py --download
```

## 5. Figure to code

| Figure | Script | Key parameters |
|---|---|---|
| 1 | `src/theory/lambda_heatmap.py` | kappa = 5 |
| 2 | `src/quadratic/quadratic_experiment.py` | α = 0.001, γ = {0.5,0.7,0.9} (a) / {0.97,0.98,0.99} (b) |
| 3 | `src/quadratic/quadratic_experiment.py` | n₀ = 200 |
| 4 | `src/quadratic/quadratic_experiment.py` | n₀ = 500 |
| 5 | `src/clt/clt_experiment.py` | n₀ = 1000, n = 2000, γ = 0.9 |
| 6 | `src/sensitivity/sensitivity_experiment.py` | α ∈ {2⁻¹ … 2⁻⁹}, γ = 0.8 |
| 7 | `src/sensitivity/sensitivity_experiment.py` | T = 500, γ = {0.8, 0.9} |
| 8 | `src/sensitivity/sensitivity_experiment.py` | γ = 0.8, averaged |
| 9 | `src/clt/clt_experiment.py` | γ = {0.8, 0.9}, varying α |
| 10 | `src/logistic/logistic_experiment.py` | α = 0.5, γ = {0.3,0.5,0.7} (a) / {0.8,0.9} (b) |
| 11 | `src/logistic/logistic_experiment.py` | n₀ = 10 |
| 12 | `src/logistic/logistic_experiment.py` | n₀ = 40 |
| 13 | `src/mnist/mnist_experiment.py` | B = 256, α = 1.0 |

## 6. Conventions

* **Update.** `m_t = γ m_{t-1} + (1-γ) g_t`, `x_{t+1} = x_t - α m_t`; SGD is the
  `γ = 0` case. On MNIST this maps to PyTorch's
  `SGD(lr = α(1-γ), momentum = γ)`.
* **Adaptive weight.** `γ = ((1 - μα)/(1 + μα))²`, with μ the smallest
  eigenvalue of the population Hessian.
* **Averaged iterate.** `x̄_{n₀+t} = (1/t) Σ_{j=n₀+1}^{n₀+t} x_j`.
* All random number generators are seeded per replication, and SGD and SGDM
  share the same pre-sampled mini-batch sequence within a replication.
