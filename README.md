# Code and Data for "Acceleration of Stochastic Gradient Descent with Momentum by Averaging: Finite-Sample Rates and Asymptotic Normality"

Code and data needed to reproduce the numerical experiments in the paper.

Repository (anonymous mirror for review): <https://anonymous.4open.science/r/sgdm-averaging-2F7D>

---

## 1. Layout

```
.
├── README.md
├── LICENSE            MIT
├── AUTHORS
├── requirements.txt
├── src/               source code (one script per group of figures)
├── scripts/           run_all.sh — reproduce everything in one command
├── data/              public datasets + the archived Figure-16 logs
├── docs/              submission statements
└── results/           output figures and raw arrays
```

## 2. Requirements

* **Python >= 3.9**, **numpy**, **matplotlib** — every script, including the
  default (exact) Figure 16.
* **PyTorch >= 1.10** — the MNIST experiments (Figures 13-15 and EC.4) and the
  optional `src/cifar/cifar_experiment.py --mode train`.

```bash
pip install -r requirements.txt
```

The quadratic-loss simulations (Figures 2-7) are single-threaded and need
< 2 GB RAM and no GPU; the MNIST and CIFAR-10 experiments are much faster on a
GPU but also run on CPU.

## 3. Data

* **Simulations** — synthetic, generated in-code; nothing to download.
* **MNIST** — public dataset, <http://yann.lecun.com/exdb/mnist/>. Fetched
  automatically with `--download`, or place the four IDX files under
  `data/MNIST/raw/`.
* **CIFAR-10** — public dataset, <https://www.cs.toronto.edu/~kriz/cifar.html>.
  Fetched automatically with `--mode train --download` (~170 MB), or place the
  extracted python batches under `data/CIFAR10/`. The dataset is **not** needed
  for the published Figure 16.
* **Figure 16 training logs** — `data/cifar_logs/` ships the 12 per-iteration
  loss series (4 momentum weights × 3 seeds) that the published CIFAR-10 panel
  is drawn from. These are the only result data files distributed here.

See `data/README.md`.

## 4. Reproducing the figures

Run from the repository root. `bash scripts/run_all.sh` does everything
(`FAST=1` for a quick reduced-fidelity pass).

**Figure 1** — colour maps of the spectral radius λ of the SGDM recursion.
Panel (a) is the spectral radius itself; panel (b) highlights, in yellow, the
region where SGDM is faster than optimally specified SGD, i.e. where
λ_SGDM(α, γ) < (L − μ)/(L + μ).
```bash
python src/theory/lambda_heatmap.py --kappa 5
```

**Figures 2-4** — SGD / SGDM and their averaged versions on the quadratic loss.
N = 20,000, d = 10, B = 0.2N, α = 0.001, n = 1500 iterations, 200 replications; average
adaptive momentum weight 0.96.
```bash
python src/quadratic/quadratic_experiment.py --num_seed 200
```
* Figure 2 — Performance of SGD and SGDM on the quadratic loss, with small
  momentum weights γ = 0.5, 0.7, 0.9 in panel (a) and large ones
  γ = 0.97, 0.98, 0.99 in panel (b).
* Figure 3 — Performance of x̄_{n₀+t} for Averaged SGD and Averaged SGDM, n₀ = 500.
* Figure 4 — Performance of xₜ for SGD and SGDM, and their averaged versions, γ = 0.96.

**Figure 5** — SGD and SGDM with γ = 0.96 and their averaged versions with
different n₀.
```bash
python src/n0_sensitivity/n0_sensitivity_experiment.py --num_seed 200
```

**Figure 6** — SGD and SGDM with the adaptive γ under different condition
numbers L/μ = 3.5, 9.9, 14.7 (ρ ∈ {1, 5, 10}), n₀ = 250. The script reports the
condition numbers it computes from the realised population Hessian with the
paper's recipe Aᵢ = ρ VᵢᵀVᵢ + 10 I_d.
```bash
python src/diff_lmu/diff_lmu_experiment.py --num_seed 200
```

**Figure 7** — SGD and SGDM with γ = 0.96 and their averaged versions with
decaying learning rates αₜ = α₀(max{n₀,t})^{-1/2}.
```bash
python src/decaying/decaying_experiment.py --num_seed 200
```

**Figures 8 and 12** — asymptotic normality of the averaged estimator.
```bash
python src/clt/clt_experiment.py
```
* Figure 8 — Frequency of Z for averaged SGD and averaged SGDM, γ = 0.9,
  n₀ = 1000, n = 2000, over 1000 trials.
* Figure 12 — The probability P(|Z| < 1.96) for SGD and SGDM, γ = 0.8, 0.9, over
  α = 2⁻¹¹, …, 2²; an n = 500-iteration trajectory with n₀ = 200 and 1000
  trials per learning rate.

**Figures 9-11** — sensitivity of SGD / SGDM to the learning rate. The two
panels of Figures 9 and 11 are (left) SGD with α ∈ {2⁻⁹, 2⁻⁷, 2⁻⁵} and
(right) SGDM with γ = 0.8 and α ∈ {2⁻⁹, …, 2⁻¹}.
```bash
python src/sensitivity/sensitivity_experiment.py --num_seed 200
```
* Figure 9 — convergence of the last iterate xₜ under different learning rates.
* Figure 10 — final error at T = 500 versus α = 2⁻¹¹, …, 2², γ = 0.8, 0.9.
* Figure 11 — averaged SGD and averaged SGDM, n₀ = 500.

**Figures 13-15** — MNIST with a two-layer tanh network (784 → 500 → 10).
B = 256, α = 1.0, seeds 1-3.
```bash
python src/mnist_tanh/mnist_tanh_experiment.py --download
```
* Figure 13 — Performance of SGD and SGDM (small / large γ).
* Figure 14 — Averaged SGD and averaged SGDM, n₀ = ⌊N/B⌋ = 234.
* Figure 15 — Averaged SGD and averaged SGDM, n₀ = 2⌊N/B⌋ = 468.

The network is defined in `src/mnist_tanh/nn_models.py`; its module docstring
gives the correspondence with the paper's `f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)`.

**Figure 16** — CIFAR-10 with a VGG-11 network (no batch normalisation).
α = 0.5, γ ∈ {0.7, 0.9, 0.99}, 50 epochs, 391 mini-batches per epoch (B = 128).
The learning rate is **not** rescaled by (1 − γ): the PyTorch step is
`SGD(lr = 0.5, momentum = γ)` for every momentum weight.

The panel is plotted from the per-iteration training losses behind the published
figure, which ship under `data/cifar_logs/` (4 momentum weights × 3 seeds ×
19,550 iterations). The default mode regenerates the published panel exactly
from them; `--mode train` re-trains from scratch instead.
```bash
python src/cifar/cifar_experiment.py            # exact, from the archived logs
python src/cifar/cifar_experiment.py --mode train --download   # re-train
```

**Figures EC.1-EC.3** — SGD/SGDM on the logistic loss (electronic companion,
Section EC.7.1). N = 20,000, d = 10, B = 0.2N, α = 0.5; average adaptive
momentum weight 0.75.
```bash
python src/logistic/logistic_experiment.py --num_seed 200
```
* Figure EC.1 — Performance of SGD and SGDM on the logistic loss.
* Figure EC.2 — Averaged SGD and averaged SGDM, n₀ = 10.
* Figure EC.3 — Same, n₀ = 40.

Panel (a) of each figure shows SGD, SGDM-0.3/0.5/0.7 and SGDM-adap; panel (b)
shows SGDM-adap, SGDM-0.8 and SGDM-0.9 — the (b) panels carry no SGD curve.
All six panels are drawn for `t = 0, …, 98`, over the axis limits EC.1:
3.3e-4 … 1.42, EC.2: 1e-4 … 2e-1, EC.3: 8e-5 … 4e-2.

The panels use the authors' plotting geometry — `figsize=(4, 4)`,
`left=0.1, right=0.9, top=0.9, bottom=0.2`, `markevery=0.3` — which is what
fixes the marker positions.

**Figure EC.4** — MNIST with multinomial logistic regression (electronic
companion, Section EC.7.2). B = 256, α = 1.0, γ ∈ {0.1, …, 0.99}. The published
panel is drawn for t = 0, …, 1000, so the curves are cut at `--max_iter 1000`
(its default) even though the run trains more epochs.
```bash
python src/mnist/mnist_experiment.py --download
```

## 5. Figure to code

| Figure | Script | Key parameters |
|---|---|---|
| 1 | `src/theory/lambda_heatmap.py` | κ = 5; (a) λ map, (b) SGDM-beats-SGD region |
| 2 | `src/quadratic/quadratic_experiment.py` | α = 0.001, γ = {0.5,0.7,0.9} (a) / {0.97,0.98,0.99} (b) |
| 3 | `src/quadratic/quadratic_experiment.py` | averaged, n₀ = 500 |
| 4 | `src/quadratic/quadratic_experiment.py` | γ = 0.96, n₀ = 250, 500 |
| 5 | `src/n0_sensitivity/n0_sensitivity_experiment.py` | γ = 0.96, n₀ = {250,500,750,1000} |
| 6 | `src/diff_lmu/diff_lmu_experiment.py` | ρ = {1,5,10} → L/μ = {3.5, 9.9, 14.7}, adaptive γ, n₀ = 250 |
| 7 | `src/decaying/decaying_experiment.py` | decayed α, γ = 0.96, n₀ = 250, 500 |
| 8 | `src/clt/clt_experiment.py` | n₀ = 1000, n = 2000, γ = 0.9, 1000 trials |
| 9 | `src/sensitivity/sensitivity_experiment.py` | last iterate, α ∈ {2⁻⁹,2⁻⁷,2⁻⁵} / γ = 0.8, α ∈ {2⁻⁹ … 2⁻¹} |
| 10 | `src/sensitivity/sensitivity_experiment.py` | γ = {0.8, 0.9}, T = 500, α = 2⁻¹¹ … 2² |
| 11 | `src/sensitivity/sensitivity_experiment.py` | averaged, γ = 0.8, n₀ = 500 |
| 12 | `src/clt/clt_experiment.py` | γ = {0.8, 0.9}, n = 500, n₀ = 200, 1000 trials, α = 2⁻¹¹ … 2² |
| 13 | `src/mnist_tanh/mnist_tanh_experiment.py` | tanh MLP, B = 256, α = 1.0 |
| 14 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 234 |
| 15 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 468 |
| 16 | `src/cifar/cifar_experiment.py` | VGG-11 (no BN), B = 128, α = 0.5, 50 epochs; drawn from `data/cifar_logs/` |
| EC.1 | `src/logistic/logistic_experiment.py` | α = 0.5, (a) γ = {0.3,0.5,0.7,adap} / (b) {adap,0.8,0.9}; `m₁ = g₁` |
| EC.2 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 10; `m₁ = g₁` |
| EC.3 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 40; `m₁ = g₁` |
| EC.4 | `src/mnist/mnist_experiment.py` | logistic regression, B = 256, α = 1.0 |

## 6. Notation and conventions

### 6.1 Notation

The code uses the paper's symbols for the paper's quantities, so a script can be
read against the text line by line:

| Paper | Code | Meaning |
|---|---|---|
| `N` | `N`, `--N` | training-set size |
| `d` | `d`, `--d` | dimension of the parameter vector `x` |
| `B` | `batch_size` | mini-batch size (`--batch_frac` as a fraction of `N`) |
| `n` | `n`, `--n` | number of iterations of the trajectory |
| `n₀` | `n0`, `--n0_hist` / `--n0_cov` / `--n0_list` | burn-in, i.e. the iteration averaging starts at |
| `α` | `alpha`, `--alpha` (`--lr` in the PyTorch scripts) | learning rate |
| `γ` | `gamma`, `--gamma` / `--gammas` / `--gamma_hist` | momentum weight (SGD is `γ = 0`) |
| `Σ`, `A` | `A` | population Hessian `E[Aᵢ]` |
| `Aᵢ`, `bᵢ` | `Ai`, `bi` | per-sample loss parameters |
| `x*` | `x_star` | minimiser of the population risk |
| `μ`, `L` | `mu`, `L` | smallest / largest eigenvalue of `A` |
| `κ = L/μ` | `kappa` | condition number |
| `xₜ` | `traj[t]` | last iterate (`traj` is zero-based, see below) |
| `x̄_{n₀+t}` | `averaged_errors(...)` | Polyak-averaged iterate |
| `Z` | `clt_projection(...)` | projected CLT statistic |

A script uses a numbered variant of a symbol when it runs more than one
configuration of the same kind — in `src/clt/clt_experiment.py`, `--n_hist` and
`--n_cov` are the paper's `n` for the histogram and the coverage panel, and
`gamma_hist` is its `γ` for the histogram.

**Indexing.** A trajectory from `sgdm_core.run_sgd` / `run_sgdm` is zero-based:
`traj[k]` is the paper's `x_{k+1}`, so `traj[0]` is the initial point `x₁`. The
Polyak average is formed from `traj[n0]` onwards, i.e. `x̄_{n₀+t}` for
`t = 1, 2, …`, exactly as in the definition.

### 6.2 Conventions

* **Update.** `m_t = γ m_{t-1} + (1-γ) g_t`, `x_{t+1} = x_t - α m_t`; SGD is the
  `γ = 0` case. In PyTorch this is `SGD(lr = α(1-γ), momentum = γ)`, except for
  Figure 16, which uses the unscaled `lr = α` (see section 4).
* **First step.** The logistic experiment (Figures EC.1-EC.3) starts the
  momentum from the raw mini-batch gradient, `m₁ = g₁`, rather than from
  `(1-γ) g₁`. Every other experiment uses the plain recursion above with
  `m₀ = 0`, i.e. `m₁ = (1-γ) g₁`.
* **Adaptive weight.** `γ = ((1 - μα)/(1 + μα))²`, with μ the smallest
  eigenvalue of the population Hessian.
* **Averaged iterate.** `x̄_{n₀+t} = (1/t) Σ_{j=n₀+1}^{n₀+t} x_j`.
* **Trajectory indexing.** The last-iterate panels start at `t = 0` with the
  initial point `x₀`, so the curve is `‖xₜ − x*‖` for `t = 0, 1, 2, …`; the
  averaged panels plot `x̄_{n₀+t}` for `t = 1, 2, …`, as the definition above
  produces.
* **Reported quantity.** The quadratic and logistic panels plot the squared
  error `‖x − x*‖²` averaged over replications by **squaring first and then
  averaging**, i.e. `mean_s ‖x_s − x*‖²` (`np.mean(err ** 2, axis=0)`). The
  MNIST and CIFAR-10 panels (Figures 13-16 and EC.4) instead plot, at iteration
  `t`, the **running mean of the per-iteration training loss over iterations
  1..t**, averaged over the three seeds; for Figures 14-15 the sequence being
  averaged is the loss of the Polyak-averaged parameters from `n₀` onwards
  (before `n₀` it is the ordinary mini-batch loss).
* **One stream per replication.** Each replication is driven by a single
  unbroken `numpy` stream, seeded with the replication index. The problem data
  and the mini-batch index sequence are drawn from it **in the authors' order**
  — `V = rand(N, d, d)` → `b_i = rand(N, d, 1)` → `u = rand(d, 1)` → batch
  indices — which is what the notebook prototypes do (`sgdm_core.draw_*` +
  `build_*` / `generate_*` with a caller-owned `rng`, and
  `make_batch_schedule(..., rng=rng)`). Starting the schedule from a freshly
  seeded stream instead gives a statistically equivalent but *different*
  realisation, so the released panels are no longer the authors' exact curves.
  SGD and SGDM share that one mini-batch sequence within a replication.
* Because the mini-batches do not depend on α or γ, every sweep (the learning
  rate grids of Figures 2-4, 6, 7, 9-11 and the CLT study) forms the mini-batch
  gradients once per replication with `sgdm_core.batch_gradient_sequence` and
  replays them for each configuration. This is numerically identical to
  re-averaging the batches for every configuration, only faster; it is a pure
  speed-up, not a different experiment.
