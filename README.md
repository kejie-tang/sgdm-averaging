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
├── data/              where the public MNIST / CIFAR-10 files come from
├── docs/              submission statements
└── results/           output figures and raw arrays
```

## 2. Requirements

* **Python >= 3.9**, **numpy**, **matplotlib** — all simulation scripts.
* **PyTorch >= 1.10** — the MNIST (Figure EC.4, Figures 13-15) and CIFAR-10
  (Figure 16) experiments.

```bash
pip install -r requirements.txt
```

The quadratic-loss simulations (Figures 2-7) are single-threaded and need
< 2 GB RAM and no GPU; the MNIST and CIFAR-10 experiments are much faster on a
GPU but also run on CPU.

## 3. Data

No data files are shipped with this repository.

* **Simulations** — synthetic, generated in-code; nothing to download.
* **MNIST** — public dataset, <http://yann.lecun.com/exdb/mnist/>. Fetched
  automatically with `--download`, or place the four IDX files under
  `data/MNIST/raw/`.
* **CIFAR-10** — public dataset, <https://www.cs.toronto.edu/~kriz/cifar.html>.
  Fetched automatically with `--download` (~170 MB), or place the extracted
  python batches under `data/CIFAR10/`.

See `data/README.md`.

## 4. Reproducing the figures

Run from the repository root. `bash scripts/run_all.sh` does everything
(`FAST=1` for a quick reduced-fidelity pass).

**Figure 1** — visualization of the spectral radius λ of the SGDM recursion.
```bash
python src/theory/lambda_heatmap.py --kappa 5
```

**Figures 2-4** — SGD / SGDM and their averaged versions on the quadratic loss.
N = 20,000, d = 10, B = 0.2N, α = 0.001, K = 1500, 200 replications; average
adaptive momentum weight 0.96.
```bash
python src/quadratic/quadratic_experiment.py --num_seed 200
```
* Figure 2 — Performance of SGD and SGDM on the quadratic loss.
* Figure 3 — Performance of x̄_{n₀+t} for Averaged SGD and Averaged SGDM, n₀ = 500.
* Figure 4 — Performance of xₜ for SGD and SGDM, and their averaged versions, γ = 0.96.

**Figure 5** — SGD and SGDM with γ = 0.96 and their averaged versions with
different n₀.
```bash
python src/n0_sensitivity/n0_sensitivity_experiment.py --num_seed 200
```

**Figure 6** — SGD and SGDM with the adaptive γ under different condition
numbers L/μ = 3.5, 9.9, 14.7 (ρ ∈ {1, 5, 10}), n₀ = 250.
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
* Figure 12 — The probability P(|Z| < 1.96) for SGD and SGDM, γ = 0.8, 0.9,
  over α; n = 1000, n₀ = 500, 1000 trials per learning rate.

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

**Figure 16** — CIFAR-10 with a VGG-11 network. α = 0.5, γ ∈ {0.7, 0.9, 0.99},
50 epochs. The run behind the published figure uses 391 mini-batches per epoch
(B = 128) — the paper's text quotes B = 256, which would give 195.
```bash
python src/cifar/cifar_experiment.py --download
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
shows SGDM-adap, SGDM-0.8 and SGDM-0.9 — the published (b) panels carry no SGD
curve. Both axes run to t = 100.

**Figure EC.4** — MNIST with multinomial logistic regression (electronic
companion, Section EC.7.2). B = 256, α = 1.0, γ ∈ {0.1, …, 0.99}.
```bash
python src/mnist/mnist_experiment.py --download
```

The MNIST and CIFAR-10 panels (Figures 13-16 and EC.4) report, at iteration t,
the **running mean of the per-iteration training loss over iterations 1..t**,
averaged over the three seeds — the authors' plotting convention. For Figures
14-15 the sequence being averaged is the loss of the Polyak-averaged
parameters from n₀ onwards (before n₀ it is the ordinary mini-batch loss).

## 5. Figure to code

| Figure | Script | Key parameters |
|---|---|---|
| 1 | `src/theory/lambda_heatmap.py` | kappa = 5 |
| 2 | `src/quadratic/quadratic_experiment.py` | α = 0.001, γ = {0.5,0.7,0.9} (a) / {0.97,0.98,0.99} (b) |
| 3 | `src/quadratic/quadratic_experiment.py` | averaged, n₀ = 500 |
| 4 | `src/quadratic/quadratic_experiment.py` | γ = 0.96, n₀ = 250, 500 |
| 5 | `src/n0_sensitivity/n0_sensitivity_experiment.py` | γ = 0.96, n₀ = {250,500,750,1000} |
| 6 | `src/diff_lmu/diff_lmu_experiment.py` | ρ = {1,5,10}, adaptive γ, n₀ = 250 |
| 7 | `src/decaying/decaying_experiment.py` | decayed α, γ = 0.96, n₀ = 250, 500 |
| 8 | `src/clt/clt_experiment.py` | n₀ = 1000, n = 2000, γ = 0.9, 1000 trials |
| 9 | `src/sensitivity/sensitivity_experiment.py` | last iterate, α ∈ {2⁻⁹,2⁻⁷,2⁻⁵} / γ = 0.8, α ∈ {2⁻⁹ … 2⁻¹} |
| 10 | `src/sensitivity/sensitivity_experiment.py` | γ = {0.8, 0.9}, T = 500, α = 2⁻¹¹ … 2² |
| 11 | `src/sensitivity/sensitivity_experiment.py` | averaged, γ = 0.8, n₀ = 500 |
| 12 | `src/clt/clt_experiment.py` | γ = {0.8, 0.9}, n = 1000, n₀ = 500, α = 2⁻¹¹ … 2² |
| 13 | `src/mnist_tanh/mnist_tanh_experiment.py` | tanh MLP, B = 256, α = 1.0 |
| 14 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 234 |
| 15 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 468 |
| 16 | `src/cifar/cifar_experiment.py` | VGG-11, B = 128, α = 0.5, 50 epochs |
| EC.1 | `src/logistic/logistic_experiment.py` | α = 0.5, (a) γ = {0.3,0.5,0.7,adap} / (b) {adap,0.8,0.9} |
| EC.2 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 10 |
| EC.3 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 40 |
| EC.4 | `src/mnist/mnist_experiment.py` | logistic regression, B = 256, α = 1.0 |

## 6. Conventions

* **Update.** `m_t = γ m_{t-1} + (1-γ) g_t`, `x_{t+1} = x_t - α m_t`; SGD is the
  `γ = 0` case. In PyTorch this is `SGD(lr = α(1-γ), momentum = γ)`.
* **Adaptive weight.** `γ = ((1 - μα)/(1 + μα))²`, with μ the smallest
  eigenvalue of the population Hessian.
* **Averaged iterate.** `x̄_{n₀+t} = (1/t) Σ_{j=n₀+1}^{n₀+t} x_j`.
* **Reported quantity.** The quadratic and logistic panels plot the squared
  error `‖x − x*‖²`; the MNIST and CIFAR-10 panels plot the running mean of the
  training loss, as described above.
* All random number generators are seeded per replication, and SGD and SGDM
  share the same pre-sampled mini-batch sequence within a replication.
* Because the mini-batches do not depend on α or γ, every sweep (the learning
  rate grids of Figures 2-4, 6, 7, 9-11 and the CLT study) forms the mini-batch
  gradients once per replication with `sgdm_core.batch_gradient_sequence` and
  replays them for each configuration. This is numerically identical to
  re-averaging the batches for every configuration, only faster; it is a pure
  speed-up, not a different experiment.

## 7. Scope of this release

Every figure in the paper's main text (Figures 1-16) and electronic companion
(Figures EC.1-EC.4) has a script here. Two caveats are worth stating.

* **Figure 16 (CIFAR-10) is a reconstruction, and it does not reproduce the
  published SGD curve.** The authors' original CIFAR training script was not
  preserved with the sources, so `src/cifar/` was rebuilt from the paper's
  description; its settings (learning rate 0.5, 50 epochs, 391 mini-batches per
  epoch, momentum weights, and the running-mean reporting rule) were recovered
  from the stored training logs. What it reproduces is the *ordering* and the
  qualitative picture of the paper: SGD is the least stable method (its
  per-iteration loss spikes to ~39, against ~10 for SGDM-0.9) and SGDM with
  γ = 0.99 converges fastest. What it does **not** reproduce is the final
  blow-up: in the paper SGD's loss runs away in the last few hundred iterations
  (Figure 16 rises past 2.5), whereas the reconstruction, like SGDM, keeps
  descending. The gap is a recipe detail that the lost script would have fixed
  (weight initialisation, weight decay, whether the 4-pixel crop/flip
  augmentation is applied, or the exact learning-rate convention). Batch
  normalisation is *required*: without it SGD does not move at all at α = 0.5
  (the loss stays at 2.303), so it is kept.
* The tanh network behind Figures 13-15 is a two-layer perceptron,
  784 → 500 (tanh) → 10. See the module docstring of
  `src/mnist_tanh/nn_models.py` for how it maps onto the paper's
  `f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)`.

A third, smaller, caveat: Figure 6 is labelled with the paper's nominal
condition numbers 3.5 / 9.9 / 14.7. The code computes them from the realised
population Hessian and prints 3.3 / 9.9 / 14.7 — the first panel differs only
because the paper rounds L/μ ≈ 35/10.
