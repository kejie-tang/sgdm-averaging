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
N = 20,000, d = 10, B = 0.2N, α = 0.001, K = 1500, 200 replications; average
adaptive momentum weight 0.96.
```bash
python src/quadratic/quadratic_experiment.py --num_seed 200
```
* Figure 2 — Performance of SGD and SGDM on the quadratic loss, with small
  momentum weights γ = 0.5, 0.7, 0.9 in panel (a) and large ones
  γ = 0.97, 0.98, 0.99 in panel (b).
* Figure 3 — Performance of x̄_{n₀+t} for Averaged SGD and Averaged SGDM, n₀ = 500.
  Panel (b) is labelled with the large momentum weights γ = 0.97/0.98/0.99 but
  plots the γ = 0.5/0.7/0.9 curves of panel (a); see section 7.
* Figure 4 — Performance of xₜ for SGD and SGDM, and their averaged versions, γ = 0.96.

**Figure 5** — SGD and SGDM with γ = 0.96 and their averaged versions with
different n₀.
```bash
python src/n0_sensitivity/n0_sensitivity_experiment.py --num_seed 200
```

**Figure 6** — SGD and SGDM with the adaptive γ under different condition
numbers L/μ = 3.5, 9.9, 14.7 (ρ ∈ {1, 5, 10}), n₀ = 250. Those labels are the
authors' own — their plotting cell hardcodes them. The script also computes the
condition numbers from the realised population Hessian with the paper's recipe
Aᵢ = ρ VᵢᵀVᵢ + 10 I_d and prints 3.3 / 9.9 / 14.7: the ρ = 5 and ρ = 10 values
match the caption, the ρ = 1 value does not. See section 7.
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

**Figure 16** — CIFAR-10 with a VGG-11 network (no batch normalisation).
α = 0.5, γ ∈ {0.7, 0.9, 0.99}, 50 epochs. The run behind the published figure
uses 391 mini-batches per epoch (B = 128) — the paper's text quotes B = 256,
which would give 195. The learning rate is **not** rescaled by (1 − γ) here:
the authors' log prints `lr = 0.5` for every momentum weight.

The panel is drawn from the per-iteration training losses that produced the
paper's figure; those series ship with this release under `data/cifar_logs/`
(4 momentum weights × 3 seeds × 19,550 iterations), so the default mode
regenerates the published panel exactly.
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
shows SGDM-adap, SGDM-0.8 and SGDM-0.9 — the published (b) panels carry no SGD
curve. All six panels are drawn for `t = 0, …, 98`, over the axis limits the
authors' own plotting cells use (EC.1: 3.3e-4 … 1.42; EC.2: 1e-4 … 2e-1;
EC.3: 8e-5 … 4e-2).

Two details of the authors' recursion and plotting code change the panels
visibly and are followed here. First, their *first* momentum is the raw
mini-batch gradient, `g₁ = ∇f(x₁)`, with no `(1 − γ)` factor
(`if k == 0: gt = gradient(...)`, applied to every γ). Since the recursion is
an AR(2) whose roots have modulus √γ, γ = 0.9 is only lightly damped, and
that ten-times-larger first step is what drives the γ = 0.9 curve in Figure
EC.1(b): it dips to ≈ 1e-2 near `t = 15` and then rises to a ≈ 8e-2 hump near
`t = 28`. Reinstating the `(1 − γ)` factor flattens the hump to ≈ 1.8e-2 and
moves it to `t = 40`, and makes Figure EC.2(b) qualitatively wrong — γ = 0.9
would then look like the *fastest* curve at large `t` instead of the slowest.
Second, the markers are placed with `markevery=0.3`; in current matplotlib a
float `markevery` spaces markers by that fraction of the axes-box diagonal
measured *along the curve*, so the published markers sit at irregular `t` and
are reproduced only with the authors' axes geometry
(`figsize=(4, 4)`, `left=0.1, right=0.9, top=0.9, bottom=0.2`).

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

The paper's EC.7.2 text describes this curve as "an average of the past
⌊N/B⌋ batches", but the authors' own plotting helper does not window the
series: `moving_ave2` returns `ma[t] = mean(loss[0:t])` (its window argument
is unused, and the call sites all use that helper rather than the true
right-aligned `moving_ave`). `src/mnist_tanh/`, `src/mnist/` and `src/cifar/`
follow the code, which is what the published panels show.

## 5. Figure to code

| Figure | Script | Key parameters |
|---|---|---|
| 1 | `src/theory/lambda_heatmap.py` | κ = 5; (a) λ map, (b) SGDM-beats-SGD region |
| 2 | `src/quadratic/quadratic_experiment.py` | α = 0.001, γ = {0.5,0.7,0.9} (a) / {0.97,0.98,0.99} (b) |
| 3 | `src/quadratic/quadratic_experiment.py` | averaged, n₀ = 500; (b) relabelled, see section 7 |
| 4 | `src/quadratic/quadratic_experiment.py` | γ = 0.96, n₀ = 250, 500 |
| 5 | `src/n0_sensitivity/n0_sensitivity_experiment.py` | γ = 0.96, n₀ = {250,500,750,1000} |
| 6 | `src/diff_lmu/diff_lmu_experiment.py` | ρ = {1,5,10} → L/μ = {3.5, 9.9, 14.7} (labels; realised {3.3, 9.9, 14.7}), adaptive γ, n₀ = 250 |
| 7 | `src/decaying/decaying_experiment.py` | decayed α, γ = 0.96, n₀ = 250, 500 |
| 8 | `src/clt/clt_experiment.py` | n₀ = 1000, n = 2000, γ = 0.9, 1000 trials |
| 9 | `src/sensitivity/sensitivity_experiment.py` | last iterate, α ∈ {2⁻⁹,2⁻⁷,2⁻⁵} / γ = 0.8, α ∈ {2⁻⁹ … 2⁻¹} |
| 10 | `src/sensitivity/sensitivity_experiment.py` | γ = {0.8, 0.9}, T = 500, α = 2⁻¹¹ … 2² |
| 11 | `src/sensitivity/sensitivity_experiment.py` | averaged, γ = 0.8, n₀ = 500 |
| 12 | `src/clt/clt_experiment.py` | γ = {0.8, 0.9}, n = 1000, n₀ = 500, α = 2⁻¹¹ … 2² |
| 13 | `src/mnist_tanh/mnist_tanh_experiment.py` | tanh MLP, B = 256, α = 1.0 |
| 14 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 234 |
| 15 | `src/mnist_tanh/mnist_tanh_experiment.py` | averaged, n₀ = 468 |
| 16 | `src/cifar/cifar_experiment.py` | VGG-11 (no BN), B = 128, α = 0.5, 50 epochs; drawn from `data/cifar_logs/` |
| EC.1 | `src/logistic/logistic_experiment.py` | α = 0.5, (a) γ = {0.3,0.5,0.7,adap} / (b) {adap,0.8,0.9}; `m₁ = g₁` |
| EC.2 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 10; `m₁ = g₁` |
| EC.3 | `src/logistic/logistic_experiment.py` | averaged, n₀ = 40; `m₁ = g₁` |
| EC.4 | `src/mnist/mnist_experiment.py` | logistic regression, B = 256, α = 1.0 |

## 6. Conventions

* **Update.** `m_t = γ m_{t-1} + (1-γ) g_t`, `x_{t+1} = x_t - α m_t`; SGD is the
  `γ = 0` case. In PyTorch this is `SGD(lr = α(1-γ), momentum = γ)`. Figure 16
  is the exception: the authors' CIFAR log prints the same `lr = 0.5` for every
  momentum weight, so there the PyTorch learning rate is the unscaled α (see
  section 4).
* **First step.** The logistic experiment (Figures EC.1-EC.3) starts the
  momentum from the raw mini-batch gradient, `m₁ = g₁`, rather than from
  `(1-γ) g₁`, because that is what the authors' notebook does; see section 4
  for why it matters. Every other experiment uses the plain recursion above
  with `m₀ = 0`, i.e. `m₁ = (1-γ) g₁`, which is also what the corresponding
  authors' notebooks do.
* **Adaptive weight.** `γ = ((1 - μα)/(1 + μα))²`, with μ the smallest
  eigenvalue of the population Hessian.
* **Averaged iterate.** `x̄_{n₀+t} = (1/t) Σ_{j=n₀+1}^{n₀+t} x_j`.
* **Trajectory indexing.** The last-iterate panels start at `t = 0` with the
  initial point `x₀`, so the curve is `‖xₜ − x*‖` for `t = 0, 1, 2, …`; the
  averaged panels plot `x̄_{n₀+t}` for `t = 1, 2, …`, exactly as the paper's
  definition produces. The logistic panels (EC.1-EC.3) are all drawn over
  `t = 0, …, 98`, the range the authors' runs cover.
* **Reported quantity.** The quadratic and logistic panels plot the squared
  error `‖x − x*‖²` averaged over replications by **squaring first and then
  averaging**, i.e. `mean_s ‖x_s − x*‖²`, which is what the authors' plotting
  code does (`np.mean(err ** 2, axis=0)`); the MNIST and CIFAR-10 panels plot
  the running mean of the training loss, as described above.
* All random number generators are seeded per replication, and SGD and SGDM
  share the same pre-sampled mini-batch sequence within a replication. The
  logistic experiment draws the features, the labels and the mini-batch index
  sequence from a single unbroken stream per replication, which is how the
  authors' notebook generates them.
* Because the mini-batches do not depend on α or γ, every sweep (the learning
  rate grids of Figures 2-4, 6, 7, 9-11 and the CLT study) forms the mini-batch
  gradients once per replication with `sgdm_core.batch_gradient_sequence` and
  replays them for each configuration. This is numerically identical to
  re-averaging the batches for every configuration, only faster; it is a pure
  speed-up, not a different experiment.

## 7. Scope of this release

Every figure in the paper's main text (Figures 1-16) and electronic companion
(Figures EC.1-EC.4) has a script here, and each one reproduces the published
panel. A few points are worth stating explicitly, because in every case the
released code follows the authors' own code — or their stored logs — rather
than the paper's prose.

* **Figure 3, panel (b): the legend labels do not match the curves.** Panel (b)
  is labelled SGDM-0.97 / SGDM-0.98 / SGDM-0.99, but the curves it draws are the
  SGDM-0.5 / 0.7 / 0.9 curves of panel (a). The authors' plotting cell indexes
  the momentum array with `i` where it means `i + 4`:

  ```python
  for i in range(3):
      ax1.plot(loss_sgdm_array[:, :, i] ** 2, label="SGDM-{}".format(beta_set[i + 4]))
  ```

  with `beta_set = [0.5, 0.7, 0.9, gamma, 0.97, 0.98, 0.99]`. Re-running the
  authors' own notebook on 200 replications confirms it: the three curves of
  panel (b) start at ≈ 4.4e-8 / 4.2e-8 / 2.4e-8, which is what the
  γ = 0.5 / 0.7 / 0.9 runs give (4.67e-8 / 4.16e-8 / 2.18e-8), not what the
  γ = 0.97 / 0.98 / 0.99 runs give (8.16e-9 / 2.93e-7 / 4.70e-5) — the
  γ = 0.99 value alone sits two orders of magnitude above the panel's upper
  axis limit of 1e-7. `src/quadratic/quadratic_experiment.py` reproduces the
  published panel by default; setting `FIG3B_HONEST = True` at the top of that
  file plots the true large-momentum curves instead.

* **Figure 6: for ρ = 1 the condition number is 3.3, not the 3.5 of the
  caption and legend.** The code computes L/μ from the realised population
  Hessian with the paper's own recipe `Aᵢ = ρ VᵢᵀVᵢ + 10 I_d`. For ρ = 5 and
  ρ = 10 it returns 9.9 and 14.7, matching the caption exactly; for ρ = 1 it
  returns 3.3 where the caption quotes 3.5. Solving the same recipe by hand
  gives 3.31, so this is a typo in the caption, not a code error. The figure
  itself is drawn with the legend's value, as the published one is.

* **Section EC.7.1 draws the logistic labels from the complementary
  probability.** The prose sets `b_i = 1` with probability
  `p_x(a_i) = 1/(1 + exp(-x^T a_i))` with `x = θ₀`, but the authors' notebook
  draws `y = (sigmoid(θ₀^T a) < U(0,1))`, i.e. with probability
  `1 - sigmoid(θ₀^T a) = sigmoid(-θ₀^T a)`. The released generator follows the
  code, because that is what produced Figures EC.1-EC.3. The difference is
  immaterial to the figures — it only flips the sign of the population solution,
  and since `a_i` is centred Gaussian the marginal law of every error curve is
  unchanged (the realised adaptive weight is 0.75 under both conventions).

* **Figure 16 (CIFAR-10) is reproduced exactly from the archived loss logs, and
  a from-scratch re-training is provided alongside.** The panel is not
  approximated: its four curves are plotted from the per-iteration training
  losses that generated the published figure, which ship with this release
  under `data/cifar_logs/` (4 momentum weights × 3 seeds × 19,550 iterations).
  `src/cifar/cifar_experiment.py` regenerates it in its default `--mode plot`.

  The authors' original CIFAR training script was not preserved, so
  `src/cifar/vgg.py` and the `--mode train` loop are a reconstruction from the
  paper's description and those logs (VGG-11 without batch normalisation,
  B = 128, 50 epochs, α = 0.5, weight decay 5e-4, and the running-mean
  reporting rule). The three SGDM curves are in the stable regime and the
  reconstruction tracks them closely: a mean absolute deviation of 0.11-0.19 in
  the running mean against the authors' three-seed mean, compared with 0.5 or
  more — and with the deviations in the wrong direction — when batch
  normalisation is left in. None of the four curves in the shipped panel depends
  on that reconstruction in any case: `fig_cifar.png` is drawn from the stored
  series.

  The SGD curve is the one feature that does not transfer to `--mode train`, and
  the reason is identified: at α = 0.5 the no-batch-norm network is
  **bistable**, and which side a given seed lands on — running away to a good
  solution, or sitting at chance level log(10) = 2.3036 — is decided by the
  exact random stream (weight initialisation, augmentation and shuffle order),
  which was not preserved. In the archived logs two of the three seeds stagnate
  for the whole run while the third escapes around epoch 6, descends to a
  per-epoch mean of ≈ 0.45 and blows up at the start of epoch 48 (iteration
  18,447) — that single blow-up is what makes the spike at the right edge of the
  published panel. Re-running the released reconstruction with the same three
  seeds lands on the other mixture: all three seeds escape smoothly, none blows
  up, so its seed-mean SGD curve falls from 2.294 to 0.428 per epoch and from
  2.294 to 0.787 in the running mean, rather than to ≈ 1.5 with a spike.
  Matching the published SGD curve would require the original seeds, weight
  initialisation and data-augmentation RNG, none of which are recorded. That is
  why the released `fig_cifar.png` is drawn from the archived logs rather than
  from `--mode train`.

* The tanh network behind Figures 13-15 is a two-layer perceptron,
  784 → 500 (tanh) → 10. See the module docstring of
  `src/mnist_tanh/nn_models.py` for how it maps onto the paper's
  `f_i(X) = H(tanh(X^T a_1i) a_2i, b_i)`.
