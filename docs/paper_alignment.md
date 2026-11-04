# Alignment with SGDM-v2

This revision prioritizes fixing code while retaining the manuscript's mathematical
conventions and the appearance of the figures. The comparison baseline is commit
`ef465cc8507de36d67315c4ff7caca48bf23055c`. It is not assumed to regenerate every
image embedded in the PDF exactly.

## Changes that preserve the historical squared-error curves

* Quadratic gradients now use `A_i @ x - b_i`, with the positive minimizer in
  equation (4). The initial perturbation is reflected along with the minimizer.
  Thus the new iterate is the negative of the historical iterate for the same
  batches, and every squared error is unchanged (up to roundoff).
* The random tensor is stored as the transpose of the historical draw, then
  `A_i = rho * V_i.T @ V_i + shift * I`. Its entries remain iid Uniform(0,1),
  while the individual Hessians are unchanged. No manuscript change from
  `V_i.T @ V_i` to `V_i @ V_i.T` is needed.
* Logistic labels now follow the stated positive generating parameter. They
  complement the historical labels. From zero initialization the minimizer and
  iterates are reflected, preserving squared errors. This statement concerns
  the label correction, not the separate averaging-window correction.
* The decaying schedule implements the literal manuscript formula
  `alpha_t = alpha0 / sqrt(max(n0,t))`. When `--alpha0` is omitted, its coefficient
  is `--initial_lr * sqrt(n0)`, with default `--initial_lr 0.001`; this preserves
  the historical learning-rate sequence. Old commands that passed `--alpha0
  0.001` to mean the plateau should instead pass `--initial_lr 0.001`.

## Averaging, labels, and first-step exception

`traj[k]` stores the pre-update iterate `x_{k+1}`. Tail averaging now starts at
`traj[n0]`, retains `n-n0` points, and follows the manuscript literally. The old
extra shift affected Figures 3–7, EC.2–EC.3, and the CLT statistics; Figure 11
already used the literal endpoint. The alias `averaged_errors_shifted` now uses
the same endpoint as `averaged_errors`. Logistic averaged panels now use
`t=1,...,99` for their 99 retained points, instead of plotting the first mean
at t=0. Comparison plots labeled "Plot index" use a zero-based array index.

Figure 6 legend values are calculated from the mean realized condition numbers
across replications. The default rounds to one decimal, correcting the old 3.5
label to approximately 3.3. Custom rho values also receive calculated labels.

The logistic default retains the historical first update: in the manuscript's
indexing, `m_2 = g_1`, followed by the ordinary recursion for `t >= 2`. Changing it
to `(1-gamma)*g_1` changes the transients substantially. EC.7.1 should disclose
this exception in one sentence; it should not claim unchanged theorems under a
different initialization. `--first_momentum scaled` permits a comparison.

## Figures 8 and 12

Both CLT experiments now default to independent sampling **with replacement**,
matching the iid mini-batch model in Corollary 3. Add a short experimental
sampling sentence, rather than altering the corollary or its statistic.
`--sampling without-replacement` applies the matching finite-population factor
`(N-B)/(N-1)` automatically. Ordinary quadratic/logistic experiments keep their
historical sampling without replacement within each batch.

Figure 8 uses the full-data covariance at the simulation minimizer, as already
stated in the manuscript. Figure 12 defaults to the actual Remark 8 estimator:

```
S_hat = mean_{t,i in batch_t} (A_i x_t-b_i)(A_i x_t-b_i)^T
H_hat = mean_{t,i in batch_t} A_i
V_hat = H_tilde^{-1} S_hat H_tilde^{-1} / B
```

Hessian eigenvalues are floored at `lambda_lower/2`. The construction bound
`lambda_lower = 1.03/1.04` is known from the generated Hessians; it does not use
the empirical minimum eigenvalue or the oracle minimizer. The full matrix
estimator is available as `online_covariance`. For a multi-configuration sweep,
`online_projected_variances` calculates the exact projection of that estimator
by postprocessing stored trajectories and their sampled indices. It is not an
implementation of a streaming memory-saving algorithm, although it gives the
same estimator without substituting oracle gradients. A regression test compares
it against direct accumulation of the full sampled-gradient matrix.

`--covariance oracle` remains available for diagnostics. Non-finite statistics
count as non-coverage and remain in the replication denominator. The NPZ records
raw statistics, their finite masks, sampling, covariance method, retained-point
settings, and the projection vector.

Figure 12 retains the historical direction `ones/sqrt(d)` by default.
`--projection random` is a diagnostic: a fixed seed-1 uniform vector normalized
once after generating the fixed data. It is not a recovered setting for the PDF.
Do not choose a direction or estimator after looking at its agreement with the
old image. The baseline script itself differs visibly from the PDF's Figure 12;
a new figure must be identified by its recorded protocol, rather than called an
exact reproduction of that image. A single finite-horizon coverage curve assesses
normal calibration, rather than proving convergence in distribution.

## Manuscript changes still needed

The code cannot preserve both the current curves and the Gaussian description
in Section 5.2: use uniform entries, `A_i=(V_i.T@V_i+1.03I)/1.04`, uniform positive
`b_i`, and condition number approximately 14.5. Section 5.1 and equation (4) can
remain as written.

Keep the MNIST training and cumulative loss curves. Correct the model formula to
the actual 784–500–10 network with trainable biases, and replace the claimed
one-epoch moving average by a cumulative mean. The averaged-parameter loss
sequence retains ordinary burn-in losses in that cumulative mean; conclusions
about the burn-in are limited to the two tested values. The EC.4 linear model also
has a trainable bias. No neural-network training was performed for this revision.

CIFAR code, logs, and Figure 16 were not changed or run. Suggested corrections
from the source audit are batch size 128, cumulative-loss reporting, and the
phrase comparing SGDM with SGD. The original trainer was not preserved; metadata
and the reconstructed trainer do not prove every detail of the historical run.

## Reproducible comparisons

Run from the repository root with numpy and matplotlib:

```bash
python tests/test_conventions.py
python scripts/compare_alignment.py --phase quadratic --trials 200 --out results/alignment
python scripts/compare_alignment.py --phase logistic --trials 200 --out results/alignment
python scripts/compare_alignment.py --phase histogram --trials 200 --out results/alignment
python scripts/compare_alignment.py --phase clt --trials 200 --out results/alignment
```

These are screening comparisons, not complete reruns of every paper panel. The
quadratic screen uses N=20,000, d=10, B=4,000, n=2,000, rho=1, eight fixed momentum
weights and four burn-ins. It does not numerically rerun the rho=5/10 or decaying
panels at full scale. The logistic screen uses the default 141 stored iterates
and six fixed weights, not the adaptive-weight curve. The histogram and coverage
screens use the manuscript horizons but 200 rather than 1000 trajectories.
Reduced-size CLI smoke runs check the other modified simulation scripts.

The comparison utility reads the baseline module with `git show`, so that baseline
commit must be available locally. Raw arrays and JSON summaries are saved next to
each comparison PNG. The diagnostic random-direction comparison intentionally
changes direction as well as sampling/covariance and must not be interpreted as
isolating the covariance change.
