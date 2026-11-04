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

The author prioritizes retaining Figure 12's published coverage pattern. Git
history supplies a protocol that predates the fixed-ones/n=1000 revision:
commit `ddcb561` records n=500, n0=200 and one random projection per trajectory;
`7ba5f07` adds the covariance correction for within-batch sampling without
replacement. The fixed-ones direction and n=1000, n0=500 were introduced later
in `ef465cc`. They are not an established protocol for the supplied PDF image.
The original notebook and raw values are unavailable. Historical comments that
refer to a notebook are evidence of the repository's history, not independent
access to that notebook.

Figure 12 now defaults to n=500, n0=200, sampling without replacement inside
independent batches, and a unit-normalized vector with iid Uniform(0,1) entries
from `RandomState(seed+12345)` for each trajectory. That direction is shared
across every learning rate and momentum weight in the trajectory. The Hessian
remains the author-requested `V_i.T@V_i+I`. Its reference covariance is

```
V_B = (N-B)/(B*(N-1)) * H^{-1} S_star H^{-1}
S_star = mean_i (A_i x_star-b_i)(A_i x_star-b_i)^T
```

This is an oracle covariance at the known simulation minimizer, not the online
estimator of Remark 8. Retaining the published figure's qualitative results
therefore requires a short correction of its n/n0, covariance and projection
sentences, including the reference on manuscript page 16. The theoretical
estimator and its analysis in Remark 8 remain intact. Historical reconstruction
with a simplified Hessian must not be described as exact recovery of the missing
raw values or proof that the original image used the same protocol.

Figure 8 keeps n=2000, n0=1000, full-data reference covariance and sampling with
replacement. `--sampling` explicitly overrides sampling for both figures;
otherwise the NPZ records their different default sampling methods separately.
Non-finite statistics count as non-coverage and remain in the denominator.
The NPZ records every trajectory's direction, raw Z values and retained horizon.

The actual Remark 8 implementation remains available through `--covariance
online`: sampled pre-update gradients estimate S_hat, sampled Hessians estimate
H_hat, eigenvalues are floored at `lambda_lower/2` (lambda_lower=1), and
`V_hat=H_tilde^{-1} S_hat H_tilde^{-1}/B` with the matching sampling correction.
`online_projected_variances` postprocesses stored trajectories and sampled indices
and agrees with direct full-matrix accumulation. It is not a streaming memory
implementation. The previous online/iid/fixed-ones diagnostic is explicit:

```bash
python src/clt/clt_experiment.py --n_cov 1000 --n0_cov 500 \
    --covariance online --sampling with-replacement --projection ones
```

The restored protocol has been checked at N=20,000, d=10, B=4,000 over
1000 trajectory seeds. At alpha=2^-8, coverage is 0.141/0.146/0.154; at
2^-7 it is 0.789/0.841/0.903 for SGD/SGDM-0.8/SGDM-0.9. The plateau is
approximately 0.96-0.97 and the first zero-coverage grid values on the right
are 2^-3, 2^0 and 2^1. This recovers the published pattern, with remaining
pointwise differences. Figure 8 was not rerun at 1000 trials in this revision.

The direction and horizon are recovered from Git history rather than searched
for their fit to the PDF. An exact-source match is unresolved without the missing
original notebook/raw data. One finite-horizon coverage curve assesses normal
calibration rather than proving convergence in distribution.

## Simplified Section 5.2 setting

Following the author's preference, Figures 9–12 now use `A_i=V_i.T@V_i+I`.
The historical normalization is retained only in comparison controls, not in
the experiment defaults. The manuscript's matrix formula can therefore remain
as written; its Gaussian generation and condition number still need correction.
The 200-trajectory Figure-12 results from commit d76c722 used the previous
normalized Hessians and are historical diagnostics, not results from this setting.
Figure 8 and Section 5.1 keep their `+10I` setting.
Across data seeds 0–199, the mean condition number is 14.70561. A 20-seed
paired check at N=20,000, d=10, B=4,000 retained the same finite-horizon
stable grid limits (1/16, 1/2, 1 for gamma=0, 0.8, 0.9). A separate ten-trajectory
online-covariance pilot ran both Hessians on identical with-replacement batches;
these are historical diagnostics of the earlier online/iid setting, separate
from the restored oracle Figure-12 protocol.

## Manuscript changes still needed

To describe the simplified Section 5.2 experiments, use uniform entries, `A_i=V_i.T@V_i+I`, uniform positive
`b_i`, and condition number approximately 14.7. Section 5.1 and equation (4) can
remain as written. For Figure 12, correct n to 500, n0 to 200,
specify the full-data covariance with finite-population correction and per-trial
random projection, and remove the assertion that this plot validates Remark 8.

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
