#!/usr/bin/env bash
# Reproduce every figure in the paper.
#
# Usage:
#   bash scripts/run_all.sh            # full reproduction (slow: several hours)
#   FAST=1 bash scripts/run_all.sh     # reduced-fidelity quick pass (minutes)
#   SKIP_NEURAL=1 bash scripts/run_all.sh   # skip the MNIST/CIFAR experiments
#
# Run from the repository root (the folder containing README.md).

set -e
cd "$(dirname "$0")/.."

if [ "${FAST:-0}" = "1" ]; then
  SEED=20
  SEED_CLT=100
  EPOCHS=1
  echo ">>> FAST mode: reduced number of replications"
else
  SEED=200
  SEED_CLT=1000
  EPOCHS=5
fi

echo ">>> Figure 1: spectral radius"
python src/theory/lambda_heatmap.py

echo ">>> Figures 2-4: quadratic loss"
python src/quadratic/quadratic_experiment.py --num_seed "$SEED"

echo ">>> Figure 5: sensitivity to n0"
python src/n0_sensitivity/n0_sensitivity_experiment.py --num_seed "$SEED"

echo ">>> Figure 6: different condition numbers"
python src/diff_lmu/diff_lmu_experiment.py --num_seed "$SEED"

echo ">>> Figure 7: decaying learning rates"
python src/decaying/decaying_experiment.py --num_seed "$SEED"

echo ">>> Figures 8 and 12: asymptotic normality (CLT)"
python src/clt/clt_experiment.py --num_seed "$SEED_CLT" \
    --num_seed_cov "$SEED_CLT"

echo ">>> Figures 9-11: sensitivity to the learning rate"
python src/sensitivity/sensitivity_experiment.py --num_seed "$SEED"

echo ">>> Figures EC.1-EC.3: logistic loss"
python src/logistic/logistic_experiment.py --num_seed "$SEED"

if [ "${SKIP_NEURAL:-0}" = "1" ]; then
  echo ">>> skipping the MNIST / CIFAR-10 experiments (SKIP_NEURAL=1)"
else
  echo ">>> Figures 13-15: MNIST with the tanh network"
  python src/mnist_tanh/mnist_tanh_experiment.py --download --epochs "$EPOCHS"

  echo ">>> Figure EC.4: MNIST with logistic regression"
  python src/mnist/mnist_experiment.py --download --epochs "$EPOCHS"

  # Figure 16 is drawn from the archived logs that produced the paper's panel
  # (data/cifar_logs/), which reproduces it exactly.  To re-train the
  # reconstruction from scratch instead, add "--mode train --download".
  echo ">>> Figure 16: CIFAR-10 with VGG-11"
  python src/cifar/cifar_experiment.py
fi

echo ">>> all figures written to results/"
