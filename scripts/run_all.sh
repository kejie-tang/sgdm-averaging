#!/usr/bin/env bash
# Reproduce every figure in the paper.
#
# Usage:
#   bash scripts/run_all.sh            # full reproduction (slow: several hours)
#   FAST=1 bash scripts/run_all.sh     # reduced-fidelity quick pass (minutes)
#
# Run from the repository root (the folder containing README.md).

set -e
cd "$(dirname "$0")/.."

if [ "${FAST:-0}" = "1" ]; then
  SEED=20
  SEED_CLT=100
  echo ">>> FAST mode: reduced number of replications"
else
  SEED=200
  SEED_CLT=1000
fi

echo ">>> Figure 1: spectral radius heat map"
python src/theory/lambda_heatmap.py

echo ">>> Figures 2-4: quadratic loss"
python src/quadratic/quadratic_experiment.py --num_seed "$SEED"

echo ">>> Figures 5-7: sensitivity to the learning rate"
python src/sensitivity/sensitivity_experiment.py --num_seed "$SEED"

echo ">>> fig:clt and fig:clt-alpha"
python src/clt/clt_experiment.py --num_seed "$SEED_CLT" \
    --num_seed_cov "$SEED_CLT"

echo ">>> Figures 7-9: logistic loss"
python src/logistic/logistic_experiment.py --num_seed "$SEED"

echo ">>> Figure 10: MNIST"
python src/mnist/mnist_experiment.py

echo ">>> all figures written to results/"
