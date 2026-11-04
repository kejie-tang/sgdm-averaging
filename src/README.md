# `src/` — experiment code

Run each script from the repository root. Shared data generators, iterates and
statistical helpers are defined in `sgdm_core.py`.

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
| `cifar/` | 16 | `python src/cifar/cifar_experiment.py` |
| `logistic/` | EC.1, EC.2, EC.3 | `python src/logistic/logistic_experiment.py` |
| `mnist/` | EC.4 | `python src/mnist/mnist_experiment.py --download` |

`bash scripts/run_all.sh` runs all figures. `FAST=1` reduces replications and
training epochs; `SKIP_NEURAL=1` runs only the synthetic experiments.

## Shared conventions

- `N` is the dataset size, `d` the parameter dimension, `batch_size` the
  mini-batch size, `n` the trajectory length, `n0` the averaging burn-in,
  `alpha` the learning rate and `gamma` the momentum weight.
- `traj[k]` stores x_{k+1}; `traj[0]` is the initial point x_1.
- `averaged_errors(traj, x_star, n0, T)` starts at `traj[n0]`, evaluating
  xbar_{n0+t} = (1/t) sum_{j=n0+1}^{n0+t} x_j for t=1,...,T, up to n-n0.
- SGDM uses m_{t+1}=gamma*m_t+(1-gamma)*g_t and x_{t+1}=x_t-alpha*m_{t+1}.
  The logistic experiments initialize m_2=g_1; other experiments use m_1=0.
- In PyTorch, the same recursion is implemented with
  `SGD(lr=alpha*(1-gamma), momentum=gamma)`; see
  [momentum_convention.md](../docs/momentum_convention.md).
- Quadratic and logistic panels report the mean squared parameter error over
  replications. MNIST and CIFAR-10 panels report the cumulative mean training
  loss, averaged over seeds. For averaged MNIST parameters, losses before n0
  are the ordinary mini-batch losses.

## Sampling and computation

Synthetic error-curve experiments draw problem data followed by mini-batch
indices from one random stream per replication. Methods within a replication
share the same batches. `batch_gradient_sequence` caches mini-batch Hessian
and linear-term means for reuse across the quadratic parameter sweeps.

The CLT experiments use a fixed dataset and independent trajectory seeds.
Figure 8 samples with replacement and uses one fixed random projection.
Figure 12 samples each batch without replacement and uses one random
projection per trajectory, shared across methods and learning rates. Both use
the full-data reference covariance at x_star; Figure 12 includes (N-B)/(N-1).

Each script saves its figures and raw arrays under `results/<experiment>/`.
The main [README](../README.md) lists the parameters for each figure.

## Neural-network models

- Figures 13–15: `mnist_tanh/nn_models.py`, 784 → 500 (tanh) → 10, with biases
  in both linear layers.
- Figure EC.4: multinomial logistic regression with a bias term.
- Figure 16: `cifar/vgg.py`, VGG-11 without batch normalization. The default
  plotting command uses `data/cifar_logs/`; `--mode train --download` trains
  the network from scratch.
