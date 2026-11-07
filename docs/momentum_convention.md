# Momentum implementation

The experiment update is

```
m_{t+1} = γ m_t + (1 − γ) g_t
x_{t+1} = x_t − α m_{t+1}
```

The quadratic and CLT experiments implement it directly in `src/sgdm_core.py`.
The logistic experiments (Figures EC.1–EC.3) use the first update `m₂ = g₁`,
then apply the recursion above. Other experiments initialize `m₁ = 0`.

Figures 13–16 and EC.4 use `torch.optim.SGD` with zero dampening and no
Nesterov momentum:

```
buf_{t+1} = γ buf_t + g_t
x_{t+1} = x_t − lr buf_{t+1}
```

With `m_t = (1 − γ) buf_t` and `lr = α(1 − γ)`, this gives the same update.
The corresponding optimizer configuration is

```python
optim.SGD(params, lr=alpha * (1 - gamma), momentum=gamma)
```

It is used in `src/mnist_tanh/mnist_tanh_experiment.py`,
`src/mnist/mnist_experiment.py` and `src/cifar/cifar_experiment.py`.
