# Momentum in `torch.optim.SGD` versus the paper's recursion

Every experiment in this release trains with the recursion of the paper,

```
m_t = γ m_{t-1} + (1 − γ) g_t                                 (paper)
x_t = x_{t-1} − α m_t
```

where `g_t` is the mini-batch gradient, `α` the learning rate and `γ` the
momentum weight (`γ = 0` is SGD). The convex experiments (Figures 2–12 and
EC.1–EC.4) implement this recursion directly, so no translation is involved.
The CIFAR-10 experiment of Figure 16 goes through
`torch.optim.SGD`, whose update rule is written differently, and the two must
not be confused: passing the paper's `α` straight into `lr` changes the
effective step size by a factor `1/(1 − γ)`.

This note records the exact correspondence, because it is a natural question
for a reader comparing the released code with the paper.

## 1. What PyTorch does

`torch.optim.SGD` with `momentum = γ`, no dampening and no Nesterov momentum
(`torch/optim/sgd.py`, momentum branch) computes

```
buf_t = γ · buf_{t-1} + g_t                                   (PyTorch)
p_t   = p_{t-1} − lr · buf_t
```

The gradient enters the buffer **unscaled**; the learning rate is applied
afterwards, to the whole buffer.

## 2. The correspondence

Define `m_t = (1 − γ) · buf_t`. Multiplying the buffer recursion by `(1 − γ)`
gives

```
m_t = γ · m_{t-1} + (1 − γ) · g_t
```

which is the paper's momentum recursion, and the parameter update becomes

```
p_t = p_{t-1} − lr · buf_t = p_{t-1} − lr · m_t / (1 − γ)
```

So PyTorch's `buf_t` is the paper's `m_t` scaled by `1/(1 − γ)`, and the
effective step size on `m_t` is

```
α_eff = lr / (1 − γ).
```

Setting

```python
optim.SGD(params, lr=alpha * (1 - gamma), momentum=gamma)
```

therefore gives `α_eff = α` and reproduces the paper's recursion exactly. That
is the form used throughout the release, including Figure 16.

Passing the paper's `α` directly, `lr = α`, instead trains with
`α_eff = α / (1 − γ)`:

| γ | `α_eff` at `α = 0.5` | relative to the paper |
|---|---|---|
| 0.0 | 0.50 | 1× |
| 0.7 | 1.67 | 3.3× |
| 0.9 | 5.00 | 10× |
| 0.99 | 50.0 | 100× |

At `γ = 0.9` that is ten times the intended step, well outside the stability
region, and the runs stay at the chance loss. Note that only `γ = 0` is
unaffected, which is why an SGD curve can look right while every SGDM curve is
flat.

## 3. Numerical check

`docs/verify_momentum_equivalence.py` runs both the paper's recursion and
`torch.optim.SGD` on twenty random gradients and reports the largest
discrepancy in the parameters. The two agree to machine precision with the
scaled learning rate, and disagree by a large margin without it:

```
$ python docs/verify_momentum_equivalence.py
lr = alpha*(1-gamma)       max|diff| vs paper = 2.220e-16   MATCH
lr = alpha  (unscaled)     max|diff| vs paper = 1.348e+01   differs
```

## 4. Where this matters in the release

* `src/cifar/cifar_experiment.py` (`run_one`) — `lr = lr * (1 - gamma)`.
* `src/mnist_tanh/mnist_tanh_experiment.py` — `lr * (1 - gamma)`.
* `src/mnist/mnist_experiment.py` — the same convention.

The convex experiments do not use PyTorch at all: `src/sgdm_core.run_sgdm`
implements `m = γm + (1 − γ)g`, `x = x − αm` verbatim, so their `α` and `γ` are
the paper's without translation.
