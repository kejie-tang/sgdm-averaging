"""Check that ``SGD(lr = alpha*(1-gamma), momentum = gamma)`` reproduces the
paper's recursion.

Paper (src/sgdm_core.run_sgdm):

    m_t = gamma * m_{t-1} + (1 - gamma) * g_t
    x_t = x_{t-1} - alpha * m_t

torch.optim.SGD (momentum branch, no dampening, no Nesterov):

    buf_t = momentum * buf_{t-1} + g_t          <- gradient enters unscaled
    p_t   = p_{t-1} - lr * buf_t

With lr = alpha*(1-gamma) the two coincide; with lr = alpha the effective step
is alpha/(1-gamma) times too large.  See docs/momentum_convention.md.

Run:  python docs/verify_momentum_equivalence.py
"""
from __future__ import annotations

import numpy as np
import torch


def paper_recursion(x0, grads, alpha, gamma):
    """The recursion of src/sgdm_core.run_sgdm, step by step."""
    x = x0.copy()
    m = np.zeros_like(x)
    out = []
    for g in grads:
        m = gamma * m + (1.0 - gamma) * g
        x = x - alpha * m
        out.append(x.copy())
    return np.array(out)


def torch_recursion(x0, grads, lr, gamma):
    """The same trajectory through torch.optim.SGD."""
    p = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr, momentum=gamma)
    out = []
    for g in grads:
        opt.zero_grad()
        p.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        out.append(p.detach().numpy().copy())
    return np.array(out)


def main():
    rng = np.random.default_rng(0)
    d, n = 5, 20
    alpha, gamma = 0.5, 0.9
    x0 = rng.normal(size=d)
    grads = rng.normal(size=(n, d))

    ref = paper_recursion(x0, grads, alpha, gamma)
    print(f"alpha = {alpha}, gamma = {gamma}, {n} steps, d = {d}\n")
    for name, lr in (("lr = alpha*(1-gamma)", alpha * (1.0 - gamma)),
                     ("lr = alpha  (unscaled)", alpha)):
        got = torch_recursion(x0, grads, lr, gamma)
        err = float(np.max(np.abs(got - ref)))
        verdict = "MATCH" if err < 1e-10 else "differs"
        print(f"{name:26s} max|diff| vs paper = {err:.3e}   {verdict}")

    # the effective step size each choice implies
    print("\neffective step on the paper's m_t, alpha_eff = lr/(1-gamma):")
    for g in (0.0, 0.7, 0.9, 0.99):
        print(f"  gamma={g:<5}: lr=alpha  ->  alpha_eff = {alpha/(1-g):7.2f}"
              f"   ({1/(1-g):5.1f}x the paper's alpha)"
              f" | lr=alpha*(1-gamma) -> {alpha:.2f}  (1.0x)")


if __name__ == "__main__":
    main()
