"""
Figure 1 (fig:lambda): contour/heatmap of the spectral radius lambda.

Reproduces the theoretical figure in Section 3.1 of the paper (the
"Linear convergence to a local neighborhood" discussion).

For a fixed condition number L/mu = kappa, the contraction factor lambda of
SGDM on a quadratic loss is a closed-form function of (alpha, gamma):

    lambda = lambda(phi, gamma),
    phi = min( alpha*sqrt(mu*L)/rho , 2(1+gamma)/(1-gamma) - alpha*sqrt(mu*L) )

with rho = sqrt(kappa).  The value plotted is the largest root of the
characteristic polynomial (see Theorem thm:lambda in the appendix).

The horizontal axis is alpha*sqrt(mu*L) in [0, 2], the vertical axis is gamma
in [0, 1]; brighter colours mean smaller lambda (faster convergence).
The minimum is attained near alpha ~ 1/sqrt(mu L) and
gamma* = (sqrt(5)-1)^2/(sqrt(5)+1)^2 ~ 0.146, lambda* ~ 0.382.

Outputs (results/theory/):
    fig_lambda.png, lambda_surface.npz

Run:  python src/theory/lambda_heatmap.py
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402


def spectral_radius_grid(kappa, n_grid=401):
    """Return (x, y, Z) where Z[i, j] = lambda(alpha*sqrt(muL), gamma).

    Rows index gamma in [0, 1], columns index alpha*sqrt(muL) in [0, 2].
    Entries are np.nan when the learning rate violates the stability condition.
    """
    x = np.linspace(0, 2, n_grid)      # alpha * sqrt(mu L)
    y = np.linspace(0, 1, n_grid)      # gamma
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, np.nan)
    for j in range(n_grid):
        for i in range(n_grid):
            aL = X[i, j] * math.sqrt(kappa)
            gamma = Y[i, j]
            if 2 * (1 + gamma) >= aL * (1 - gamma):
                if gamma < 1:
                    phi = min(aL / kappa, 2 * (1 + gamma) / (1 - gamma) - aL)
                else:
                    phi = aL / kappa
                c = gamma + 1 - (1 - gamma) * phi
                disc = c ** 2 - 4 * gamma
                if gamma < (1 - phi) ** 2 / (1 + phi) ** 2:
                    Z[i, j] = (c + math.sqrt(max(disc, 0.0))) / 2
                else:
                    Z[i, j] = math.sqrt(gamma)
            else:
                Z[i, j] = np.nan
    return x, y, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--n_grid", type=int, default=401)
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "results", "theory"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    x, y, Z = spectral_radius_grid(args.kappa, args.n_grid)

    fig = plt.figure(figsize=(5, 4), dpi=150)
    ax = fig.add_subplot(gridspec.GridSpec(
        1, 1, left=0.13, right=0.9, top=0.9, bottom=0.18, figure=fig)[0])
    im = ax.imshow(Z, extent=[0, 2, 0, 1], aspect="auto", cmap="viridis_r",
                   interpolation="bicubic", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\alpha\sqrt{\mu L}$")
    ax.set_ylabel(r"$\gamma$")
    fig.savefig(os.path.join(args.out, "fig_lambda.png"), bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(os.path.join(args.out, "lambda_surface.npz"),
                        x=x, y=y, Z=Z, kappa=args.kappa)
    print(f"[theory] results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
