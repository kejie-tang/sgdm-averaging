"""Regression tests for the conventions the paper's figures depend on.

Checks cover averaging indices, data generation, momentum, covariance,
learning-rate schedules and the experiment command-line interfaces.

Run:  python -m pytest tests/ -q          (or: python tests/test_conventions.py)
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sgdm_core import (  # noqa: E402
    adaptive_gamma, asymptotic_covariance, averaged_errors,
    decaying_schedule, make_batch_schedule, run_sgd,
    run_sgdm, build_quadratic_data,
    generate_logistic_data,
)


# --------------------------------------------------------------------------- #
# Averaging index
# --------------------------------------------------------------------------- #
def _toy_traj(n=30, d=3, seed=0):
    rng = np.random.RandomState(seed)
    traj = rng.rand(n, d, 1)
    return traj, rng.rand(d, 1)


def test_averaged_errors_matches_bruteforce():
    """averaged_errors must equal its definition, sum starting at traj[n0]."""
    traj, xs = _toy_traj()
    n0, T = 5, 10
    got = averaged_errors(traj, xs, n0, T)
    want = np.array([
        np.linalg.norm(traj[n0:n0 + t].mean(axis=0) - xs)
        for t in range(1, T + 1)
    ])
    assert np.allclose(got, want), (got, want)


def test_averaged_errors_returns_n_minus_n0_points():
    traj, xs = _toy_traj(n=40)
    n0 = 10
    assert averaged_errors(traj, xs, n0, 10_000).size == 40 - n0


def test_n0_out_of_range_raises():
    traj, xs = _toy_traj()
    for bad in (-1, traj.shape[0], traj.shape[0] + 5):
        try:
            averaged_errors(traj, xs, bad, 3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"n0={bad} should raise")


# --------------------------------------------------------------------------- #
# SGD is the gamma = 0 case of SGDM
# --------------------------------------------------------------------------- #
def test_sgd_equals_sgdm_at_gamma_zero():
    n, d = 50, 4
    rng = np.random.RandomState(1)
    Ai = rng.rand(n, d, d) * 0.05 + np.eye(d)
    bi = rng.rand(n, d, 1)
    xs = rng.rand(d, 1)
    x0 = rng.rand(d, 1)
    schedule = make_batch_schedule(n, 8, 30, seed=0, replace=False)
    a = run_sgd(Ai, bi, xs, x0, 8, schedule, 0.01, 30)
    b = run_sgdm(Ai, bi, xs, x0, 8, schedule, 0.01, 0.0, 30)
    assert np.array_equal(a, b)


def test_gamma_accepts_a_per_iteration_schedule():
    """A constant gamma sequence must equal the scalar gamma."""
    n, d = 40, 3
    rng = np.random.RandomState(2)
    Ai = rng.rand(n, d, d) * 0.05 + np.eye(d)
    bi = rng.rand(n, d, 1)
    xs = rng.rand(d, 1); x0 = rng.rand(d, 1)
    schedule = make_batch_schedule(n, 8, 20, seed=1, replace=False)
    scalar = run_sgdm(Ai, bi, xs, x0, 8, schedule, 0.01, 0.9, 20)
    vector = run_sgdm(Ai, bi, xs, x0, 8, schedule, 0.01, np.full(20, 0.9), 20)
    assert np.array_equal(scalar, vector)


# --------------------------------------------------------------------------- #
# Covariance: Omega/B versus the finite-population correction
# --------------------------------------------------------------------------- #
def test_fpc_factor_and_full_batch_limit():
    n, d, B = 1000, 3, 200
    rng = np.random.RandomState(0)
    Ai = np.array([np.eye(d) * (1.0 + rng.rand()) for _ in range(n)])
    bi = rng.rand(n, d, 1)
    A = Ai.mean(0)
    xs = np.linalg.solve(A, bi.mean(0))
    c_plain = asymptotic_covariance(Ai, bi, xs, A, B)
    c_fpc = asymptotic_covariance(Ai, bi, xs, A, B, replace=False)
    factor = (n - B) / (n - 1)
    assert np.allclose(c_fpc, c_plain * factor)
    # sampling the whole population carries no noise
    c_full = asymptotic_covariance(Ai, bi, xs, A, n, replace=False)
    assert np.allclose(c_full, 0.0)


# --------------------------------------------------------------------------- #
# Decaying schedule
# --------------------------------------------------------------------------- #
def test_decaying_schedule_shape():
    n0, n, a0 = 100, 500, 0.01
    a = decaying_schedule(n0, n, a0)
    assert a.shape == (n,)
    assert np.allclose(a[:n0], a0 / np.sqrt(n0))                       # constant up to n0
    t = np.arange(n0, n + 1)
    assert np.allclose(a[n0 - 1:], a0 / np.sqrt(t))  # then decays


def test_adaptive_gamma_in_unit_interval():
    for mu, alpha in ((1.8, 0.001), (10.0, 0.05), (0.1, 0.5)):
        g = adaptive_gamma(mu, alpha)
        assert 0.0 <= g < 1.0


# --------------------------------------------------------------------------- #
# Experiment command-line interfaces
# --------------------------------------------------------------------------- #
def test_all_scripts_import_and_expose_help():
    """Every script parses its CLI.

    The neural-network scripts import torch at module level, so they are only
    checked when torch is installed; the rest need numpy and matplotlib only.
    """
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    needs_torch = {"cifar_experiment.py", "mnist_experiment.py",
                   "mnist_tanh_experiment.py"}

    scripts = []
    for exp in sorted(os.listdir(os.path.join(ROOT, "src"))):
        d = os.path.join(ROOT, "src", exp)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith(("_", "vgg", "nn_models")):
                scripts.append(os.path.join(d, f))
    assert scripts, "no scripts found"

    skipped = 0
    for s in scripts:
        if not has_torch and os.path.basename(s) in needs_torch:
            skipped += 1
            continue
        r = subprocess.run([sys.executable, s, "--help"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{s} --help failed:\n{r.stderr}"
    if skipped:
        print(f"    (skipped {skipped} torch-only scripts)")


def test_cifar_train_curves_honours_the_requested_gammas():
    """The CIFAR trainer must use the requested momentum values."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        print("    (skipped: torch not installed)")
        return
    import types
    sys.path.insert(0, os.path.join(ROOT, "src", "cifar"))
    import cifar_experiment as ce

    seen = []
    real_run_one = ce.run_one
    real_load = ce.load_cifar10

    class _FakeX:
        shape = (50000, 3, 32, 32)

    ce.run_one = lambda seed, gamma, *a, **k: (seen.append(gamma)
                                               or np.zeros(5))
    ce.load_cifar10 = lambda root, download=False: (_FakeX(), np.zeros(50000))
    try:
        args = types.SimpleNamespace(data_root=".", download=False, seeds=[1],
                                     lr=0.5, epochs=1, batch_size=128,
                                     max_iter=5)
        ce.train_curves(args, "cpu", [0.0, 0.2])
    finally:
        ce.run_one = real_run_one
        ce.load_cifar10 = real_load
    assert seen == [0.0, 0.2], f"requested [0.0, 0.2] but trained {seen}"



def test_quadratic_minimizer_and_initial_distance():
    rng = np.random.RandomState(9)
    V = rng.rand(30, 3, 3)
    bi = rng.rand(30, 3, 1)
    u = rng.rand(3, 1)
    data = build_quadratic_data(V, bi, u)
    xs = data["x_star"]
    assert np.allclose(data["A"] @ xs - data["b"], 0)
    assert np.isclose(np.linalg.norm(data["initial"] - xs), 0.1)
    assert np.linalg.eigvalsh(data["Ai"]).min() >= 10
    # The quadratic objective must increase away from its minimizer.
    def loss(x):
        return float((0.5 * x.T @ data["A"] @ x - data["b"].T @ x).item())
    for _ in range(5):
        assert loss(xs + rng.randn(3, 1)) > loss(xs)


def test_logistic_labels_match_positive_paper_parameter():
    X, y, theta = generate_logistic_data(50, 3, 3)
    rng = np.random.RandomState(3)
    want_X = rng.multivariate_normal(np.zeros(3), np.eye(3), size=50)
    draws = rng.random(50)
    assert np.array_equal(X, want_X)
    assert np.array_equal(y, draws <= (1 / (1 + np.exp(-X @ theta))).ravel())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
