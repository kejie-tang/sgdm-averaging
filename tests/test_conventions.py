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
    generate_logistic_data, online_covariance, online_projected_variances,
    thresholded_inverse,
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


def test_remark8_uses_individual_gradients_at_each_iterate():
    """Individual-gradient moments differ from batch-mean or final-point moments."""
    Ai = np.ones((2, 1, 1))
    bi = np.array([[[1.]], [[-1.]]])
    trajectory = np.array([[[2.]], [[0.]]])
    schedule = np.array([0, 1, 0, 1])
    # S_hat = ((1^2+3^2)+((-1)^2+1^2))/4 = 3, H_hat=1, B=2.
    got = online_covariance(Ai, bi, trajectory, 2, schedule)
    assert np.allclose(got, [[1.5]])
    assert np.allclose(thresholded_inverse(np.array([[.1]]), 1.), [[2.]])


def test_remark8_projection_matches_full_sandwich_with_duplicate_draws():
    """The projected calculation preserves the full sampled sandwich estimator."""
    rng = np.random.RandomState(14)
    v = rng.randn(9, 3, 3)
    Ai = v.transpose(0, 2, 1) @ v + np.eye(3)
    bi = rng.randn(9, 3, 1)
    schedule = np.array([0, 0, 2, 4, 3, 3, 5, 6, 7, 8, 1, 1])
    trajectories = rng.randn(2, 4, 3, 1)
    phi = rng.randn(3, 1)
    expected = [float((phi.T @ online_covariance(
        Ai, bi, t, 3, schedule) @ phi).item()) for t in trajectories]
    bad = np.full((1, 4, 3, 1), np.nan)
    got = online_projected_variances(
        Ai, bi, np.concatenate((trajectories, bad)), 3, schedule, phi)
    assert np.allclose(got[:2], expected, rtol=1e-12, atol=1e-12)
    assert np.isnan(got[2])


def test_coverage_statistic_uses_sampled_covariance_and_paper_average():
    """Check the complete statistic, including the B scaling and burn-in index."""
    import importlib.util
    path = os.path.join(ROOT, "src", "clt", "clt_experiment.py")
    spec = importlib.util.spec_from_file_location("coverage_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.RandomState(13)
    v = rng.rand(20, 3, 3)
    data = build_quadratic_data(v, rng.rand(20, 3, 1), rng.rand(3, 1), shift=1.)
    data["phi"] = np.array([[1.], [0.], [0.]])
    n, n0, B = 12, 4, 5
    schedule = make_batch_schedule(20, B, n, 17, replace=True)
    # A Figure-12 trial must not call the full-data reference covariance.
    module.asymptotic_covariance = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("oracle covariance used in coverage trial"))
    got = module.coverage_trial(data, schedule, B, [.01], [.8], n, n0)
    traj = run_sgdm(data["Ai"], data["bi"], data["x_star"], data["initial"],
                    B, schedule, .01, .8, n)
    covariance = online_covariance(data["Ai"], data["bi"], traj, B, schedule)
    numerator = np.sqrt(n-n0)*(data["phi"].T @ (traj[n0:].mean(0)-data["x_star"])).item()
    expected = numerator/np.sqrt((data["phi"].T @ covariance @ data["phi"]).item())
    assert np.allclose(got, [[expected]], rtol=1e-12, atol=1e-12)


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
    """Every CLI exposes --help without importing optional training libraries."""

    scripts = []
    for exp in sorted(os.listdir(os.path.join(ROOT, "src"))):
        d = os.path.join(ROOT, "src", exp)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith(("_", "vgg", "nn_models")):
                scripts.append(os.path.join(d, f))
    assert scripts, "no scripts found"

    for s in scripts:
        r = subprocess.run([sys.executable, s, "--help"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"{s} --help failed:\n{r.stderr}"


def test_cifar_train_curves_honours_the_requested_gammas():
    """The CIFAR trainer must use the requested momentum values."""
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


def _run_all_commands(fast, directory):
    """Execute the shell runner with a recorder in place of training."""
    import json
    from pathlib import Path
    folder = Path(directory)
    bin_dir = folder / "bin"
    bin_dir.mkdir(exist_ok=True)
    recorder = folder / "record_commands.py"
    recorder.write_text(
        "import json, os, sys\n"
        "with open(os.environ['SGDM_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    python = bin_dir / "python"
    python.write_text('#!/usr/bin/env bash\n'
                      'exec "$SGDM_TEST_PYTHON" "$SGDM_TEST_RECORDER" "$@"\n')
    python.chmod(0o755)
    log = folder / f"commands_{fast}.jsonl"
    env = dict(os.environ, FAST=str(fast), SKIP_NEURAL="0",
               SGDM_TEST_PYTHON=sys.executable, SGDM_TEST_RECORDER=str(recorder),
               SGDM_TEST_LOG=str(log), PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_all.sh")],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_fast_runner_keeps_averaged_mnist_curves_nonempty():
    """FAST=1 must generate data for both averaged MNIST panels."""
    import contextlib
    import importlib.util
    import io
    import tempfile
    import types
    from unittest.mock import patch
    from matplotlib.axes import Axes

    with tempfile.TemporaryDirectory() as directory:
        full = _run_all_commands(0, directory)
        fast = _run_all_commands(1, directory)
        for script in ("src/mnist_tanh/mnist_tanh_experiment.py",
                       "src/mnist/mnist_experiment.py"):
            full_args = next(args for args in full if args[0] == script)
            assert full_args[full_args.index("--epochs") + 1] == "5"
            if "--seeds" in full_args:
                assert full_args[full_args.index("--seeds") + 1:] == ["1", "2", "3"]

        args = next(args for args in fast if args[0].startswith("src/mnist_tanh/"))
        models = types.ModuleType("nn_models")
        models.Net = object
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        path = os.path.join(ROOT, args[0])
        old_path = sys.path[:]
        captured, trained = [], []
        real_plot = Axes.plot

        def record_plot(axis, x, y, *positional, **kwargs):
            if kwargs.get("label", "").startswith("Ave-"):
                captured.append((np.asarray(x), np.asarray(y)))
            return real_plot(axis, x, y, *positional, **kwargs)

        def loss_sequence(seed, gamma, lr, epochs, batch_size, X, y,
                          n0_list, device, max_iter=None):
            trained.append((seed, gamma))
            n = min(epochs * (len(X) // batch_size), max_iter)
            losses = np.full(n, 0.3)
            return losses, {n0: losses.copy() for n0 in n0_list}

        try:
            with patch.dict(sys.modules, {"nn_models": models, "torch": torch}), \
                    contextlib.redirect_stdout(io.StringIO()):
                spec = importlib.util.spec_from_file_location("fast_mnist_test", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.load_mnist = lambda *a, **k: (np.zeros((60000, 1)), np.zeros(60000))
                module.run_one = loss_sequence
                argv = [path] + args[1:] + ["--out", directory]
                with patch.object(sys, "argv", argv), patch.object(Axes, "plot", record_plot):
                    module.main()
                assert captured and all(len(x) == len(y) > 0 for x, y in captured)
                assert {int(x[0]) for x, _ in captured} == {234, 468}
                assert {seed for seed, _ in trained} == {1}

                # Neither a short epoch budget nor a cutoff at n0 may produce an empty plot.
                for epochs, max_iter in ((1, 1000), (3, 468)):
                    before = len(trained)
                    argv = [path, "--epochs", str(epochs), "--max_iter", str(max_iter),
                            "--out", directory]
                    error = io.StringIO()
                    with patch.object(sys, "argv", argv), contextlib.redirect_stderr(error):
                        try:
                            module.main()
                        except SystemExit as exc:
                            assert exc.code == 2
                        else:
                            raise AssertionError("an empty averaging window was accepted")
                    assert len(trained) == before, "invalid settings started training"
                    assert "n0" in error.getvalue()
        finally:
            sys.path[:] = old_path



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
    import test_runtime
    fns += [v for k, v in sorted(vars(test_runtime).items()) if k.startswith("test_")]
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
