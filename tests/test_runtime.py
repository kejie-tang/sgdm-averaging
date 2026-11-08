"""Small-input checks of experiment entry points, downloads and packaging."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def experiment(relative):
    old_path = sys.path[:]
    path = ROOT / "src" / relative
    try:
        spec = importlib.util.spec_from_file_location("runtime_" + path.parent.name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path


def call_main(module, args):
    with patch.object(sys, "argv", [module.__file__, *args]), \
            contextlib.redirect_stdout(io.StringIO()):
        module.main()


def test_synthetic_cli_sweeps_and_labels():
    """Extra curves must not crash plotting or acquire a false momentum label."""
    from matplotlib.axes import Axes
    cases = [
        ("clt/clt_experiment.py", ["--figure", "12", "--num_seed_cov", "2",
          "--n_cov", "12", "--n0_cov", "4", "--workers", "1", "--exps", "-9", "-7",
          "--gammas", "0.2", "0.4", "0.6", "0.8"]),
        ("sensitivity/sensitivity_experiment.py", ["--n", "12", "--T", "5", "--n0", "5",
          "--num_seed", "1", "--gammas", "0.2", "0.4", "0.6"]),
        ("n0_sensitivity/n0_sensitivity_experiment.py", ["--n", "12", "--num_seed", "1",
          "--n0_list", "0", "1", "2", "3", "4"]),
        ("diff_lmu/diff_lmu_experiment.py", ["--n", "12", "--num_seed", "1", "--n0", "2",
          "--rho_list", "1", "2", "3", "4"]),
        ("decaying/decaying_experiment.py", ["--n", "12", "--num_seed", "1",
          "--n0_list", "2", "4", "--gamma", "0.8"]),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for relative, args in cases:
            labels = []
            original_plot = Axes.plot

            def record(axis, *a, **kwargs):
                labels.append(kwargs.get("label", ""))
                return original_plot(axis, *a, **kwargs)

            with experiment(relative) as module, patch.object(Axes, "plot", record):
                call_main(module, ["--N", "40", "--d", "3", *args,
                                   "--out", str(Path(directory) / Path(relative).parent)])
            if relative.startswith("decaying/"):
                assert "SGDM-0.8" in labels and "Ave-SGDM-0.8" in labels, labels
                assert not any("0.96" in label for label in labels), labels


def test_mnist_cli_plots_requested_momenta():
    """Check real plotting with mock losses; no neural-network training is run."""
    from matplotlib.axes import Axes
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    requested = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.8]
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": torch}):
        for relative in ("mnist/mnist_experiment.py", "mnist_tanh/mnist_tanh_experiment.py"):
            with experiment(relative) as module:
                module.load_mnist = lambda *a, **k: (np.zeros((60000, 1)), np.zeros(60000))
                trained = []

                def losses(seed, gamma, *a, **kwargs):
                    trained.append(gamma)
                    values = np.full(8, 0.3)
                    return ((values, {2: values.copy(), 4: values.copy()})
                            if relative.startswith("mnist_tanh/") else values)

                module.run_one = losses
                for custom in (False, True):
                    labels, trained[:] = [], []
                    original_plot = Axes.plot

                    def record(axis, *a, **kwargs):
                        label = kwargs.get("label", "")
                        labels.append(label)
                        assert len(a[0]) > 0, label
                        return original_plot(axis, *a, **kwargs)

                    args = ["--epochs", "1", "--max_iter", "8", "--seeds", "1", "--out", directory]
                    if relative.startswith("mnist_tanh/"):
                        args += ["--n0_list", "2", "4"]
                    if custom:
                        args += ["--gamma", *map(str, requested)]
                    with patch.object(Axes, "plot", record):
                        call_main(module, args)
                    if custom:
                        assert set(trained) == {0.0, *requested}, trained
                        assert all(f"SGDM-{gamma}" in labels for gamma in requested), labels
                    else:
                        assert "SGDM-0.3" in labels and "SGDM-0.99" in labels
                        assert ("SGDM-0.1" in labels) == relative.startswith("mnist/"), labels


def test_invalid_cli_settings_fail_before_work():
    cases = [
        ("quadratic/quadratic_experiment.py", ["--num_seed", "0"]),
        ("n0_sensitivity/n0_sensitivity_experiment.py", ["--batch_frac", "0"]),
        ("diff_lmu/diff_lmu_experiment.py", ["--n0", "-1"]),
        ("decaying/decaying_experiment.py", ["--n0_list", "0"]),
        ("clt/clt_experiment.py", ["--figure", "12", "--num_seed_cov", "0"]),
        ("sensitivity/sensitivity_experiment.py", ["--n0", "-1"]),
        ("logistic/logistic_experiment.py", ["--ave_len", "1"]),
        ("mnist/mnist_experiment.py", ["--epochs", "0"]),
        ("mnist_tanh/mnist_tanh_experiment.py", ["--batch_size", "0"]),
        ("mnist_tanh/mnist_tanh_experiment.py", ["--max_iter", "0"]),
        ("mnist_tanh/mnist_tanh_experiment.py", ["--n0_list", "1", "1"]),
        ("cifar/cifar_experiment.py", ["--mode", "train", "--epochs", "0"]),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for relative, args in cases:
            with experiment(relative) as module, contextlib.redirect_stderr(io.StringIO()):
                try:
                    call_main(module, [*args, "--out", directory])
                except SystemExit as error:
                    assert error.code == 2, relative
                else:
                    raise AssertionError(f"{relative} accepted {args}")


def test_failed_downloads_do_not_poison_dataset_cache():
    import pickle
    import struct
    import tarfile
    import urllib.request

    def interrupt(url, destination):
        Path(destination).write_bytes(b"incomplete transfer")
        raise OSError("interrupted")

    def mnist_file(url, destination):
        if "images" in url:
            data = struct.pack(">BBBBIII", 0, 0, 8, 3, 1, 28, 28) + bytes(784)
        else:
            data = struct.pack(">BBBBI", 0, 0, 8, 1, 1) + bytes([3])
        Path(destination).write_bytes(data)

    def cifar_file(url, destination):
        with tarfile.open(destination, "w:gz") as archive:
            for i in range(1, 6):
                data = pickle.dumps({b"data": np.zeros((1, 3072), dtype=np.uint8), b"labels": [i]})
                entry = tarfile.TarInfo(f"cifar-10-batches-py/data_batch_{i}")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))

    with tempfile.TemporaryDirectory() as directory:
        for relative, loader, valid in (
                ("mnist/mnist_experiment.py", "load_mnist", mnist_file),
                ("cifar/cifar_experiment.py", "load_cifar10", cifar_file)):
            folder = Path(directory) / Path(relative).parent
            with experiment(relative) as module:
                with patch.object(urllib.request, "urlretrieve", interrupt), \
                        contextlib.redirect_stdout(io.StringIO()):
                    try:
                        getattr(module, loader)(str(folder), download=True)
                    except (FileNotFoundError, RuntimeError):
                        pass
                    else:
                        raise AssertionError("interrupted download was accepted")
                assert not any(p.is_file() for p in folder.rglob("*")), list(folder.rglob("*"))
                with patch.object(urllib.request, "urlretrieve", valid), \
                        contextlib.redirect_stdout(io.StringIO()):
                    X, y = getattr(module, loader)(str(folder), download=True)
                assert len(X) == len(y) > 0


def test_release_archive_is_portable_and_contains_tests():
    import tarfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for folder in ("src", "scripts", "docs", "tests", "data/cifar_logs", "results/theory"):
            (root / folder).mkdir(parents=True)
        for name in ("README.md", "LICENSE", "AUTHORS", "requirements.txt", ".gitignore",
                     "src/sgdm_core.py", "scripts/run_all.sh", "tests/test_conventions.py",
                     "tests/test_runtime.py", "data/README.md", "data/cifar_logs/lr0.5_gamma0.9.csv",
                     "results/theory/fig_lambda.png", "results/theory/raw.npz"):
            (root / name).write_bytes(b"fixture")
        shutil.copy2(ROOT / "scripts/make_tarball.sh", root / "scripts/make_tarball.sh")
        for raw in ("1", "0"):
            destination = root / f"release-{raw}.tar.gz"
            result = subprocess.run(["bash", str(root / "scripts/make_tarball.sh")],
                env=dict(os.environ, OUT=str(destination), RAW=raw),
                capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            with tarfile.open(destination) as archive:
                names = archive.getnames()
            assert "tests/test_conventions.py" in names
            assert "tests/test_runtime.py" in names
            assert ("results/theory/raw.npz" in names) == (raw == "1")
        assert not (root / "results/RAW_ARRAYS_EXCLUDED.md").exists()
