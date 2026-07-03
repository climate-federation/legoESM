"""Direct test for scripts/cluster/scaling_derecho/mc_nccl_probe.py.

Runs the probe single-process (size 1 — no distributed init) on local
devices: every stage (device visibility, psum, ppermute ring, latency loop)
must pass and exit 0. The multi-process path is the Derecho canary's job
(mc_nccl_canary.sh); the probe's arg guards are covered here too.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cluster"
           / "scaling_derecho" / "mc_nccl_probe.py")


def _load():
    spec = importlib.util.spec_from_file_location("mc_nccl_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_single_process_all_stages_pass(monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(sys, "argv", ["probe", "--size", "1", "--iters", "3"])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "stage3 psum" in out and "OK=True" in out
    assert "stage4 ppermute ring OK=True" in out
    assert "CANARY PASS" in out


def test_probe_requires_coordinator_for_multiprocess(monkeypatch):
    mod = _load()
    monkeypatch.delenv("LEGOESM_JAX_COORDINATOR", raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "--size", "2", "--rank", "0"])
    assert mod.main() == 2          # refuses BEFORE any distributed init


def test_probe_rejects_bad_iters(monkeypatch):
    mod = _load()
    monkeypatch.setattr(sys, "argv", ["probe", "--size", "1", "--iters", "0"])
    assert mod.main() == 2


def test_probe_never_imports_mpi4jax():
    """Mixed-stack guard: the probe must not IMPORT mpi4jax (the documented
    jax.distributed + mpi4jax deadlock hazard). Docstring mentions are fine."""
    src = _SCRIPT.read_text()
    assert "import mpi4jax" not in src
    assert "from mpi4jax" not in src
    assert "import mpi4py" not in src
    assert "from mpi4py" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
