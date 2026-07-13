"""Direct test for scripts/bench/bench_ocean_latlon_spmd_scaling.py.

Exercises the builder + the nd=1 single-device main() end-to-end on a tiny
grid (JSONL record shape) and the argument guards. The multi-device timing
path reuses make_sharded_ocean_step, whose correctness gate is
tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
cluster-gated (see the script docstring).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "bench" / "bench_ocean_latlon_spmd_scaling.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_ocean_latlon_spmd_scaling", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_model_and_state_tiny():
    mod = _load()
    model, state = mod.build_model_and_state(8, 16, 3)
    assert state.T.data.shape == (8, 16, 3)
    assert state.v.data.shape == (9, 16, 3)          # staggered v
    assert bool(np.isfinite(np.asarray(state.T.data)).all())
    # Perturbed, not the trivial rest state.
    assert float(np.max(np.abs(np.asarray(state.u.data)))) > 0.0


def test_main_single_device_writes_record(tmp_path, monkeypatch):
    mod = _load()
    out = tmp_path / "rec.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--n-lat", "8", "--n-lon", "16", "--nlev", "3",
        "--n-devices", "1", "--steps", "2", "--warmup", "1",
        "--dt", "300.0", "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "ocean"
    assert rec["n_devices"] == 1 and rec["n_lat"] == 8
    assert rec["multicontroller"] is False
    # Measurement-contract fields (fused scan blocks + separate dispatch
    # probe): per-block times, the fused headline, and the individually-
    # synced latency — the retired per_step_ms key must NOT come back.
    assert len(rec["block_ms"]) == 2            # default --blocks 2
    assert rec["block_steps"] == 2              # --steps = per-block length
    assert rec["fused_step_ms"] > 0.0
    assert rec["step_latency_ms"] > 0.0
    assert rec["steady_median_ms"] > 0.0
    assert "per_step_ms" not in rec


def test_main_rejects_bad_timing_window(monkeypatch):
    """--steps validated BEFORE any model/device work (a bad block length
    would otherwise surface only after the expensive build).  --warmup is
    CLI-compat-only and IGNORED: a warmup >= steps combination that the
    retired per-step slicing rejected must now be accepted (a one-step
    parity run with the default --warmup=2 was spuriously refused)."""
    mod = _load()
    monkeypatch.setattr(sys, "argv", [
        "bench", "--n-lat", "8", "--n-lon", "16", "--nlev", "3",
        "--n-devices", "1", "--steps", "0"])
    with pytest.raises(SystemExit, match="--steps"):
        mod.main()


def test_main_rejects_too_many_devices(monkeypatch):
    mod = _load()
    monkeypatch.setattr(sys, "argv", [
        "bench", "--n-lat", "8", "--n-lon", "16", "--nlev", "3",
        "--n-devices", "4096"])
    with pytest.raises(SystemExit, match="devices"):
        mod.main()


def test_main_rejects_indivisible_nlat(monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip("needs >=2 devices for an indivisible split")
    mod = _load()
    monkeypatch.setattr(sys, "argv", [
        "bench", "--n-lat", "9", "--n-lon", "16", "--nlev", "3",
        "--n-devices", "2"])
    with pytest.raises(SystemExit, match="not divisible"):
        mod.main()
