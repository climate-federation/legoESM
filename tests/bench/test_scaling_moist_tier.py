"""Direct tests for the grid-agnostic 'moist' (Kessler) scaling tier.

These cover the wiring added so the GPU scaling driver can run a moist
benchmark on EVERY grid — and, crucially, on the icosahedral Voronoi MPI
path — so a fair cross-grid many-GPU scaling comparison uses the same column
closure on each grid (rather than swapping dry-vs-moist physics between
grids, which confounds the comm/compute ratio).

The single-device run_benchmark smoke runs on CPU (Metal backend is broken
for this repo) at x64 and tiny resolution.  The Voronoi-MPI moist path is
exercised separately under ``mpirun`` (see scripts/tmp probes / the CPU MPI
scaling driver); it cannot run inside a single-process pytest.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "run_levante_gpu_scaling.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_levante_gpu_scaling", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module's @dataclass decorators resolve
    # cls.__module__ via sys.modules during class creation.
    sys.modules["run_levante_gpu_scaling"] = mod
    spec.loader.exec_module(mod)
    return mod


rl = _load()

_ALL_GRIDS = ("spectral", "latlon", "icosahedral", "cubed-sphere")


def test_moist_in_physics_choices():
    assert "moist" in rl.PHYSICS_CHOICES


@pytest.mark.parametrize("grid", _ALL_GRIDS)
def test_moist_supported_on_every_grid(grid):
    # The Kessler column closure is grid-agnostic, so 'moist' is a valid tier
    # for every grid (unlike the radiative gray_sbm/rrtmg_full tiers).
    assert "moist" in rl._SUPPORTED_PHYSICS[grid]


@pytest.mark.parametrize("grid", _ALL_GRIDS)
def test_build_moist_physics_fn_per_grid(grid):
    fn = rl._build_moist_physics_fn(grid, 600.0)
    assert callable(fn)


def test_build_moist_physics_fn_rejects_unknown_grid():
    # Dispatch hardening: a silent None would benchmark dycore-only under a
    # 'moist' label.  Must raise (CLAUDE.md dispatch rule).
    with pytest.raises(ValueError):
        rl._build_moist_physics_fn("bogus", 600.0)


def test_validate_physics_allows_icosahedral_moist():
    # No exception.
    rl._validate_physics("icosahedral", "moist")
    rl._validate_physics("spectral", "moist")


def test_validate_physics_still_blocks_icosahedral_radiative():
    # The radiative tiers stay cubed-sphere / lat-lon only (segment path).
    with pytest.raises(ValueError):
        rl._validate_physics("icosahedral", "gray_sbm")
    with pytest.raises(ValueError):
        rl._validate_physics("spectral", "rrtmg_full")


@pytest.mark.parametrize("grid", ["icosahedral", "cubed-sphere"])
def test_single_device_moist_benchmark_runs(grid):
    """End-to-end: a moist benchmark builds the IC tracers, threads the
    Kessler forcing into the step, and produces a finite positive timing."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    r = rl.run_benchmark(
        n_grid=2,
        n_levels=5,
        n_gpus=1,
        precision="float64",
        mode="single",
        n_warmup=1,
        n_timing=1,
        dt=300.0,
        grid_type=grid,
        physics_level="moist",
    )
    assert r.physics_level == "moist"
    assert r.time_per_step_ms == r.time_per_step_ms  # not NaN
    assert r.time_per_step_ms > 0.0
    assert r.total_cells > 0
