"""Regression test for the plane CRM production driver argparse defaults.

iter-59 found ``scripts/run_rce_mpi_long.py`` argparse defaults still
shipping with ``--n-acoustic-substeps`` defaulting to ``24`` (set
for iter-1's ``dt=1.0 s`` config) while the driver's ``--dt`` default
is ``5.0 s`` (iter-12/14 production). The combination produces an
over-stable but 2x-slower substep vs iter-14's measured production
contract (N=12 at dt=5).

This test parses the driver source via ``ast`` to find every
``p.add_argument`` call and asserts the production-contract defaults
match iter-14 + iter-38 measurements. Catches future silent reverts
in < 1 s before any of the slow nightly tests would.

Mirrors the iter-58 pattern (``test_run_rce_30day_wrapper_defaults.py``)
which locks the wrapper-script env-var defaults.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER = REPO_ROOT / "scripts" / "run_rce_mpi_long.py"


def _parse_argparse_defaults(source: str) -> dict[str, object]:
    """Walk the AST for every ``p.add_argument(...)`` call and return
    a dict mapping CLI flag → default value literal. Returns only
    flags with literal defaults (skips ones using ``default=None``
    or non-constant defaults).
    """
    tree = ast.parse(source)
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant):
            continue
        flag = first.value
        if not (isinstance(flag, str) and flag.startswith("--")):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                out[flag] = kw.value.value
                break
    return out


@pytest.fixture(scope="module")
def driver_defaults() -> dict[str, object]:
    assert DRIVER.exists(), f"driver script missing: {DRIVER}"
    return _parse_argparse_defaults(DRIVER.read_text())


def test_dt_default_matches_iter14(driver_defaults):
    """iter-14 measured 132x132 1-sim-hour PASS at dt=5.0 s.
    Driver default must match."""
    assert driver_defaults["--dt"] == 5.0


def test_n_acoustic_substeps_default_matches_iter14(driver_defaults):
    """iter-14 + iter-38 production use N_ACOUSTIC=12 at dt=5.0 s.
    iter-59 refreshed the driver default from the stale 24
    (iter-1 dt=1.0 s value) to 12."""
    assert driver_defaults["--n-acoustic-substeps"] == 12, (
        f"--n-acoustic-substeps default = "
        f"{driver_defaults['--n-acoustic-substeps']!r}, expected 12 "
        f"(iter-14 + iter-38 production contract). With dt=5.0 + "
        f"N=24 the acoustic CFL ratio halves — over-stable but "
        f"2x-cost substep, silently differs from the iter-14 "
        f"measurement."
    )


def test_hyperdiff_default_matches_iter14(driver_defaults):
    """iter-15 + iter-38 production use hyperdiff=5e6."""
    assert driver_defaults["--hyperdiff"] == 5.0e6


def test_smag_cs_default_matches_iter14(driver_defaults):
    """iter-3 + iter-38 production use Smag c_s=0.2."""
    assert driver_defaults["--smag-cs"] == 0.2


def test_no_bubble_default(driver_defaults):
    """F7/F10 contract: production runs on clean Wing IC (no bubble)."""
    assert driver_defaults["--bubble-theta-pert"] == 0.0


def test_no_qv_noise_default(driver_defaults):
    """F7/F10 contract: production runs without qv noise."""
    assert driver_defaults["--qv-noise-amp"] == 0.0


def test_nx_ny_production_defaults(driver_defaults):
    """iter-12+ production grid is 132x132."""
    assert driver_defaults["--nx"] == 132
    assert driver_defaults["--ny"] == 132


def test_nlev_production_default(driver_defaults):
    """iter-12 + iter-14 production uses nlev=30."""
    assert driver_defaults["--nlev"] == 30


# iter-61 Codex MEDIUM coverage-gap fix: assert the production-anchored
# defaults that iter-59 missed. These are values that affect either the
# stability envelope (sponge_*, advection, semi-implicit) or the
# observability (log_every_steps, rad_call_interval_s) of the iter-14
# production measurement. Each is grounded in CRM_implementation.md.


def test_dx_production_default(driver_defaults):
    """iter-12 production grid spacing: dx=2 km."""
    assert driver_defaults["--dx"] == 2000.0


def test_H_production_default(driver_defaults):
    """iter-12 production domain height: H=33 km."""
    assert driver_defaults["--H"] == 33_000.0


def test_dz_sfc_production_default(driver_defaults):
    """iter-12 surface dz=100 m."""
    assert driver_defaults["--dz-sfc"] == 100.0


def test_vertical_grid_production_default(driver_defaults):
    """iter-14/38 production: --vertical-grid uniform (dz=H/nlev ≈ 1100m
    at nlev=30, H=33km). Stretched grid is opt-in via CLI."""
    assert driver_defaults["--vertical-grid"] == "uniform"


def test_n_physics_substeps_production_default(driver_defaults):
    """iter-14/38 production: 10 physics substeps per dycore step."""
    assert driver_defaults["--n-physics-substeps"] == 10


def test_rad_call_interval_default(driver_defaults):
    """Driver default rad cadence = 600 s. Production runs without
    --no-radiation. iter-39 added --no-radiation flag with default
    False (production unaffected)."""
    assert driver_defaults["--rad-call-interval-s"] == 600.0
    assert driver_defaults["--no-radiation"] is False


def test_use_dd_default(driver_defaults):
    """iter-5 added --use-dd; default False keeps the legacy
    rank-0-broadcast F8-stable path as the canonical entry."""
    assert driver_defaults["--use-dd"] is False


def test_advection_production_default(driver_defaults):
    """iter-7 production: upwind1 (WENO5 is opt-in via CLI)."""
    assert driver_defaults["--advection"] == "upwind1"


def test_sponge_production_defaults(driver_defaults):
    """iter-1 sponge layer config."""
    assert driver_defaults["--sponge-coeff"] == 0.05
    assert driver_defaults["--sponge-width"] == 10_000.0


def test_implicit_buoyancy_default(driver_defaults):
    """F2/F10: KW78 substep is inert at production dt; default off."""
    assert driver_defaults["--implicit-buoyancy"] is False


def test_qv_noise_seed_default(driver_defaults):
    """Deterministic seed for qv-noise (when amp > 0)."""
    assert driver_defaults["--qv-noise-seed"] == 0
