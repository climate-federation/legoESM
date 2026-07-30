"""DCMIP-2025 small-planet rotation must agree across every grid.

The DCMIP-2025 small-planet convention scales the rotation rate with the
radius reduction factor X so the Rossby number is preserved: radius R/X,
Omega*X.  TC2 (mountain-triggered meso-scale flow) is ROTATING per the
official case definition; TC3 (squall line) specifies NO Coriolis.

Two real defects motivated these tests (2026-07-30):

1. ``SpectralCompressibleEulerModel.__init__`` scaled ``grid.f`` by the
   small-earth factor UNCONDITIONALLY, so the spectral TC3 arm ran at
   f*60 while its cube and MPAS siblings ran at f=0 — a cross-grid
   Coriolis mismatch that invalidated any spectral-vs-cube comparison of
   TC3.  (An initialiser-local ``small_grid`` also omitted ``omega``, but
   that grid is discarded: the model re-scales its own.  Testing the
   initialiser would have passed while the model stayed wrong, so these
   tests assert on the grid the MODEL holds.)
2. TC2 was briefly defaulted to non-rotating by false analogy with FV3's
   HIWPP Schaer mountain-wave cases (tools/test_cases.F90:3079), which
   are a DIFFERENT test from DCMIP TC2.

Each test below fails if either defect returns.
"""
from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (
    TC2_PARAMS, TC3_PARAMS,
)
from legoesm.grids.cubed_sphere import (
    apply_small_earth_scaling, create_cubed_sphere,
)


def _max_f(grid):
    """Peak |f| carried by the grid — the quantity the dycore consumes.

    Asserting on a stored ``omega`` attribute would be asserting on
    bookkeeping; ``f = 2 Omega sin(lat)`` is what reaches the momentum
    equation, and it scales linearly with Omega.
    """
    import jax.numpy as jnp
    return float(jnp.max(jnp.abs(grid.f)))


def test_tc2_is_rotating_and_tc3_is_not():
    """Case definitions, not incidental defaults."""
    # Require the key EXPLICITLY: ``.get(..., True)`` would pass if the
    # entry were deleted, silently reverting the contract (codex r4 P2).
    assert "rotating" in TC2_PARAMS and TC2_PARAMS["rotating"] is True, (
        "DCMIP-2025 TC2 is a ROTATING small planet (radius/X, Omega*X). "
        "The FV3 HIWPP Schaer cases that zero f0/fC are a different test.")
    assert "rotating" in TC3_PARAMS and TC3_PARAMS["rotating"] is False, (
        "DCMIP-2025 TC3 (squall line) specifies no Coriolis.")


def _rel_tol(grid):
    """Tolerance matched to the grid's own dtype.

    ``f`` is cast to the grid dtype on construction, so a float32 grid
    cannot support an fp64-tight comparison (this test first failed for
    exactly that reason, not for a physics reason).
    """
    import numpy as _np
    return 1e-12 if _np.dtype(grid.f.dtype).itemsize >= 8 else 1e-6


def test_small_earth_scaling_applies_omega_times_x():
    grid = create_cubed_sphere(8)
    scaled = apply_small_earth_scaling(grid, 20.0)
    assert _max_f(grid) > 0.0                      # control: probe can fail
    assert _max_f(scaled) == pytest.approx(20.0 * _max_f(grid),
                                           rel=_rel_tol(grid))
    assert float(scaled.radius) == pytest.approx(constants.R_earth / 20.0)


def test_small_earth_scaling_non_rotating_zeroes_f():
    grid = create_cubed_sphere(8)
    assert _max_f(grid) > 0.0                      # control: probe can fail
    scaled = apply_small_earth_scaling(grid, 60.0, rotating=False)
    assert _max_f(scaled) == 0.0
    assert float(scaled.radius) == pytest.approx(constants.R_earth / 60.0)


def test_no_op_factor_with_non_rotating_raises_rather_than_rebuilding():
    """factor=1 + rotating=False cannot be served without a rebuild that
    would drop dtype/duogrid provenance — it must raise, not silently
    return a grid with the wrong halo topology."""
    grid = create_cubed_sphere(8)
    assert apply_small_earth_scaling(grid, 1.0) is grid
    with pytest.raises(ValueError, match="rotating=False"):
        apply_small_earth_scaling(grid, 1.0, rotating=False)


@pytest.mark.parametrize("rotating,factor,expect_ratio", [
    (True, 20.0, 20.0),     # TC2 convention: f scales with the radius reduction
    (False, 60.0, 0.0),     # TC3 convention: no Coriolis at all
])
def test_spectral_model_honours_case_rotation(rotating, factor, expect_ratio):
    """The SPECTRAL MODEL must apply the case's rotation, not just its
    initialiser.

    codex r3 P1: the initialiser builds a local ``small_grid`` whose ``f``
    the running model never consumes — the model re-scales ``grid.f`` itself
    in ``__init__``.  A test that spied on the initialiser would pass while
    the model ran at the wrong Coriolis, which is exactly the defect that
    left the spectral TC3 arm at f*60 against f=0 on cube and MPAS.  So
    assert on the grid the MODEL ends up holding.
    """
    import jax.numpy as jnp
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
    )
    from legoesm.grids.vertical import (
        create_height_coordinate, compute_terrain_metric,
    )

    base = create_gaussian_grid(21)
    f_base = float(jnp.max(jnp.abs(base.f)))
    assert f_base > 0.0                            # control: probe can fail

    hcoord = create_height_coordinate(8, 20000.0)
    tmetric = compute_terrain_metric(
        jnp.zeros((base.n_lat, base.n_lon)), hcoord)
    model = SpectralCompressibleEulerModel(
        base, hcoord, tmetric,
        SpectralNHConfig(small_earth_factor=factor, rotating=rotating),
    )
    f_model = float(jnp.max(jnp.abs(model.grid.f)))
    assert f_model == pytest.approx(expect_ratio * f_base, abs=1e-12,
                                    rel=1e-10)


def test_spectral_matrix_branch_reads_case_rotation():
    """The matrix's spectral branch must take ``rotating`` from the SAME
    case parameter dicts the cube and MPAS arms read, so a change to a case
    definition cannot silently desynchronise one grid."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    text = (root / "scripts" / "matrix"
            / "run_atmosphere_test_matrix.py").read_text()
    assert "rotating=_rotating," in text, (
        "spectral SpectralNHConfig no longer receives the case rotation")
    assert re.search(r'_rotating = bool\(_TC3_P\.get\("rotating"', text), (
        "spectral tc3 branch no longer reads TC3_PARAMS['rotating']")


@pytest.mark.parametrize("case,factor,rotating", [("tc2", 20.0, True),
                                                  ("tc3", 60.0, False)])
def test_mpas_initialiser_honours_case_rotation(case, factor, rotating,
                                                monkeypatch):
    """The MPAS arms must build their mesh from the case's rotation too.

    Their initialisers previously hard-coded omega, so an explicit
    override was silently ignored (codex r3 P2).  Capture the omega
    actually handed to ``create_voronoi_mesh`` -- and assert BOTH the
    default and an override, so restoring the hard-coded value goes red
    either way.
    """
    import importlib
    mod = importlib.import_module(
        "tests.atmosphere.nonhydrostatic.test_cases.dcmip2025."
        f"test_case_{'2' if case == 'tc2' else '3'}_mpas")

    seen: dict = {}
    real = mod.create_voronoi_mesh

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "create_voronoi_mesh", spy)

    from legoesm.grids.voronoi import create_voronoi_mesh as _cvm
    base = _cvm(2)
    init = getattr(mod, f"dcmip25_{case}_init_mpas")

    # default
    try:
        init(base, n_levels=10)
    except Exception:
        pass
    assert "omega" in seen, f"{case} MPAS mesh built without an explicit omega"
    expected = constants.Omega * factor if rotating else 0.0
    assert seen["omega"] == pytest.approx(expected), (
        f"{case} MPAS default rotation disagrees with the cube/spectral arms")

    # explicit override must be HONOURED, not hard-coded past
    seen.clear()
    try:
        init(base, n_levels=10, params={"rotating": not rotating})
    except Exception:
        pass
    flipped = 0.0 if rotating else constants.Omega * factor
    assert seen.get("omega") == pytest.approx(flipped), (
        f"{case} MPAS initialiser ignored an explicit rotating override "
        "(omega is hard-coded)")
