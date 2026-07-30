"""DCMIP-2025 small-planet rotation must agree across every grid.

The DCMIP-2025 small-planet convention scales the rotation rate with the
radius reduction factor X so the Rossby number is preserved: radius R/X,
Omega*X.  TC2 (mountain-triggered meso-scale flow) is ROTATING per the
official case definition; TC3 (squall line) specifies NO Coriolis.

Two real defects motivated these tests (2026-07-30):

1. The spectral TC2/TC3 initialisers called ``create_gaussian_grid`` with
   ``radius=R/factor`` but WITHOUT ``omega``, so they silently kept the
   full-Earth Omega while the cube arm ran at Omega*X — a 20x (TC2) /
   60x (TC3) cross-grid confound that invalidated every spectral-vs-cube
   comparison of these cases.
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


def _omega_of(grid):
    """Rotation rate carried by a grid object, whatever it is called."""
    for attr in ("omega", "Omega", "rotation_rate"):
        if hasattr(grid, attr):
            return float(getattr(grid, attr))
    raise AssertionError(f"no rotation attribute on {type(grid).__name__}")


def test_tc2_is_rotating_and_tc3_is_not():
    """Case definitions, not incidental defaults."""
    assert TC2_PARAMS.get("rotating", True) is True, (
        "DCMIP-2025 TC2 is a ROTATING small planet (radius/X, Omega*X). "
        "The FV3 HIWPP Schaer cases that zero f0/fC are a different test.")
    assert TC3_PARAMS.get("rotating", True) is False, (
        "DCMIP-2025 TC3 (squall line) specifies no Coriolis.")


def test_small_earth_scaling_applies_omega_times_x():
    grid = create_cubed_sphere(8)
    scaled = apply_small_earth_scaling(grid, 20.0)
    assert _omega_of(scaled) == pytest.approx(constants.Omega * 20.0)
    assert float(scaled.radius) == pytest.approx(constants.R_earth / 20.0)


def test_small_earth_scaling_non_rotating_zeroes_omega():
    grid = create_cubed_sphere(8)
    scaled = apply_small_earth_scaling(grid, 60.0, rotating=False)
    assert _omega_of(scaled) == 0.0
    assert float(scaled.radius) == pytest.approx(constants.R_earth / 60.0)


def test_no_op_factor_with_non_rotating_raises_rather_than_rebuilding():
    """factor=1 + rotating=False cannot be served without a rebuild that
    would drop dtype/duogrid provenance — it must raise, not silently
    return a grid with the wrong halo topology."""
    grid = create_cubed_sphere(8)
    assert apply_small_earth_scaling(grid, 1.0) is grid
    with pytest.raises(ValueError, match="rotating=False"):
        apply_small_earth_scaling(grid, 1.0, rotating=False)


@pytest.mark.parametrize("case,factor,rotating", [("tc2", 20.0, True),
                                                  ("tc3", 60.0, False)])
def test_spectral_initialiser_passes_a_consistent_omega(case, factor,
                                                        rotating, monkeypatch):
    """The spectral arm must build its small planet with the SAME rotation
    rate as the cube arm.  Captures the omega actually handed to
    ``create_gaussian_grid`` — a source-text assertion would pass on a
    delegating wrapper while proving nothing."""
    import legoesm.atmosphere.dynamics.gcm.spectral_nh as snh

    seen: dict = {}
    real = snh.create_gaussian_grid

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(snh, "create_gaussian_grid", spy)

    from legoesm.grids.gaussian import create_gaussian_grid as _cgg
    base = _cgg(21)
    init = (snh.dcmip25_tc2_init_spectral if case == "tc2"
            else snh.dcmip25_tc3_init_spectral)
    try:
        init(base, n_levels=10)
    except Exception:
        # The initialiser may fail downstream on this tiny grid; the grid
        # construction we are asserting on has already happened.
        pass

    assert "omega" in seen, (
        f"{case} spectral initialiser built its small planet without an "
        "explicit omega — it therefore ran at full-Earth Omega while the "
        "cube arm ran at Omega*X (cross-grid confound)")
    expected = constants.Omega * factor if rotating else 0.0
    assert seen["omega"] == pytest.approx(expected)
    assert seen["radius"] == pytest.approx(constants.R_earth / factor)
