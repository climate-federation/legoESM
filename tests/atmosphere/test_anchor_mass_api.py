"""Regression test for the iter-18..21 anchor-mass API.

Every anchored dycore model class must expose the four-method
interface:

- ``compute_mass(state)`` / ``compute_dry_mass(state)`` — fp64 snapshot
- ``reset_target_mass()`` — clear cached target
- ``set_target_mass(target)`` — explicit anchor (e.g. checkpoint restart)

This test asserts the API is present on every implementation and that
the methods round-trip correctly (set → use → reset → re-snapshot).
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import math
import pytest


# ---------------------------------------------------------------------------
# API presence checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "import_path",
    [
        # SW
        "legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid."
        "CGridLatLonShallowWaterModel",
        "legoesm.atmosphere.dynamics.shallow_water_mpas."
        "MPASShallowWaterModel",
        "legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid."
        "FV3EdgeShallowWaterModel",
        "legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid."
        "FV3FBShallowWaterModel",
        # PE
        "legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid."
        "CGridLatLonPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.primitive_eq_mpas."
        "MPASPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.primitive_eq_cdgrid."
        "CDGridPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.spectral_pe."
        "SpectralPrimitiveEquationModel",
        # NH
        "legoesm.atmosphere.dynamics.compressible_euler_mpas."
        "MPASCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.compressible_euler_cdgrid."
        "CDGridCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.spectral_nh."
        "SpectralCompressibleEulerModel",
    ],
)
def test_anchor_api_methods_present(import_path):
    """Every anchored model must expose reset_target_mass +
    set_target_mass + one of compute_mass / compute_dry_mass."""
    module_path, cls_name = import_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    assert hasattr(cls, "reset_target_mass"), (
        f"{cls_name} missing reset_target_mass")
    assert hasattr(cls, "set_target_mass"), (
        f"{cls_name} missing set_target_mass")
    has_mass = hasattr(cls, "compute_mass")
    has_dry = hasattr(cls, "compute_dry_mass")
    assert has_mass or has_dry, (
        f"{cls_name} missing compute_mass / compute_dry_mass")


# ---------------------------------------------------------------------------
# Lifecycle round-trip: set → step → reset → step ends with the fresh anchor
# ---------------------------------------------------------------------------

def test_anchor_lifecycle_latlon_pe():
    """``set_target_mass`` overrides the lazy snapshot; ``reset_target_mass``
    re-arms it.  Uses lat-lon PE as the canonical SW/PE/NH twin (iter-2)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon

    grid = create_latlon_grid(36, 72)
    sigma = create_sigma_coordinate(8)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(200.0, 0.5 * dx_pole / 300.0)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    state = hydrostatic_to_cgrid(
        held_suarez_init_latlon(grid, sigma), grid,
    )

    # Pre-step: target unset.
    assert model._target_mass is None

    # Manual set → step → target stays the user-supplied value.
    custom_target = jnp.asarray(1.234e19, dtype=jnp.float64)
    model.set_target_mass(custom_target)
    assert model._target_mass is custom_target
    _ = model.step(state, dt)
    # The Python wrapper still checks ``_target_mass is None`` before
    # snapshotting, so an explicit set is preserved across step().
    assert model._target_mass is custom_target

    # reset_target_mass clears it, and the next step re-snapshots from state.
    model.reset_target_mass()
    assert model._target_mass is None
    state2 = model.step(state, dt)
    # New snapshot is fp64 scalar, close to compute_mass(state).
    expected = float(model.compute_mass(state))
    actual = float(model._target_mass)
    assert abs(actual - expected) / max(abs(expected), 1.0) < 1e-12

    # Run the model with a fresh state — the now-cached _target_mass
    # is the iter-snapshot, NOT a stale custom target.
    _ = state2  # silence linter
