"""Regression test for the iter-18..21 anchor-mass API + matrix-runner gates.

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


def test_spectral_sw_compute_mass_present():
    """iter-35: spectral SW has no anchor (baseline drift already
    bit-clean) but exposes ``compute_mass`` for API parity with the
    cube / lat-lon / MPAS SW twins."""
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralShallowWaterModel,
    )
    assert hasattr(SpectralShallowWaterModel, "compute_mass"), (
        "SpectralShallowWaterModel.compute_mass missing — iter-35 "
        "added it for uniform per-grid public API"
    )


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


def test_anchor_lazy_snapshot_is_sticky():
    """iter-31: document that the lazy snapshot fires ONCE.

    A second initial state will see the first state's cached
    ``_target_mass`` unless the caller invokes ``reset_target_mass()``
    or ``set_target_mass(...)`` first.  This is the right behaviour for
    a long Python-loop integration (the matrix runner's pattern) but
    the wrong behaviour for AD / multi-experiment callers that re-anchor
    on every gradient step — the documented workaround is to call
    ``reset_target_mass()`` between gradient evaluations.

    Test asserts: ``state.p_s * 1.01`` (1% more mass) on the second
    ``step()`` call still reports the FIRST state's mass as the
    anchored target.  After ``reset_target_mass()`` the next step
    correctly re-snapshots.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(8)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    state = held_suarez_init(grid, sigma)

    _ = model.step(state, 600.0)
    mass_after_first = float(model._target_mass)

    # 1%-perturbed initial state — should anchor to a different target.
    state2 = state._replace(
        p_s=state.p_s.replace(data=state.p_s.data * 1.01),
    )
    _ = model.step(state2, 600.0)
    mass_after_second = float(model._target_mass)
    # Stale: first state's mass survives because lazy snapshot doesn't refire.
    assert mass_after_first == mass_after_second, (
        "Expected sticky lazy snapshot: target should NOT re-fire when "
        "_target_mass is already set."
    )

    # Caller-side fix: reset and re-step.
    model.reset_target_mass()
    _ = model.step(state2, 600.0)
    mass_after_reset = float(model._target_mass)
    # Now reflects the 1% extra mass.
    rel = (mass_after_reset - mass_after_first) / mass_after_first
    assert 0.0099 < rel < 0.011, (
        f"After reset_target_mass, anchor should track new state: "
        f"expected rel ~1e-2, got {rel:.4e}"
    )


def test_matrix_runner_mass_drift_constants_sane():
    """iter-32: pin the iter-30 matrix-runner mass-drift gate constants.

    A regression to ``1e-2`` (the iter-117/118 pre-iter-23 default) on
    any of the gates would silently let a 100x conservation regression
    slip through CI; the explicit lower-bound assertions here flag that
    sort of change.
    """
    import sys
    import importlib
    sys.path.insert(0, "scripts")
    try:
        runner = importlib.import_module("run_atmosphere_test_matrix")
    finally:
        sys.path.pop(0)

    assert hasattr(runner, "_DYCORE_MASS_DRIFT_TOL")
    assert hasattr(runner, "_DYCORE_MASS_DRIFT_TOL_CB")

    # SW W5/W6, hydro HS/baroclinic/AMIP, NH TC1/TC2a/TC3 ceiling.
    tol = runner._DYCORE_MASS_DRIFT_TOL
    assert tol <= 1e-6, (
        f"_DYCORE_MASS_DRIFT_TOL relaxed to {tol:.0e} — should stay "
        f"<= 1e-6 per iter-23..28"
    )
    assert tol >= 1e-14, (
        f"_DYCORE_MASS_DRIFT_TOL tightened to {tol:.0e} — fp64 floor "
        f"limits the realistic ceiling; reconsider if intentional"
    )

    # cosine_bell gate (intentional lat-lon raw-FV benchmark) at 1e-4
    # per iter-29.
    cb_tol = runner._DYCORE_MASS_DRIFT_TOL_CB
    assert cb_tol <= 1e-4, (
        f"_DYCORE_MASS_DRIFT_TOL_CB relaxed to {cb_tol:.0e} — should "
        f"stay <= 1e-4 per iter-29"
    )
    assert cb_tol >= tol, (
        f"_DYCORE_MASS_DRIFT_TOL_CB ({cb_tol:.0e}) should be >= "
        f"_DYCORE_MASS_DRIFT_TOL ({tol:.0e}) since lat-lon cosine_bell "
        f"intentionally measures raw-FV transport drift (~1.49e-05)"
    )


def test_anchored_step_scan_compat_under_fp64_policy():
    """iter-36: ``integrate_scan`` works with the anchored fixer when the
    precision policy is fp64 (so post-fix state stays type-stable).

    iter-18 documented that under the default fp32 storage policy
    ``cast_pytree(allow_downcast=False)`` keeps post-fix p_s in fp64
    while scan_fn carry input is fp32 — scan errors with a dtype
    mismatch.  Setting ``PrecisionPolicy.fp64()`` removes that mismatch
    (state is fp64 everywhere) and ``integrate_scan`` succeeds with
    machine-precision drift, the same as a Python-loop integration.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(8)
        cfg = CDGridPrimitiveEquationConfig(
            use_conservation_fixer=True,
            fix_mass=True,
            anchor_mass_to_initial=True,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
        state = held_suarez_init(grid, sigma)

        # Lazy snapshot fires inside scan trace; the fp64 carry keeps
        # the trace type-stable so the scan body emits a valid program.
        final, _traj = model.integrate_scan(state, 5, 600.0)

        def _mass(s):
            return float(jnp.sum(
                s.p_s.data.astype(jnp.float64)
                * grid.area.astype(jnp.float64),
            ))

        rel = abs(_mass(final) - _mass(state)) / max(abs(_mass(state)), 1.0)
        assert rel < 1e-12, (
            f"integrate_scan under fp64 policy drift {rel:.2e} > 1e-12 "
            "— anchored fixer should be bit-clean here"
        )
    finally:
        set_policy(saved)
