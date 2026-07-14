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
        "legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid."
        "CGridLatLonShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.shallow_water_mpas."
        "MPASShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid."
        "FV3EdgeShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid."
        "FV3FBShallowWaterModel",
        # PE
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid."
        "CGridLatLonPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas."
        "MPASPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid."
        "CDGridPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.spectral_pe."
        "SpectralPrimitiveEquationModel",
        # NH
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas."
        "MPASCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid."
        "CDGridCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.gcm.spectral_nh."
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
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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
    sys.path.insert(0, "scripts/matrix")
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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


def test_anchored_step_supports_jax_grad():
    """iter-37: ``jax.grad`` through ``step()`` with anchor enabled.

    Differentiable workflows (training, sensitivity analysis) require
    AD to flow through the conservation fixer.  The anchored target is
    snapshotted as a fp64 scalar inside ``step()``; once cached on
    ``self``, all subsequent steps inside the same trace use it as a
    closure constant.  The fixer itself is a pure ``state -> state``
    function so the autodiff chain through it is well-defined.

    Test: build a 1-step loss ``∫ T² dV`` and assert ``jax.grad`` w.r.t.
    the initial ``p_s`` produces a finite, non-zero gradient.  Catches
    AD regressions (e.g. a future fixer change that introduces a
    ``float(...)`` host callback or other non-differentiable op).
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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
        state0 = held_suarez_init(grid, sigma)

        def loss(p_s_data):
            state = state0._replace(p_s=state0.p_s.replace(data=p_s_data))
            s1 = model.step(state, 600.0)
            # Return a JAX scalar — must not call ``float(...)``
            # inside an AD-traced function.
            return jnp.sum(s1.T.data ** 2)

        # iter-31 sticky-snapshot semantics: re-anchor before each
        # gradient evaluation so the target tracks the differentiation
        # variable rather than caching a stale value.
        model.reset_target_mass()
        grad = jax.grad(loss)(state0.p_s.data)

        assert jnp.all(jnp.isfinite(grad)), (
            "jax.grad through anchored step produced non-finite values"
        )
        assert float(jnp.max(jnp.abs(grad))) > 0.0, (
            "jax.grad through anchored step is identically zero — "
            "fixer may be silently breaking the autodiff chain"
        )
        # fp64 policy keeps gradients in fp64.
        assert grad.dtype == jnp.float64, (
            f"Expected fp64 gradient under fp64 policy, got {grad.dtype}"
        )
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_multistep():
    """iter-38: extend iter-37 AD test to a 3-step chain.

    Validates the autodiff chain across multiple time steps with the
    anchored fixer in the loop.  Under a trace, ``step()`` does NOT
    cache ``_target_mass`` (iter-59 tracer-leak fix); each step
    receives its pre-step mass as a traced argument, and because the
    fixer is exact the per-step targets telescope to the initial mass
    — numerically identical to the old cached-tracer behaviour while
    keeping the chain rule clean (the fixer stays pure state→state).
    Catches regressions where multi-step gradients would diverge /
    become non-finite.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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
        state0 = held_suarez_init(grid, sigma)

        def loss(p_s_data):
            state = state0._replace(
                p_s=state0.p_s.replace(data=p_s_data))
            s = model.step(state, 600.0)
            s = model.step(s, 600.0)
            s = model.step(s, 600.0)
            return jnp.sum(s.T.data ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.p_s.data)

        assert jnp.all(jnp.isfinite(grad))
        # 3-step gradient should be larger than 1-step (chain rule
        # accumulates the upstream sensitivity through more state
        # updates) — assert it grew rather than collapsed.
        assert float(jnp.max(jnp.abs(grad))) > 1e-2, (
            f"3-step grad max |{float(jnp.max(jnp.abs(grad))):.2e}| "
            "smaller than expected — chain rule may be broken across "
            "multi-step anchored fixer"
        )
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_spectral_pe():
    """iter-39: ``jax.grad`` through anchored ``step()`` on spectral PE.

    Different code path from iter-37's cube PE test:
      * Prognostic state in spectral space (complex128 ``lnps_hat``).
      * Mass fixer reaches grid space via ``sh_synthesis`` and writes
        back to ``lnps_hat[0]`` via ``Δρ·sqrt(4π)`` (iter-3 pattern).
      * AD chain therefore traverses an SH round-trip *inside* the
        fixer in addition to the dycore tendencies.

    Catches AD regressions specific to the spectral PE fixer — e.g.
    a host callback inside the SH synthesis path, or a
    ``stop_gradient`` slipped into the ``log(target/now)`` correction
    computation.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
        isothermal_rest_state_spectral,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_gaussian_grid(21)
        sigma = create_sigma_coordinate(8)
        cfg = SpectralPEConfig(
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = SpectralPrimitiveEquationModel(
            grid, sigma, cfg, allow_unsupported_backend=True,
        )
        state0 = isothermal_rest_state_spectral(grid, sigma)

        def loss(lnps_hat_data):
            state = state0._replace(
                lnps_hat=state0.lnps_hat.replace(data=lnps_hat_data))
            s = model.step(state, 600.0)
            # |T_hat|^2 sums (complex → real magnitude).
            return jnp.sum(jnp.abs(s.T_hat.data) ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.lnps_hat.data)

        assert jnp.all(jnp.isfinite(jnp.abs(grad))), (
            "spectral PE jax.grad produced non-finite gradient"
        )
        assert float(jnp.max(jnp.abs(grad))) > 0.0, (
            "spectral PE jax.grad collapsed to zero — possible "
            "stop_gradient or non-diff op in the SH round-trip fixer"
        )
        # Holomorphic AD on complex inputs yields complex gradients.
        assert grad.dtype == jnp.complex128, (
            f"Expected complex128 gradient on complex lnps_hat input, "
            f"got {grad.dtype}"
        )
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_cube_nh():
    """iter-40: ``jax.grad`` through anchored ``step()`` on cube NH.

    Third AD code path after iter-37 (cube PE additive fixer) and
    iter-39 (spectral PE SH log-scale fixer). Cube NH uses the
    ``fix_mass_nonhydrostatic`` path that applies a uniform additive
    correction to ``rho_prime`` (3-D state) normalised by ``∫ J·dz·dA``.

    Catches NH-specific AD regressions — e.g. a stop_gradient on the
    volume normaliser or a host-callback inside ``compute_nh_dry_mass``.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel,
        CDGridCompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_cubed_sphere(8)
        state0, hcoord, tmetric = dcmip25_tc1_init(grid, n_levels=8)
        cfg = CDGridCompressibleEulerConfig(
            n_acoustic_substeps=10, semi_implicit_acoustic=True,
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = CDGridCompressibleEulerModel(
            grid, hcoord, tmetric, cfg)

        def loss(rho_prime_data):
            state = state0._replace(
                rho_prime=state0.rho_prime.replace(data=rho_prime_data))
            s = model.step(state, 5.0)
            return jnp.sum(s.theta_prime.data ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.rho_prime.data)

        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.float64
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_spectral_nh():
    """iter-41: ``jax.grad`` through anchored ``step()`` on spectral NH.

    Fourth structural AD path:
      cube PE  additive 2-D       (iter-37, iter-38 multi-step)
      spectral PE SH 2-D log-scale (iter-39)
      cube NH  additive 3-D       (iter-40)
      spectral NH SH 3-D log-scale (this iter)

    Spectral NH fixer ``_apply_mass_fixer`` reaches grid space via
    ``sh_synthesis_3d`` on a (n_sh, nlev) state and writes back to
    ``rho_prime_hat[0, :]`` (per-level (0,0) coefficient) with
    ``Δρ·sqrt(4π)``.  AD must traverse the 3-D SH round-trip + the
    sigma vertical integral inside the fixer.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_gaussian_grid(21)
        state0, hcoord, tmetric = dcmip25_tc1_init_spectral(
            grid, n_levels=8)
        cfg = SpectralNHConfig(
            n_acoustic_substeps=10, semi_implicit_acoustic=True,
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = SpectralCompressibleEulerModel(
            grid, hcoord, tmetric, cfg, allow_unsupported_backend=True,
        )

        def loss(rho_prime_hat_data):
            state = state0._replace(
                rho_prime_hat=state0.rho_prime_hat.replace(
                    data=rho_prime_hat_data))
            s = model.step(state, 5.0)
            return jnp.sum(jnp.abs(s.theta_prime_hat.data) ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.rho_prime_hat.data)

        assert jnp.all(jnp.isfinite(jnp.abs(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.complex128
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_latlon_pe():
    """iter-55: ``jax.grad`` through anchored ``step()`` on lat-lon PE.

    Different code path from iter-37 (cube PE) and iter-39/41 (spectral):
    lat-lon PE's fix uses ``_apply_safety_rails`` which combines T-floor,
    p_s-floor, mass fixer, and hybrid-coord tracer rescale.  AD must
    flow through every branch even when tracers are absent.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_latlon_grid(36, 72)
        sigma = create_sigma_coordinate(8)
        dx_pole = float(grid.radius) * grid.dlon * math.cos(
            math.pi / 2 - grid.dlat / 2)
        dt = min(200.0, 0.5 * dx_pole / 300.0)
        cfg = CGridLatLonPrimitiveEquationConfig(
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
        state0 = hydrostatic_to_cgrid(
            held_suarez_init_latlon(grid, sigma), grid,
        )

        def loss(p_s_data):
            state = state0._replace(p_s=p_s_data)
            s = model.step(state, dt)
            return jnp.sum(s.T ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.p_s)

        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.float64
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_mpas_sw():
    """iter-56: ``jax.grad`` through anchored ``step()`` on MPAS SW.

    Voronoi-mesh code path — structurally different from the cubed-sphere
    PPM transport (cube PE/SW/NH) and the Gaussian-grid SH transform
    (spectral PE/NH).  Uses TRiSK vector Laplacian and Bernoulli-form
    SW tendencies; mass fixer is the iter-6 anchored additive correction
    via ``fix_mass_mpas`` with the iter-8 ``target_mass`` kwarg path.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test5_mpas,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        mesh = create_voronoi_mesh(4)
        cfg = MPASShallowWaterConfig(
            nu_del4=0.0, fix_mass=True, anchor_mass_to_initial=True,
        )
        model = MPASShallowWaterModel(mesh, cfg)
        state0 = williamson_test5_mpas(mesh)

        def loss(h_data):
            state = state0._replace(h=state0.h.replace(data=h_data))
            s = model.step(state, 300.0)
            return jnp.sum(s.h.data ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.h.data)

        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.float64
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_mpas_pe():
    """iter-57: ``jax.grad`` through anchored ``step()`` on MPAS PE.

    Seventh AD structural path — MPAS hydrostatic PE with vertical
    coupling.  Uses Voronoi mesh + TRiSK (like iter-56 MPAS SW) but
    adds 3-D thermodynamics and the iter-11 anchored
    ``_fix_mass_mpas_hydro`` (uniform p_s correction with fp64 budget).
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        mesh = create_voronoi_mesh(4)
        sigma = create_sigma_coordinate(8)
        cfg = MPASPrimitiveEquationConfig(
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
        state0 = held_suarez_init_mpas(mesh, sigma)

        def loss(p_s_data):
            state = state0._replace(
                p_s=state0.p_s.replace(data=p_s_data))
            s = model.step(state, 200.0)
            return jnp.sum(s.T.data ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.p_s.data)

        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.float64
    finally:
        set_policy(saved)


def test_anchored_step_no_tracer_leak_across_traces():
    """iter-59: anchored ``step()`` under TWO separate outer jits.

    Historic bug (A1 gate, job 8463229): ``step()`` cached
    ``global_integral(state.p_s)`` on ``self._target_mass`` even when
    called inside an OUTER trace (``jax.jit``/``lax.scan`` wrapper, the
    make_sharded_step / probe / bench-driver pattern).  The cached
    value was a tracer from trace 1; the second outer jit then read the
    dead tracer → ``UnexpectedTracerError`` at
    ``primitive_eq_cdgrid.py: step``.  Fixed by never caching traced
    snapshots — the per-call pre-step mass is threaded into
    ``_step_fv3`` as a traced argument instead (telescoping fixer
    semantics, identical to the non-anchor branch).

    Asserts: two distinct jit closures over the SAME model both run;
    the traced path leaves ``_target_mass`` unset; a later eager call
    still caches a concrete snapshot (designed sticky behaviour).
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
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

        @jax.jit
        def run_a(s):
            return model.step(s, 600.0)

        @jax.jit
        def run_b(s):
            def body(c, _):
                return model.step(c, 600.0), None
            return jax.lax.scan(body, s, None, length=2)[0]

        out_a = run_a(state)
        # Traced snapshot must NOT be cached on self.
        assert model._target_mass is None, (
            "step() under an outer jit cached a traced _target_mass — "
            "the next trace would read a dead tracer "
            "(UnexpectedTracerError)"
        )
        # Second, separate trace over the same model: the historic
        # crash site.  Must build and run cleanly.
        out_b = run_b(state)
        assert jnp.all(jnp.isfinite(out_b.p_s.data))

        # Mass is still anchored (telescoping): post-step mass equals
        # the pre-step mass to fp precision in BOTH traced runs.
        def _mass(s):
            return float(jnp.sum(
                s.p_s.data.astype(jnp.float64)
                * grid.area.astype(jnp.float64)))

        for out in (out_a, out_b):
            rel = abs(_mass(out) - _mass(state)) / abs(_mass(state))
            assert rel < 1e-12, (
                f"anchored mass drift {rel:.2e} > 1e-12 under outer jit"
            )

        # Eager call afterwards still caches the concrete snapshot
        # (sticky designed behaviour, iter-31).
        _ = model.step(state, 600.0)
        assert model._target_mass is not None
        assert not isinstance(model._target_mass, jax.core.Tracer)

        # DIRECT public-alias callers (step_cell_centre, which does not
        # go through step()'s threading) must still honour a concrete
        # set_target_mass(...): the consumption-site fallback prefers
        # the cached concrete target over the per-call pre-step mass.
        custom = jnp.asarray(_mass(state) * 1.01, dtype=jnp.float64)
        model.set_target_mass(custom)
        out_c = model.step_cell_centre(state, 600.0)
        rel_c = abs(_mass(out_c) - float(custom)) / float(custom)
        assert rel_c < 1e-12, (
            f"step_cell_centre ignored set_target_mass: post-step mass "
            f"off custom anchor by rel {rel_c:.2e}"
        )
        model.reset_target_mass()
    finally:
        set_policy(saved)


def test_anchored_step_supports_jax_grad_mpas_nh():
    """iter-58: ``jax.grad`` through anchored ``step()`` on MPAS NH.

    Final structural AD path — completes the 8-cover matrix:
      cube PE / spectral PE / cube NH / spectral NH / lat-lon PE
      / MPAS SW / MPAS PE / MPAS NH

    MPAS NH adds the iter-8 ``fix_mass_nonhydrostatic_mpas`` (uniform
    rho_prime correction normalised by ``∫ J·dz·dA``) on top of the
    iter-56/57 Voronoi-mesh AD foundation.  AD now flows through the
    split-explicit acoustic substeps + slow-tendency RK3 + the
    rho_prime fixer.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
    )
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
        dcmip25_tc1_init_mpas,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        mesh = create_voronoi_mesh(4)
        state0, hcoord, tmetric = dcmip25_tc1_init_mpas(
            mesh, n_levels=8)
        cfg = MPASCompressibleEulerConfig(
            n_acoustic_substeps=10,
            fix_mass=True, anchor_mass_to_initial=True,
        )
        model = MPASCompressibleEulerModel(
            mesh, hcoord, tmetric, cfg)

        def loss(rho_prime_data):
            state = state0._replace(
                rho_prime=state0.rho_prime.replace(data=rho_prime_data))
            s = model.step(state, 5.0)
            return jnp.sum(s.theta_prime.data ** 2)

        model.reset_target_mass()
        grad = jax.grad(loss)(state0.rho_prime.data)

        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
        assert grad.dtype == jnp.float64
    finally:
        set_policy(saved)


# ---------------------------------------------------------------------------
# gh-417: the three MPAS dycores carried the same lazy-cache-inside-step
# tracer leak as the cdgrid A1-gate bug above (and the ocean rigid-lid
# island cache, PR #416): step() under an OUTER jit cached a traced
# _target_mass (and, for MPAS PE, a traced _phys_state) on self, so the
# next trace read a dead tracer -> UnexpectedTracerError.  Fix = the
# cdgrid pattern: thread the per-call pre-step mass, never cache traced
# values; PE additionally skips the _phys_state stash under trace.
# ---------------------------------------------------------------------------

def _assert_no_traced_cache(model, state, step_once, scan_two):
    """Shared body: traced first-use must not cache; reuse must retrace
    cleanly; a later eager call still caches the sticky concrete snapshot."""
    out_a = step_once(state)
    assert model._target_mass is None, (
        "step() under an outer jit cached a traced _target_mass — the next "
        "trace would read a dead tracer (UnexpectedTracerError; gh-417)")
    out_b = scan_two(state)  # historic crash site: second, separate trace
    leaves = jax.tree_util.tree_leaves(out_b)
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves
               if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating))
    return out_a, out_b


def test_anchored_step_no_tracer_leak_mpas_sw():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test5_mpas,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        mesh = create_voronoi_mesh(2)
        cfg = MPASShallowWaterConfig(
            nu_del4=0.0, fix_mass=True, anchor_mass_to_initial=True)
        model = MPASShallowWaterModel(mesh, cfg)
        state = williamson_test5_mpas(mesh)

        step_once = jax.jit(lambda s: model.step(s, 300.0))

        @jax.jit
        def scan_two(s):
            def body(c, _):
                return model.step(c, 300.0), None
            return jax.lax.scan(body, s, None, length=2)[0]

        _assert_no_traced_cache(model, state, step_once, scan_two)
        # eager call afterwards: sticky concrete snapshot
        model.step(state, 300.0)
        assert model._target_mass is not None
        assert not isinstance(model._target_mass, jax.core.Tracer)
    finally:
        set_policy(saved)


def test_anchored_step_no_tracer_leak_mpas_pe():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        mesh = create_voronoi_mesh(2)
        sigma = create_sigma_coordinate(8)
        cfg = MPASPrimitiveEquationConfig(
            fix_mass=True, anchor_mass_to_initial=True)
        model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
        state = held_suarez_init_mpas(mesh, sigma)

        step_once = jax.jit(lambda s: model.step(s, 200.0))

        @jax.jit
        def scan_two(s):
            def body(c, _):
                return model.step(c, 200.0), None
            return jax.lax.scan(body, s, None, length=2)[0]

        _assert_no_traced_cache(model, state, step_once, scan_two)
        # the _phys_state side-channel must not hold a tracer either
        assert not any(
            isinstance(l, jax.core.Tracer)
            for l in jax.tree_util.tree_leaves(model._phys_state))
        model.step(state, 200.0)
        assert model._target_mass is not None
        assert not isinstance(model._target_mass, jax.core.Tracer)
    finally:
        set_policy(saved)


def test_anchored_step_no_tracer_leak_mpas_nh():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
    )
    from tests.unit.test_mpas_atmosphere import (
        _make_mesh, _make_nh_state, _add_perturbation_nh,
    )
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        nlev = 5
        mesh = _make_mesh(level=2)
        hc = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros(mesh.nCells, dtype=jnp.float64)
        tm = compute_terrain_metric(z_s, hc)
        cfg = MPASCompressibleEulerConfig(
            nu_del2=1e4, n_acoustic_substeps=4,
            fix_mass=True, anchor_mass_to_initial=True)
        model = MPASCompressibleEulerModel(mesh, hc, tm, cfg)
        state = _add_perturbation_nh(
            _make_nh_state(mesh, hc, tm, nlev), mesh, nlev)

        step_once = jax.jit(lambda s: model.step(s, 5.0))

        @jax.jit
        def scan_two(s):
            def body(c, _):
                return model.step(c, 5.0), None
            return jax.lax.scan(body, s, None, length=2)[0]

        _assert_no_traced_cache(model, state, step_once, scan_two)
        model.step(state, 5.0)
        assert model._target_mass is not None
        assert not isinstance(model._target_mass, jax.core.Tracer)
    finally:
        set_policy(saved)
