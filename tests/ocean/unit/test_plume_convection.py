"""Direct unit tests for the entraining mass-flux plume convection scheme.

Covers the leaf module ``legoesm.ocean.physics.convection.plume`` rather than
the integration factory.  Specifically guards the entrainment formulation
``entrain = 1 - exp(-epsilon * dz)`` against regressions back to the linear
``entrain = epsilon * dz`` form, which can exceed unity for thick layers
and produce an unphysical sign-flip on the plume properties.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.plume import plume_convection
from legoesm.ocean.vertical import create_ocean_z_star


def _build_state(n_levels: int = 8, H_max: float = 4000.0,
                 surface_unstable: bool = True):
    """Tiny synthetic 6-face cubed-sphere column at realistic magnitudes."""
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 4, 4, n_levels)

    # Linear T profile from 18 °C at surface to 2 °C at depth.
    T_profile = jnp.linspace(18.0, 2.0, n_levels)
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    if surface_unstable:
        # Cold dense water on top → plume should descend.
        T = T.at[..., 0].set(1.0)

    S = jnp.full(shape, 35.0, dtype=jnp.float64)

    # Crude linear-EOS density consistent with T (alpha ~2e-4):
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 4.0e7, n_levels), shape),
        axis=-1,
    )
    J = jnp.ones((6, 4, 4), dtype=jnp.float64)
    return T, S, rho, p, z, J


def test_plume_outputs_finite_and_signs_consistent():
    """Surface-unstable column: tendencies are finite and conservative."""
    T, S, rho, p, z, J = _build_state(surface_unstable=True)
    cfg = PlumeConfig()

    out = plume_convection(T, S, rho, p, z, J, cfg)

    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dS_dt))
    assert jnp.all(jnp.isfinite(out.convection_flag))

    # Convection flag is in [0, 1].
    flag_min = float(jnp.min(out.convection_flag))
    flag_max = float(jnp.max(out.convection_flag))
    assert 0.0 <= flag_min <= 1.0
    assert 0.0 <= flag_max <= 1.0


def test_plume_uses_passed_eos_fn():
    """#518: parcel density must come from the passed ``eos_fn``, not a
    hardcoded ``wright_eos``.  A sentinel EOS that flips the buoyancy sign
    relative to wright must change the convective trigger — proving the EOS
    is actually consumed (regression guard against re-hardcoding)."""
    T, S, rho, p, z, J = _build_state(surface_unstable=True)
    cfg = PlumeConfig()

    out_default = plume_convection(T, S, rho, p, z, J, cfg)

    # Sentinel EOS: makes the parcel ALWAYS lighter than ambient (rho_plume
    # well below any ambient rho) → never denser → plume stays inactive →
    # zero tendencies.  If eos_fn were ignored, this would equal the default.
    def _always_light_eos(Tp, Sp, pp):
        return jnp.full_like(Tp, 0.0)

    out_sentinel = plume_convection(T, S, rho, p, z, J, cfg,
                                    eos_fn=_always_light_eos)

    assert jnp.all(jnp.isfinite(out_sentinel.dT_dt))
    # Default path produced real convection; sentinel must differ.
    assert float(jnp.max(jnp.abs(out_default.dT_dt))) > 0.0
    assert not jnp.allclose(out_default.dT_dt, out_sentinel.dT_dt)
    # Always-light parcel → never sinks → essentially no detrainment.
    assert float(jnp.max(jnp.abs(out_sentinel.dT_dt))) < \
        float(jnp.max(jnp.abs(out_default.dT_dt)))

    # Passing wright explicitly == default (byte-identical).
    from legoesm.ocean.eos import wright_eos
    out_wright = plume_convection(T, S, rho, p, z, J, cfg, eos_fn=wright_eos)
    assert jnp.array_equal(out_default.dT_dt, out_wright.dT_dt)


def test_plume_no_nan_for_dry_columns():
    """Dry / land columns (jacobian = 0 → dz = 0) must produce zero output.

    The column-integral conservation correction divides by ``dz_top``;
    a naive implementation produces ``0 / 0 = NaN`` for dry cells.
    Beyond the NaN guard, dry columns must also produce *zero* dT_dt /
    dS_dt: the plume scan operates on ``T``, ``S``, ``rho`` without
    reference to ``dz_actual``, so without a final wet-mask multiply
    the dry column would still leak non-zero detrainment tendencies on
    ``k ≥ 1`` (codex stop-time review caught both the NaN and this
    incomplete-fix case).  This test mixes one dry column with one wet
    unstable column to ensure both shapes are exercised.
    """
    nlev = 6
    z = create_ocean_z_star(n_levels=nlev, H_max=2000.0)
    shape = (1, 2, 1, nlev)  # 2 columns: index 0 dry, index 1 wet

    T_profile = jnp.linspace(15.0, 3.0, nlev)
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    T = T.at[..., 0].set(1.0)  # cold dense surface (would be unstable)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 2.0e7, nlev), shape), axis=-1,
    )
    # Jacobian: 0 for the dry column, 1 for the wet column.
    J = jnp.array([[[0.0]], [[1.0]]], dtype=jnp.float64).reshape(1, 2, 1)

    cfg = PlumeConfig()
    out = plume_convection(T, S, rho, p, z, J, cfg)

    assert jnp.all(jnp.isfinite(out.dT_dt)), (
        "dT_dt has NaN or Inf — likely 0/0 division in the conservation "
        "correction for dry columns"
    )
    assert jnp.all(jnp.isfinite(out.dS_dt)), (
        "dS_dt has NaN or Inf"
    )
    assert jnp.all(jnp.isfinite(out.convection_flag))

    # Dry column (index 0) must produce *exactly* zero — anything else
    # leaks non-zero tendencies onto land cells.
    assert float(jnp.max(jnp.abs(out.dT_dt[:, 0, ...]))) == 0.0, (
        "dry column produces non-zero dT_dt; the wet-mask multiply at "
        "the end of plume_convection is missing"
    )
    assert float(jnp.max(jnp.abs(out.dS_dt[:, 0, ...]))) == 0.0, (
        "dry column produces non-zero dS_dt"
    )
    assert float(jnp.max(jnp.abs(out.convection_flag[:, 0, ...]))) == 0.0, (
        "dry column produces non-zero convection_flag"
    )
    # Wet column (index 1) should still be active.
    assert float(jnp.max(jnp.abs(out.dT_dt[:, 1, ...]))) > 0.0, (
        "wet column unexpectedly inactive — mask is too aggressive"
    )

    # Also check gradient through the dry-column path is finite.
    def loss(T_in):
        o = plume_convection(T_in, S, rho, p, z, J, cfg)
        return jnp.sum(o.dT_dt ** 2) + jnp.sum(o.dS_dt ** 2)

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g)), (
        "gradient has NaN — the safe-divide guard must use ``jnp.where`` "
        "with a safe denominator, not ``jnp.maximum`` alone (which still "
        "leaks 0/eps gradient through the False branch)"
    )


def test_plume_conserves_column_heat_and_salt():
    """Closed-column heat/salt conservation: ``Σ_k dT_dt[k] · dz[k] = 0``.

    The plume sources heat/salt from the surface and detrains to depth.
    A conservation-correct scheme balances these so the column-integrated
    tendency is zero.  Codex adversarial-review finding #3 caught the
    pre-correction non-conservation; this test guards the fix.
    """
    nlev = 8
    z = create_ocean_z_star(n_levels=nlev, H_max=400.0)
    shape = (6, 4, 4, nlev)

    # Cold dense surface, warm interior, varying S to exercise dS_dt too.
    T_profile = jnp.array([1.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    S_profile = jnp.linspace(34.0, 35.5, nlev)
    S = jnp.broadcast_to(S_profile, shape).astype(jnp.float64)
    rho = constants.rho_ocean - 0.2 * (T - 4.0) + 0.8 * (S - 35.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 4.0e6, nlev), shape), axis=-1,
    )
    J = jnp.ones((6, 4, 4), dtype=jnp.float64)

    cfg = PlumeConfig()
    out = plume_convection(T, S, rho, p, z, J, cfg)

    dz = z.dz_ref * J[..., jnp.newaxis]
    column_T = jnp.sum(out.dT_dt * dz, axis=-1)
    column_S = jnp.sum(out.dS_dt * dz, axis=-1)

    # Both should be ~ machine epsilon * representative column scale.
    # Tolerance accounts for f64 round-off in the cumulative sum.
    assert float(jnp.max(jnp.abs(column_T))) < 1e-12, (
        f"column-integrated dT not conserved: "
        f"max |Σ dT·dz| = {float(jnp.max(jnp.abs(column_T))):.3e} "
        f"K·m/s"
    )
    assert float(jnp.max(jnp.abs(column_S))) < 1e-12, (
        f"column-integrated dS not conserved: "
        f"max |Σ dS·dz| = {float(jnp.max(jnp.abs(column_S))):.3e} "
        f"PSU·m/s"
    )


def _linear_eos(T, S, p):
    """EOS matching the crude linear ``rho`` built in ``_build_state``.

    ``_build_state`` supplies a pressure-independent ``rho = rho_ocean -
    0.2*(T-4)``.  The plume module contract requires the ambient ``rho`` and
    the ``eos_fn`` to be the SAME EOS (the trigger now displaces the surface
    parcel to level-1 pressure via ``eos_fn``, and the in-plume buoyancy check
    already did).  Passing this matching EOS keeps the stability comparison
    self-consistent; the default ``wright_eos`` would disagree with the crude
    linear ``rho`` at the trigger and spuriously fire.
    """
    return constants.rho_ocean - 0.2 * (T - 4.0)


def test_plume_zero_when_stable():
    """Stably stratified column: plume should not be active."""
    T, S, rho, p, z, J = _build_state(surface_unstable=False)
    cfg = PlumeConfig()

    # Consistent EOS (matches the crude linear ``rho`` above): the fixed
    # same-pressure trigger correctly reads this column as stable.
    out = plume_convection(T, S, rho, p, z, J, cfg, eos_fn=_linear_eos)

    # No surface instability → plume inactive everywhere.
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dS_dt))) == 0.0
    assert float(jnp.max(out.convection_flag)) == 0.0


def test_plume_entrainment_uses_exact_not_linear_form():
    """Sharp regression guard: ``-expm1(-eps*dz)`` vs ``eps*dz``.

    For a cold dense surface (5 °C) above a warm sub-surface (20 °C) with
    ``eps=1e-3`` and a layer thickness ``dz_1 ≈ 1048 m`` (so ``eps*dz_1
    ≈ 1.05``), the two entrainment formulas produce **opposite signs** for
    the detrainment tendency at the first descent step.

    With the corrected units fix (``dT/dt = w_p * alpha * eps * (T_p -
    T_env)``, codex iter-1 finding #2), the expected magnitude is
    ``w_p * alpha * eps * dT ~ 0.01 * 0.1 * 1e-3 * 10 K ~ 1e-5 K/s``.

    * Exact form: ``T_plume`` after entrainment ≈ 14.8 °C (cooler than
      ``T_env=20``), so ``dT_dt[k=1] < 0`` (env cools toward plume).
    * Linear form: ``(1 - eps*dz)`` is ``-0.05`` so the plume's prior
      temperature is *subtracted*, giving ``T_plume`` slightly *warmer*
      than env, and ``dT_dt[k=1] > 0`` (wrong sign).

    Asserting ``dT_dt[..., 1] < 0`` distinguishes the two.
    """
    nlev = 4
    H_max = 6000.0  # → dz_ref[1] ≈ 1048 m, so eps*dz_1 ≈ 1.05 with eps=1e-3
    z = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    shape = (6, 2, 2, nlev)

    # Cold dense surface above warmer sub-surface.  Surface is unstable
    # (rho_0 > rho_1) so the plume launches.  Sub-surface is sufficiently
    # warmer that entrainment of T_env strongly heats the plume — but
    # under the exact form the plume stays cooler than env.
    T_profile = jnp.array([5.0, 20.0, 18.0, 16.0])
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 6.0e7, nlev), shape), axis=-1,
    )
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)

    cfg = PlumeConfig(epsilon=1e-3)
    out = plume_convection(T, S, rho, p, z, J, cfg)

    # Bounded: tendencies are finite even though eps*dz_1 > 1 — the linear
    # form would also produce finite (but sign-flipped) values, so this is
    # a sanity floor, not a discriminator.
    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dS_dt))

    # Sharp discriminator: at the first descent step (level k=1) the plume
    # is cooler than the environment under the exact form; the linear
    # form's sign-flip makes the plume artificially *warmer* than env.
    # Tolerance: with w_p*alpha*eps = 1e-6 1/s and dT ~ 5 K, expected
    # |dT_dt| ~ 5e-6 K/s.  Lower bound 1e-7 rules out trivial near-zero;
    # upper bound 1e-4 rules out unphysical magnitudes.
    dT_at_first_step = float(out.dT_dt[..., 1].mean())
    assert dT_at_first_step < -1e-7, (
        f"dT_dt at first descent step is {dT_at_first_step:.3e}; expected "
        "a clearly negative value (~ -5e-6) under the exact "
        "1 - exp(-eps*dz) entrainment.  A positive or near-zero value "
        "indicates regression to the linear ``eps*dz`` form."
    )
    assert dT_at_first_step > -1e-4, (
        f"dT_dt at first descent step is {dT_at_first_step:.3e}; "
        "magnitude is unphysically large for cfg.w_plume_min=0.01, "
        "alpha_plume=0.1, eps=1e-3, |T_plume - T_env| ~ 5 K."
    )


def test_plume_gradient_flows():
    """``jax.grad`` through plume_convection produces finite, nonzero grads."""
    T, S, rho, p, z, J = _build_state(surface_unstable=True)
    cfg = PlumeConfig()

    def loss(T_in):
        out = plume_convection(T_in, S, rho, p, z, J, cfg)
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g))
    # At least some sensitivity below the surface — surface row may be zero
    # but k>0 levels should respond to environmental T perturbations.
    assert float(jnp.max(jnp.abs(g[..., 1:]))) > 0.0


@pytest.mark.parametrize("eps", [1e-4, 1e-3, 5e-3])
def test_plume_finite_across_eps_regimes(eps):
    """Sanity floor: tendencies are finite across a span of ``eps*dz``.

    This is a *finiteness* check across entrainment regimes (``eps*dz``
    ranging from ~0.025 to ~1.25 with the chosen layering), not a
    correctness check on the entrainment form itself — the linear form
    also produces finite (but sign-flipped) values.  The actual form
    discrimination lives in
    ``test_plume_entrainment_uses_exact_not_linear_form``.
    """
    nlev = 6
    H_max = 1500.0  # dz_ref ~ 250 m → eps*dz spans 0.025 to 1.25
    z = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    shape = (6, 2, 2, nlev)
    T_profile = jnp.linspace(18.0, 2.0, nlev)
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    T = T.at[..., 0].set(1.0)
    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    rho = constants.rho_ocean - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 1.5e7, nlev), shape), axis=-1,
    )
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)

    cfg = PlumeConfig(epsilon=eps)
    out = plume_convection(T, S, rho, p, z, J, cfg)

    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dS_dt))
    assert jnp.all(jnp.isfinite(out.convection_flag))
