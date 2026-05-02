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
    rho = 1025.0 - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 4.0e7, n_levels), shape),
        axis=-1,
    )
    J = jnp.ones((6, 4, 4), dtype=jnp.float64)
    return T, S, rho, p, z, J


def test_plume_outputs_finite_and_signs_consistent():
    """Surface-unstable column: tendencies are finite and surface row is zero."""
    T, S, rho, p, z, J = _build_state(surface_unstable=True)
    cfg = PlumeConfig()

    out = plume_convection(T, S, rho, p, z, J, cfg)

    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dS_dt))
    assert jnp.all(jnp.isfinite(out.convection_flag))

    # Surface level is the plume launch — no detrainment tendency there.
    assert float(jnp.max(jnp.abs(out.dT_dt[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(out.dS_dt[..., 0]))) == 0.0

    # Convection flag is in [0, 1].
    flag_min = float(jnp.min(out.convection_flag))
    flag_max = float(jnp.max(out.convection_flag))
    assert 0.0 <= flag_min <= 1.0
    assert 0.0 <= flag_max <= 1.0


def test_plume_zero_when_stable():
    """Stably stratified column: plume should not be active."""
    T, S, rho, p, z, J = _build_state(surface_unstable=False)
    cfg = PlumeConfig()

    out = plume_convection(T, S, rho, p, z, J, cfg)

    # No surface instability → plume inactive everywhere.
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dS_dt))) == 0.0
    assert float(jnp.max(out.convection_flag)) == 0.0


def test_plume_entrainment_uses_exact_not_linear_form():
    """Sharp regression guard: ``-expm1(-eps*dz)`` vs ``eps*dz``.

    For a cold dense surface (5 °C) above a warm sub-surface (20 °C) with
    ``eps=1e-3`` and a layer thickness ``dz_1 ≈ 1048 m`` (so ``eps*dz_1
    ≈ 1.05``), the two entrainment formulas produce **opposite signs** for
    the detrainment tendency at the first descent step:

    * Exact form: ``T_plume`` after entrainment ≈ 14.8 °C (cooler than
      ``T_env=20``), so ``dT_dt[k=1] ≈ -5.2e-4 K/s`` (env cools toward
      plume).
    * Linear form: ``(1 - eps*dz)`` is ``-0.05`` so the plume's prior
      temperature is *subtracted*, giving ``T_plume ≈ 20.7`` (slightly
      *warmer* than env), and ``dT_dt[k=1] ≈ +7.1e-5 K/s`` (wrong sign).

    Asserting ``dT_dt[..., 1] < 0`` distinguishes the two; the magnitude
    bound rules out trivial near-zero passes.
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
    rho = 1025.0 - 0.2 * (T - 4.0)
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
    dT_at_first_step = float(out.dT_dt[..., 1].mean())
    assert dT_at_first_step < -1e-4, (
        f"dT_dt at first descent step is {dT_at_first_step:.3e}; expected "
        "a clearly negative value (~ -5e-4) under the exact "
        "1 - exp(-eps*dz) entrainment.  A positive or near-zero value "
        "indicates regression to the linear ``eps*dz`` form."
    )
    assert dT_at_first_step > -1e-2, (
        f"dT_dt at first descent step is {dT_at_first_step:.3e}; "
        "magnitude is unphysically large for cfg.alpha_plume=0.1, "
        "eps=1e-3, |T_plume - T_env| ~ 5 K."
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
    rho = 1025.0 - 0.2 * (T - 4.0)
    p = jnp.cumsum(
        jnp.broadcast_to(jnp.linspace(0.0, 1.5e7, nlev), shape), axis=-1,
    )
    J = jnp.ones((6, 2, 2), dtype=jnp.float64)

    cfg = PlumeConfig(epsilon=eps)
    out = plume_convection(T, S, rho, p, z, J, cfg)

    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(jnp.isfinite(out.dS_dt))
    assert jnp.all(jnp.isfinite(out.convection_flag))
