"""Cold-pool gustiness from the surface precipitation rate.

The term exists because a cloud-resolving model with NO mean wind still
evaporates ~2.7 mm/day while a single column reaches ~0.7 — its fluxes ride on
convective cold-pool gusts.  These checks pin the properties that make the term
safe to switch on: exact-zero gating, a finite gradient at zero precipitation
(the hazard of any fractional power), the cap that breaks the
gust->flux->convection->gust loop, and the quadrature combination.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import apply_gustiness, convective_gust_wind

_MM_DAY = 1.0 / 86_400.0  # kg/m^2/s per mm/day
_RHO = 1.2  # kg/m^3, near-surface air


def _rates(mm_day):
    return jnp.asarray([m * _MM_DAY for m in mm_day], dtype=jnp.float64)


def _rho_like(p):
    return jnp.full_like(p, _RHO)


def test_disabled_by_default_coefficient_is_exactly_zero():
    """coeff=0 must return EXACT zeros, not a small number.

    The quadrature ``sqrt(u^2+v^2+g^2)`` is only byte-identical to the
    pre-existing behaviour if g is exactly 0, so this is what keeps every
    shipped run unchanged when the feature is off.
    """
    p = _rates([0.0, 1.0, 2.7, 50.0])
    g = convective_gust_wind(p, 0.0, _rho_like(p))
    assert jnp.array_equal(g, jnp.zeros_like(p))


def test_matches_the_published_magnitude_at_rce_rain_rates():
    """u_c = (c L_v P / rho)^(1/3) must land where the literature says.

    At the RCE reference rate of 2.7 mm/day with c = 1 the published estimate
    is ~4 m/s (L_v*P ~ 78 W/m^2, /rho ~ 65 m^3/s^3, cube root ~ 4.0); with
    c = 0.5, ~3.2.  This is the check that the dimensional form is assembled
    correctly — an arbitrary reference-rate version cannot be compared with
    anything.
    """
    from legoesm import constants

    p = _rates([2.7])
    rho = jnp.full((1,), _RHO)
    g1 = float(convective_gust_wind(p, 1.0, rho)[0])
    expected = float((constants.L_v * float(p[0]) / _RHO) ** (1.0 / 3.0))
    assert g1 == pytest.approx(expected, rel=1e-10)
    assert 3.5 < g1 < 4.5, f"off the published ~4 m/s at 2.7 mm/day: {g1}"
    g_half = float(convective_gust_wind(p, 0.5, rho)[0])
    assert 2.8 < g_half < 3.6, f"off the published ~3.2 m/s at c=0.5: {g_half}"


def test_monotone_and_cube_root_scaling():
    p = _rates([1.0, 8.0])
    g = convective_gust_wind(p, 1.0, _rho_like(p))
    assert float(g[1]) > float(g[0])
    # 8x the rain is 2x the gust for the 1/3 exponent.
    assert float(g[1]) / float(g[0]) == pytest.approx(2.0, rel=1e-3)


def test_gradient_is_finite_at_zero_precipitation():
    """A bare ratio**(1/3) has an INFINITE derivative at 0.

    The double-``where`` makes the gradient finite AND the value exactly zero
    there.  A first version used an additive offset inside the power instead;
    this test's dry-column sibling caught that it left a spurious 0.02 m/s gust
    (coefficient-scaled) in a column with no rain at all.
    """
    p = _rates([0.0, 0.5, 2.7])
    g = jax.grad(lambda x: convective_gust_wind(x, 2.0, _rho_like(x)).sum())(p)
    assert bool(jnp.all(jnp.isfinite(g))), f"non-finite gradient: {g}"
    assert float(g[0]) == 0.0, "dry column must have an exactly zero subgradient"
    assert float(g[2]) > 0.0, "a raining column must still respond to precip"


def test_negative_precipitation_is_clamped_not_propagated():
    """An upstream sign error must not become a NaN gradient here."""
    p = jnp.asarray([-1.0 * _MM_DAY, 0.0], dtype=jnp.float64)
    g = convective_gust_wind(p, 2.0, _rho_like(p))
    # EXACT zero, not "small": a dry or sign-flipped column must not acquire a
    # coefficient-scaled gust out of the regularisation.
    assert float(g[0]) == 0.0
    assert float(g[1]) == 0.0
    assert bool(jnp.isfinite(jax.grad(
        lambda x: convective_gust_wind(x, 2.0, _rho_like(x)).sum())(p)[0]))


def test_cap_bounds_the_feedback_loop():
    huge = _rates([10_000.0])
    assert float(convective_gust_wind(huge, 2.0, _rho_like(huge), cap=5.0)[0]) == pytest.approx(
        5.0, rel=1e-6)
    # cap=0 means UNCAPPED, not "capped at zero" — the opposite reading would
    # silently disable the term.
    assert float(convective_gust_wind(huge, 2.0, _rho_like(huge), cap=0.0)[0]) > 5.0


def test_jit_parity():
    p = _rates([0.0, 2.7, 20.0])
    eager = convective_gust_wind(p, 2.0, _rho_like(p), cap=6.0)
    jitted = jax.jit(lambda x, r: convective_gust_wind(x, 2.0, r, cap=6.0))(
        p, _rho_like(p))
    np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), rtol=1e-12)


def test_combines_in_quadrature_never_linearly():
    """The gust joins the resolved wind through apply_gustiness, so a 0.5 m/s
    mean wind with a 3 m/s gust gives sqrt(0.25+9), NOT 3.5."""
    gust = float(convective_gust_wind(
        _rates([2.7]), 1.0, jnp.full((1,), _RHO))[0])
    eff = float(apply_gustiness(jnp.asarray(0.5), jnp.asarray(0.0), gust))
    assert eff == pytest.approx(float(np.sqrt(0.25 + gust ** 2)), rel=1e-6)
    assert eff < 0.5 + gust


# --- Wiring: the term must be invisible until it is switched on -------------

def test_surface_fluxes_unchanged_when_the_term_is_off():
    """Default config + a precipitating column must be BYTE-IDENTICAL.

    This is the claim that lets the feature ship dark: every existing run keeps
    its numbers because the gate is a static Python branch on coeff == 0.
    """
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    n = 4
    u = jnp.full((n,), 0.5); v = jnp.zeros((n,))
    T = jnp.full((n,), 298.0); q = jnp.full((n,), 0.016)
    T_s = jnp.full((n,), 300.0); q_s = jnp.full((n,), 0.02)
    rho = jnp.full((n,), 1.15)
    rain = _rates([0.0, 1.0, 5.0, 20.0])
    cfg = SurfaceLayerConfig()
    assert cfg.convective_gustiness_coeff == 0.0, "must ship OFF"

    base = compute_surface_fluxes(u, v, T, q, T_s, q_s, rho, cfg)
    with_p = compute_surface_fluxes(u, v, T, q, T_s, q_s, rho, cfg,
                                    sfc_precip=rain)
    for a, b in zip(base, with_p):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_enabling_the_term_raises_the_latent_flux_with_rain():
    """The whole point: rain must strengthen evaporation, monotonically."""
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    n = 3
    u = jnp.full((n,), 0.5); v = jnp.zeros((n,))
    T = jnp.full((n,), 298.0); q = jnp.full((n,), 0.016)
    T_s = jnp.full((n,), 300.0); q_s = jnp.full((n,), 0.02)
    rho = jnp.full((n,), 1.15)
    rain = _rates([0.0, 2.7, 20.0])
    cfg = SurfaceLayerConfig(convective_gustiness_coeff=2.0)

    _tx, _ty, _sh, lh, _us = compute_surface_fluxes(
        u, v, T, q, T_s, q_s, rho, cfg, sfc_precip=rain)
    lh = np.asarray(lh)
    assert lh[0] < lh[1] < lh[2], f"latent flux not monotone in rain: {lh}"
    # The dry column must equal the no-gust answer exactly.
    dry = compute_surface_fluxes(u, v, T, q, T_s, q_s, rho,
                                 SurfaceLayerConfig())[3]
    assert lh[0] == pytest.approx(float(np.asarray(dry)[0]), rel=1e-12)


def test_the_cap_is_honoured_through_the_flux_path():
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    n = 1
    args = (jnp.full((n,), 0.5), jnp.zeros((n,)), jnp.full((n,), 298.0),
            jnp.full((n,), 0.016), jnp.full((n,), 300.0), jnp.full((n,), 0.02),
            jnp.full((n,), 1.15))
    torrential = _rates([5_000.0])
    uncapped = compute_surface_fluxes(
        *args, SurfaceLayerConfig(convective_gustiness_coeff=2.0),
        sfc_precip=torrential)[3]
    capped = compute_surface_fluxes(
        *args, SurfaceLayerConfig(convective_gustiness_coeff=2.0,
                                  convective_gustiness_cap=5.0),
        sfc_precip=torrential)[3]
    assert float(np.asarray(capped)[0]) < float(np.asarray(uncapped)[0])


def test_physics_state_carries_the_precip_slot():
    """The hand-off slot must exist and start at zero, or the gust silently
    reads nothing for the whole run."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state

    ps = init_physics_state(ncol=6, nlev=10, physics_config=PhysicsConfig())
    assert hasattr(ps, "sfc_precip"), "PhysicsState lost the hand-off slot"
    assert ps.sfc_precip.shape == (6,)
    assert float(jnp.max(jnp.abs(ps.sfc_precip))) == 0.0


def test_density_enters_the_scaling():
    """u_c ~ rho^(-1/3): the same rain in denser air makes a weaker gust.

    Pins that rho is genuinely in the formula rather than carried along
    unused — the failure mode when a dimensional argument is added late.
    """
    p = _rates([2.7, 2.7])
    rho = jnp.asarray([1.0, 8.0])
    g = convective_gust_wind(p, 1.0, rho)
    assert float(g[0]) / float(g[1]) == pytest.approx(2.0, rel=1e-6)
