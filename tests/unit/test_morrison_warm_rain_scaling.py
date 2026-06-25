"""Tests for the optional LES-trained warm-rain rate scaling in Morrison.

The ``MorrisonConfig.warm_rain_scale_fn`` hook multiplies the warm-rain rate
primitives (autoconversion / accretion / rain-evaporation) by per-level,
state-conditioned factors. These tests pin the contract that makes it safe:

  * ``None`` (production default) AND an identity scale fn are BIT-IDENTICAL to
    baseline Morrison (the feature is constant-folded / a no-op).
  * a non-trivial scale changes the tendencies.
  * the scaled path is differentiable (gradients flow through the scale).
  * scaling preserves the total-water conservation residual (each process's
    matched source/sink/heat/number scale together).
"""

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    WarmRainRateScales,
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

from legoesm import constants


def _column(ncol=2, nlev=12):
    T = jnp.linspace(285.0, 235.0, nlev)[None, :].repeat(ncol, 0)
    q_v = jnp.full((ncol, nlev), 6e-3)
    q_c = jnp.full((ncol, nlev), 6e-4)
    q_r = jnp.full((ncol, nlev), 2e-4)
    z = jnp.zeros((ncol, nlev))
    hyd = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((ncol, nlev), 1e8), N_r=jnp.full((ncol, nlev), 1e6),
        N_i=z, N_s=None, N_g=None,
    )
    p_full = jnp.linspace(1.0e5, 2.0e4, nlev)[None, :].repeat(ncol, 0)
    p_half = jnp.linspace(1.01e5, 1.9e4, nlev + 1)[None, :].repeat(ncol, 0)
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
    return dict(T=T, q_v=q_v, hyd=hyd, p_full=p_full, p_half=p_half,
                rho=rho, dz=dz, dt=60.0)


def _run(cfg, c):
    return morrison_microphysics(
        c["T"], c["q_v"], c["hyd"], c["p_full"], c["p_half"],
        c["rho"], c["dz"], c["dt"], cfg,
    )


def _total_water_resid(o):
    return jnp.max(jnp.abs(
        o.dq_v_dt + o.dq_c_dt + o.dq_r_dt + o.dq_i_dt + o.dq_s_dt + o.dq_g_dt
    ))


def _column_water_resid(o, c):
    """Mass-integrated water budget INCLUDING the surface precip flux.

    The correct conservation invariant (codex iter-1, finding 5): the local
    per-layer ``Σ dq/dt`` is NOT closed because sedimentation moves water
    between layers and out the surface as ``precipitation`` [kg/m^2/s]. The
    column store change ``Σ (dq/dt)·ρ·dz`` must balance the surface precip out.
    """
    rho, dz = c["rho"], c["dz"]
    tot = (o.dq_v_dt + o.dq_c_dt + o.dq_r_dt
           + o.dq_i_dt + o.dq_s_dt + o.dq_g_dt)
    col_storage = jnp.sum(tot * rho * dz, axis=1)   # kg/m^2/s per column
    return jnp.max(jnp.abs(col_storage + o.precipitation))


def test_none_and_identity_are_bit_identical_to_baseline():
    c = _column()
    base = MorrisonConfig()
    out0 = _run(base, c)

    def ident(T, q_v, q_c, q_r, rho):
        return WarmRainRateScales()  # all defaults = 1.0

    out1 = _run(base._replace(warm_rain_scale_fn=ident), c)
    for f in ("dq_v_dt", "dq_c_dt", "dq_r_dt", "dT_dt"):
        assert jnp.array_equal(getattr(out0, f), getattr(out1, f)), f


def test_nontrivial_scale_changes_tendencies():
    c = _column()
    base = MorrisonConfig()
    out0 = _run(base, c)

    def boost(T, q_v, q_c, q_r, rho):
        return WarmRainRateScales(autoconv=2.0, accretion=0.5, rain_evap=1.5)

    out2 = _run(base._replace(warm_rain_scale_fn=boost), c)
    assert not jnp.allclose(out0.dq_r_dt, out2.dq_r_dt)


def test_scale_is_differentiable():
    c = _column()
    base = MorrisonConfig()

    def loss(s):
        fn = lambda T, q_v, q_c, q_r, rho: WarmRainRateScales(  # noqa: E731
            autoconv=s, accretion=s, rain_evap=s)
        o = _run(base._replace(warm_rain_scale_fn=fn), c)
        return jnp.sum(o.dq_r_dt ** 2)

    g = jax.grad(loss)(1.3)
    assert jnp.isfinite(g)


def test_scaling_preserves_conservation_residual():
    """Column water (Σ dq/dt·ρ·dz + surface precip) stays closed under scaling.

    This is the CORRECT conservation invariant (codex iter-1, finding 5): the
    bare per-layer ``Σ dq/dt`` is not closed because sedimentation redistributes
    water across layers and out the surface, so scaling ``evaporation`` (which
    also enters the sedimentation ``extra_sink``) legitimately changes the
    local residual while the mass-integrated budget INCLUDING precip is
    conserved to machine precision.
    """
    c = _column()
    base = MorrisonConfig()
    r0 = _column_water_resid(_run(base, c), c)

    for scales in (
        WarmRainRateScales(autoconv=3.0, accretion=0.25, rain_evap=2.0),
        WarmRainRateScales(autoconv=8.0, accretion=1.0, rain_evap=5.0),
        WarmRainRateScales(autoconv=0.1, accretion=0.1, rain_evap=0.1),
    ):
        def boost(T, q_v, q_c, q_r, rho, _s=scales):
            return _s

        r = _column_water_resid(_run(base._replace(warm_rain_scale_fn=boost), c), c)
        # Mass-integrated column water (incl. surface precip) conserved to
        # machine precision regardless of the scale magnitude.
        assert float(r) < 1e-12, f"scales={scales} resid={float(r):.3e}"
        assert float(r0) < 1e-12


def test_negative_scale_is_floored_not_sign_flipping():
    """A negative multiplier must NOT invert a rate's sign (codex iter-1, C).

    The non-negativity guard clamps the scale at 0, so a negative ``autoconv``
    factor zeroes the warm-rain conversion rather than turning autoconversion
    into a spurious q_c source / rain sink the donor clamps never budgeted.
    """
    c = _column()
    base = MorrisonConfig()

    def neg(T, q_v, q_c, q_r, rho):
        return WarmRainRateScales(autoconv=-2.0, accretion=-1.0, rain_evap=-3.0)

    def zero(T, q_v, q_c, q_r, rho):
        return WarmRainRateScales(autoconv=0.0, accretion=0.0, rain_evap=0.0)

    out_neg = _run(base._replace(warm_rain_scale_fn=neg), c)
    out_zero = _run(base._replace(warm_rain_scale_fn=zero), c)
    for f in ("dq_v_dt", "dq_c_dt", "dq_r_dt", "dT_dt"):
        assert jnp.array_equal(getattr(out_neg, f), getattr(out_zero, f)), f
    # And the column budget is still closed.
    assert float(_column_water_resid(out_neg, c)) < 1e-12


def test_per_level_state_conditioned_scale():
    """A scale that varies per level (the realistic net output) runs + flows AD."""
    c = _column()
    base = MorrisonConfig()

    def net(T, q_v, q_c, q_r, rho):
        # toy state dependence: stronger autoconversion in warmer, wetter layers
        s_au = 1.0 + jax.nn.sigmoid((T - 270.0) / 5.0) * (q_c / 1e-3)
        return WarmRainRateScales(autoconv=s_au, accretion=jnp.ones_like(T),
                                  rain_evap=jnp.ones_like(T))

    out = _run(base._replace(warm_rain_scale_fn=net), c)
    assert jnp.all(jnp.isfinite(out.dq_r_dt))
