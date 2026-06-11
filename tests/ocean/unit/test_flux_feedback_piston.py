"""EXT-N3: per-cell piston-velocity SSS restoring on ``flux_feedback``.

The Veros north_atlantic setup restores surface salinity with a SPATIALLY
VARYING (monthly) piston velocity, not a scalar timescale:

  * ``sss_rest`` = forcing-file field / 100 [m/s]
    (north_atlantic.py:245-249);
  * ``forc_salt_surface = sss_rest(t)·(sss_clim(t) − salt[tau,sfc])·maskT``
    [PSU·m/s] (north_atlantic.py:336-340), ice-masked (:342-344);
  * the tracer core divides by the top-cell thickness in the implicit RHS
    (core/thermodynamics.py:282: ``dt_tracer·forc_salt_surface/dzt[-1]``)
    ⇒ net rate = piston·(S* − S)/dz₀ [PSU/s].

``OceanSurfaceForcing.S_restore_piston`` [m/s] +
``FluxFeedbackConfig.salt_restore_piston=True`` implement exactly this form,
REPLACING the scalar ``1/tau_restore_s`` rate.  Default ``None``/``False`` ⇒
bit-identical scalar path.  Gate and channel must be specified together
(trace-time ValueErrors — never a silently ignored channel, never a silent
scalar-tau fallback).

Run in fp64 on CPU for determinism.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.surface_forcing.config import FluxFeedbackConfig
from legoesm.ocean.physics.surface_forcing.flux_feedback import (
    flux_feedback_surface_forcing,
)
from legoesm.ocean.state import OceanSurfaceForcing

_N_LAT, _N_LON, _NLEV = 3, 4, 5
_DZ0 = 50.0


def _fields(seed=0, *, T_surf=10.0):
    rng = np.random.default_rng(seed)
    T = jnp.full((_N_LAT, _N_LON, _NLEV), T_surf)
    S = jnp.asarray(rng.normal(34.0, 0.2, size=(_N_LAT, _N_LON, _NLEV)))
    dz_0 = jnp.full((_N_LAT, _N_LON), _DZ0)
    S_t = jnp.asarray(rng.normal(35.0, 0.2, size=(_N_LAT, _N_LON)))
    piston = jnp.asarray(rng.uniform(1e-5, 5e-4, size=(_N_LAT, _N_LON)))
    return T, S, dz_0, S_t, piston


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------

def test_piston_rate_hand_computed():
    """dS/dt = piston·(S* − S_surf)/dz₀, per cell."""
    T, S, dz_0, S_t, piston = _fields()
    sf = OceanSurfaceForcing(S_restore_target=S_t, S_restore_piston=piston)
    out = flux_feedback_surface_forcing(
        T, S, dz_0, sf, FluxFeedbackConfig(salt_restore_piston=True))
    expect = np.asarray(piston) * (np.asarray(S_t) - np.asarray(S[..., 0])) \
        / _DZ0
    np.testing.assert_allclose(np.asarray(out.dS_dt[..., 0]), expect,
                               rtol=1e-14, atol=0)
    # Surface layer only.
    assert np.abs(np.asarray(out.dS_dt[..., 1:])).max() == 0.0
    # Heat path untouched (no heat channels given).
    assert np.abs(np.asarray(out.dT_dt)).max() == 0.0


def test_veros_north_atlantic_kernel_replica():
    """Numpy replica of the Veros NA salt kernel (north_atlantic.py:336-344 +
    core/thermodynamics.py:282) matches the scheme to 1e-14."""
    rng = np.random.default_rng(7)
    T, S, dz_0, _, _ = _fields(7, T_surf=10.0)
    # Monthly fields, interpolated exactly as the harness would (f1/f2 blend
    # happens OUTSIDE the scheme — the channel carries the blended field).
    sss_rest_file = rng.uniform(0.05, 3.0, size=(_N_LAT, _N_LON, 2))  # cm/s
    sss_clim = rng.normal(35.0, 0.3, size=(_N_LAT, _N_LON, 2))
    f1, f2 = 0.3, 0.7
    sss_rest = (f1 * sss_rest_file[..., 0] + f2 * sss_rest_file[..., 1]) / 100.0
    sss_star = f1 * sss_clim[..., 0] + f2 * sss_clim[..., 1]
    maskT = np.ones((_N_LAT, _N_LON))
    # Veros: forc_salt_surface [PSU·m/s], then /dzt[-1] in the implicit RHS.
    forc_salt = sss_rest * (sss_star - np.asarray(S[..., 0])) * maskT
    veros_rate = forc_salt / _DZ0
    out = flux_feedback_surface_forcing(
        T, S, dz_0,
        OceanSurfaceForcing(S_restore_target=jnp.asarray(sss_star),
                            S_restore_piston=jnp.asarray(sss_rest)),
        FluxFeedbackConfig(salt_restore_piston=True))
    np.testing.assert_allclose(np.asarray(out.dS_dt[..., 0]), veros_rate,
                               rtol=1e-14, atol=1e-22)


# ---------------------------------------------------------------------------
# Default bit-identity
# ---------------------------------------------------------------------------

def test_default_none_is_scalar_path_bit_identical():
    """piston=None + gate False reproduces the scalar tau_restore_s rate
    EXACTLY (the pre-EXT-N3 expression, verbatim)."""
    T, S, dz_0, S_t, _ = _fields(1)
    cfg = FluxFeedbackConfig()
    out = flux_feedback_surface_forcing(
        T, S, dz_0, OceanSurfaceForcing(S_restore_target=S_t), cfg)
    ref = (S_t - S[..., 0]) / cfg.tau_restore_s * jnp.ones_like(S_t)
    np.testing.assert_array_equal(np.asarray(out.dS_dt[..., 0]),
                                  np.asarray(ref))
    # Pytree default: a no-channel forcing is structurally unchanged.
    assert OceanSurfaceForcing().S_restore_piston is None


# ---------------------------------------------------------------------------
# Dispatch / guard mutations (trace-time, the q_solar precedent)
# ---------------------------------------------------------------------------

def test_piston_without_gate_raises():
    T, S, dz_0, S_t, piston = _fields(2)
    sf = OceanSurfaceForcing(S_restore_target=S_t, S_restore_piston=piston)
    with pytest.raises(ValueError, match="salt_restore_piston is False"):
        flux_feedback_surface_forcing(T, S, dz_0, sf, FluxFeedbackConfig())


def test_gate_without_piston_raises_when_target_given():
    """The silent scalar-tau fallback (double-specification hazard) is
    rejected."""
    T, S, dz_0, S_t, _ = _fields(3)
    sf = OceanSurfaceForcing(S_restore_target=S_t)
    with pytest.raises(ValueError, match="silently fall back"):
        flux_feedback_surface_forcing(
            T, S, dz_0, sf, FluxFeedbackConfig(salt_restore_piston=True))


def test_gate_without_piston_ok_when_no_restoring():
    """No S_restore_target ⇒ no restoring at all ⇒ the gate is inert (no
    spurious raise on heat-only configurations)."""
    T, S, dz_0, _, _ = _fields(4)
    out = flux_feedback_surface_forcing(
        T, S, dz_0, OceanSurfaceForcing(),
        FluxFeedbackConfig(salt_restore_piston=True))
    assert np.abs(np.asarray(out.dS_dt)).max() == 0.0


def test_piston_without_target_raises():
    T, S, dz_0, _, piston = _fields(5)
    sf = OceanSurfaceForcing(S_restore_piston=piston)
    with pytest.raises(ValueError, match="requires S_restore_target"):
        flux_feedback_surface_forcing(
            T, S, dz_0, sf, FluxFeedbackConfig(salt_restore_piston=True))


# ---------------------------------------------------------------------------
# Ice mask + land
# ---------------------------------------------------------------------------

def test_ice_mask_zeroes_piston_restoring():
    """Veros zeroes forc_salt_surface under the ice mask
    (north_atlantic.py:342-344): cold surface + cooling heat flux ⇒ the piston
    salt rate is zeroed too."""
    T, S, dz_0, S_t, piston = _fields(6, T_surf=-2.0)   # below −1.8 °C
    q_p = jnp.full((_N_LAT, _N_LON), -50.0)             # cooling
    sf = OceanSurfaceForcing(q_prescribed=q_p, S_restore_target=S_t,
                             S_restore_piston=piston)
    out = flux_feedback_surface_forcing(
        T, S, dz_0, sf, FluxFeedbackConfig(salt_restore_piston=True))
    assert np.abs(np.asarray(out.dS_dt)).max() == 0.0
    # Warm surface: same channels, restoring active.
    T_warm = jnp.full_like(T, 10.0)
    out2 = flux_feedback_surface_forcing(
        T_warm, S, dz_0, sf, FluxFeedbackConfig(salt_restore_piston=True))
    assert np.abs(np.asarray(out2.dS_dt[..., 0])).max() > 0.0


def test_land_columns_zero():
    T, S, dz_0, S_t, piston = _fields(8)
    dz_land = np.asarray(dz_0).copy(); dz_land[0, :] = 0.0
    sf = OceanSurfaceForcing(S_restore_target=S_t, S_restore_piston=piston)
    out = flux_feedback_surface_forcing(
        T, S, jnp.asarray(dz_land), sf,
        FluxFeedbackConfig(salt_restore_piston=True))
    assert np.abs(np.asarray(out.dS_dt[0])).max() == 0.0
    assert np.abs(np.asarray(out.dS_dt[1:, :, 0])).max() > 0.0


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_grad_finite_and_analytic_through_piston():
    T, S, dz_0, S_t, piston = _fields(9)
    sf = OceanSurfaceForcing(S_restore_target=S_t, S_restore_piston=piston)
    cfg = FluxFeedbackConfig(salt_restore_piston=True)

    def loss(S_in):
        out = flux_feedback_surface_forcing(T, S_in, dz_0, sf, cfg)
        return jnp.sum(out.dS_dt ** 2)

    g = jax.grad(loss)(S)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Analytic: d/dS_surf Σ(p(S*−S)/dz)² = −2(p/dz)²(S*−S); zero below surface.
    expect = -2.0 * (np.asarray(piston) / _DZ0) ** 2 \
        * (np.asarray(S_t) - np.asarray(S[..., 0]))
    np.testing.assert_allclose(np.asarray(g[..., 0]), expect,
                               rtol=1e-12, atol=0)
    assert np.abs(np.asarray(g[..., 1:])).max() == 0.0
