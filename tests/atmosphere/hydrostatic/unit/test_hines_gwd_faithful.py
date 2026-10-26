"""Scheme-level tests for the Hines-INSPIRED bulk gravity-wave-drag closure.

This is a Hines-INSPIRED bulk heuristic, NOT a faithful Hines (1997) Doppler-spread
parameterization: the Doppler-spread SPECTRUM (an azimuthal + vertical-wavenumber spectrum whose
cutoff evolves with the background wind) — the actual Hines mechanism — is ABSENT. One bulk rms
amplitude is propagated (WKB 1/√ρ growth) and saturated against a scalar scale N/m_*; the wind
enters ONLY as the antiparallel sink direction and the tendency limiter.

The only ingredient CLAIMED AS FAITHFUL to Hines is the undamped-WKB amplitude scaling (1/√ρ). The
scalar saturation scale σ_sat=N/m_* is Hines-INSPIRED (a fixed-cutoff scalar surrogate of Hines'
velocity-scale relation) but NOT faithful — it drops Hines' wind-rms + background-wind cutoff
couplings. Everything else pinned here is a SCHEME DESIGN CONTRACT, not a Hines property — the tests
are labeled accordingly.

DESIGN CONTRACTS (this scheme's own invariants, pinned exactly):
  * KE→heat energy closure: ``c_pd·∑ρ·dT_dt·dz == eps_gwd ≥ 0`` and
    ``dT_dt = −(u·du_dt+v·dv_dt)/c_pd`` (a design choice — all mean-flow KE loss is heated back;
    Hines instead distinguishes wave-energy vs mean-flow-KE dissipation, Becker & McLandress 2009);
  * antiparallel sink: ``du_dt·u + dv_dt·v ≤ 0`` (a bulk closure, NOT a Hines property — Hines
    deposits the vector momentum of an anisotropic spectrum, not necessarily antiparallel to U);
  * rest state (u=v=0) ⇒ zero RETURNED tendency (via the projection; the packet may still
    saturate internally);
  * the tendency-magnitude limiter ``√(du_dt²+dv_dt²) ≤ min(umcfac·U/dt, tndmax)``.
CANARIES (this column / this scheme, NOT general theorems):
  * the bulk saturation scale N/m_* is live (a larger m_* ⇒ lower scale ⇒ more column drag here);
  * the bulk saturation makes the drag SUBLINEAR in the launch amplitude (a cap effect, NOT a
    spectrum discriminator — a finite Hines spectrum can also be sublinear).
Dispatch hardening: an unknown GWD scheme raises ValueError.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    HinesConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.integration import get_gwd_fn

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _columns(ncol=3, nlev=12, u0=12.0, v0=5.0):
    """A deterministic hydrostatic column: T 220→290 K top→surface, uniform (u0, v0) wind."""
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = jnp.broadcast_to(jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half_cumsum = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    u = jnp.full((ncol, nlev), u0)
    v = jnp.full((ncol, nlev), v0)
    lat = jnp.full((ncol,), 0.5)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def _run(cfg, dt=300.0, **kw):
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(**kw)
    out = hines_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, cfg)
    return out, (u, v, T, p_full, p_half, z_full, z_half, rho, lat)


def test_hines_config_defaults():
    """Canary: the Hines-inspired closure constants."""
    c = HinesConfig()
    assert c.m_star == 2.0 * math.pi / 2e3      # characteristic vertical wavenumber [1/m]
    assert c.total_rms_wind == 2.0              # launch rms wave wind [m/s]
    assert c.Fmax == 0.1                        # saturation stress cap [Pa]
    assert c.doppler_sharpness == 50.0          # sigmoid saturation sharpness
    assert c.tndmax_per_day == 400.0            # |du/dt| ceiling [m/s/day] (CAM-provenance tndmax)
    assert c.umcfac == 0.5                      # max wind fraction removed / step (CAM umcfac)
    assert c.U_mag_floor == 0.1


# ===========================================================================
# SCHEME DESIGN CONTRACTS (this scheme's own invariants — not Hines physics)
# ===========================================================================
def test_hines_energy_closure_ke_to_heat():
    """DESIGN (exact): c_pd·∑ρ·dT_dt·dz == eps_gwd ≥ 0 (all mean-flow KE loss heated back).

    This is the scheme's exact KE→heat closure, NOT a Hines faithfulness property.
    """
    out, inp = _run(HinesConfig())
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = inp
    dz = np.abs(np.asarray(z_half)[:, :-1] - np.asarray(z_half)[:, 1:])
    dz = np.clip(dz, 1.0, None)
    col_heat = constants.c_pd * np.sum(np.asarray(rho) * np.asarray(out.dT_dt) * dz, axis=1)
    eps = np.asarray(out.eps_gwd)
    assert np.all(eps >= 0.0)
    assert np.any(eps > 1e-12)                          # non-vacuity: drag actually acts
    assert np.allclose(col_heat, eps, rtol=1e-9, atol=1e-12)


def test_hines_frictional_heating_form():
    """DESIGN (regression pin): dT_dt = −(u·du_dt + v·dv_dt)/c_pd (the exact code closure form)."""
    out, inp = _run(HinesConfig())
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    assert np.any(np.abs(np.asarray(out.dT_dt)) > 1e-12)   # non-vacuity: real heating
    dT_exp = -(u * np.asarray(out.du_dt) + v * np.asarray(out.dv_dt)) / constants.c_pd
    assert np.allclose(np.asarray(out.dT_dt), dT_exp, rtol=1e-12, atol=0.0)


def test_hines_sign_antiparallel_sink():
    """DESIGN: the drag is an antiparallel sink — du_dt·u + dv_dt·v ≤ 0 at every level.

    A bulk-closure choice (force antiparallel to the local wind), NOT a Hines property.
    """
    out, inp = _run(HinesConfig())
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    proj = u * np.asarray(out.du_dt) + v * np.asarray(out.dv_dt)
    assert np.all(proj <= 1e-12)
    assert np.any(proj < -1e-12)                        # non-vacuity: real deceleration


def test_hines_rest_state_zero_tendency():
    """DESIGN idealized: a resting column (u=v=0) has zero RETURNED tendency (via projection)."""
    out, _ = _run(HinesConfig(), u0=0.0, v0=0.0)
    assert np.allclose(np.asarray(out.du_dt), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(out.dv_dt), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(out.dT_dt), 0.0, atol=1e-12)


def test_hines_tendency_magnitude_limiter():
    """DESIGN: √(du_dt²+dv_dt²) ≤ min(umcfac·U/dt, tndmax) — the code's magnitude limiter."""
    cfg = HinesConfig()
    dt = 300.0
    out, inp = _run(cfg, dt=dt)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    U_mag = np.sqrt(u ** 2 + v ** 2 + 1e-10)
    tndmax = cfg.tndmax_per_day / 86400.0
    cap = np.minimum(cfg.umcfac * U_mag / dt, tndmax)
    accel_mag = np.sqrt(np.asarray(out.du_dt) ** 2 + np.asarray(out.dv_dt) ** 2)
    assert np.any(accel_mag > 1e-12)                    # non-vacuity: real drag
    assert np.all(accel_mag <= cap + 1e-12)


# ===========================================================================
# CANARIES (this column / this scheme — NOT general theorems or Hines properties)
# ===========================================================================
def test_hines_saturation_scale_responds_to_m_star():
    """CANARY: the bulk saturation scale N/m_* is live — a larger m_* ⇒ more column drag here.

    A monotonicity canary for THIS column (not a general theorem: the scan carry, Fmax cap, and
    tendency limiter can break monotonicity in other configs); it does NOT observe σ_sat directly.
    """
    base = HinesConfig()
    hi_mstar = HinesConfig(m_star=2.0 * base.m_star)     # halves the N/m_* scale
    out_base, _ = _run(base)
    out_hi, _ = _run(hi_mstar)
    assert np.any(np.asarray(out_base.eps_gwd) > 1e-12)  # non-vacuity: baseline drag
    assert np.all(np.asarray(out_hi.eps_gwd) >= np.asarray(out_base.eps_gwd) - 1e-9)
    assert np.any(np.asarray(out_hi.eps_gwd) > np.asarray(out_base.eps_gwd) + 1e-9)


def test_hines_drag_sublinear_in_launch_amplitude():
    """CANARY: the bulk saturation makes the drag SUBLINEAR in the launch amplitude.

    Doubling ``total_rms_wind`` gives < 2× the column drag — a cap/saturation effect of THIS bulk
    scheme. (NOT a bulk-vs-spectrum discriminator: a finite Hines spectrum with a moving cutoff
    can also be sublinear in the source amplitude.)  Amplitudes 4 -> 8 m/s: with the 700 hPa
    launch the 2 m/s wave on this 12-level column is still below saturation (drag ∝ amplitude²).
    """
    out_1x, _ = _run(HinesConfig(total_rms_wind=4.0))
    out_2x, _ = _run(HinesConfig(total_rms_wind=8.0))
    eps_1x = np.asarray(out_1x.eps_gwd)
    eps_2x = np.asarray(out_2x.eps_gwd)
    assert np.any(eps_1x > 1e-12)                       # non-vacuity: baseline drag
    assert np.all(eps_2x >= eps_1x - 1e-9)              # more launch amplitude ⇒ ≥ drag
    assert np.all(eps_2x < 2.0 * eps_1x)               # ...but strongly sub-linear (cap binds)


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))
