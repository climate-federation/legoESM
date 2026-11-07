"""Scheme-level tests for the Hines (1997) Doppler-spread GWD (LMDZ r5891 port).

The oracle match lives in ``test_hines_lmdz_oracle.py``.  Pinned here:
  * config defaults = LMDZ ``hines_setup`` / driver values;
  * KE->heat energy closure ``c_pd*sum(rho*dT_dt*dz) == eps_gwd`` and
    ``dT_dt = -(u*du_dt + v*dv_dt)/c_pd`` (legoESM closure; LMDZ returns no heating).
    The sign of ``eps_gwd`` is NOT fixed: the Hines drag is not antiparallel to the
    local wind (column momentum is conserved instead), so the mean flow can gain KE;
  * the 12-level resting column carries LMDZ's small drag (its |u| < umin wind floor
    is sign-blind), bounded here;
Dispatch hardening: an unknown GWD scheme raises ValueError.
"""

from __future__ import annotations

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
    """Canary: LMDZ r5891 hines_setup / hines_gwd driver values."""
    c = HinesConfig()
    assert (c.total_rms_wind, c.f1, c.f2, c.f3) == (1.0, 1.5, 0.3, 1.0)
    assert (c.kstar_min, c.kstar_max) == (1.0e-5, 1.0e-4)
    assert c.smooth_sharpness == 100.0 and c.launch_p == 7.0e4


# ===========================================================================
# SCHEME DESIGN CONTRACTS (this scheme's own invariants — not Hines physics)
# ===========================================================================
def test_hines_energy_closure_ke_to_heat():
    """DESIGN (exact): c_pd*sum(rho*dT_dt*dz) == eps_gwd (mean-flow KE change heated back)."""
    out, inp = _run(HinesConfig())
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = inp
    dz = np.abs(np.asarray(z_half)[:, :-1] - np.asarray(z_half)[:, 1:])
    col_heat = constants.c_pd * np.sum(np.asarray(rho) * np.asarray(out.dT_dt) * dz, axis=1)
    eps = np.asarray(out.eps_gwd)
    assert np.any(np.abs(eps) > 1e-12)                  # non-vacuity: drag actually acts
    assert np.allclose(col_heat, eps, rtol=1e-9, atol=1e-12)


def test_hines_frictional_heating_form():
    """DESIGN (regression pin): dT_dt = −(u·du_dt + v·dv_dt)/c_pd (the exact code closure form)."""
    out, inp = _run(HinesConfig())
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    assert np.any(np.abs(np.asarray(out.dT_dt)) > 1e-12)   # non-vacuity: real heating
    dT_exp = -(u * np.asarray(out.du_dt) + v * np.asarray(out.dv_dt)) / constants.c_pd
    assert np.allclose(np.asarray(out.dT_dt), dT_exp, rtol=1e-12, atol=0.0)


def test_hines_rest_state_drag_is_small():
    """A resting column: LMDZ's sign-blind |u| < umin = 1e-3 m/s floor leaves a small
    asymmetric drag (oracle behaviour, matched in the rest_zero_wind oracle column)."""
    out, _ = _run(HinesConfig(), u0=0.0, v0=0.0)
    assert np.all(np.isfinite(np.asarray(out.du_dt)))
    assert np.abs(np.asarray(out.du_dt)).max() * 86400.0 < 1.0    # < 1 m/s/day


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))
