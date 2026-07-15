"""Faithfulness tests for the Rayleigh-friction GWD scheme.

Oracle: Held & Suarez (1994) for the low-level sigma-drag Rayleigh friction. The BL form matches the
model's own ``held_suarez.py`` (``k_v = k_f·max(0,(σ−σ_b)/(1−σ_b))``, ``du_dt=−k_v·u``); the config
defaults are cross-checked by importing ``held_suarez.K_F``/``SIGMA_B`` directly.

FAITHFUL to HS94 (the MOMENTUM tendency): the BL drag ``k_bl = k_max·max(0,(σ−σ_b)/(1−σ_b))`` on
(u,v), defaults ``k_max = 1/day = k_f`` and ``σ_b = 0.7``, pinned to rtol 1e-12 vs an independent
NumPy reimpl. (The frictional HEATING is a departure — HS94 has no KE-compensating heating term.)
ADDITIONS / DEPARTURES (canaries):
  * the upper sin² SPONGE is NOT HS94 — a numerical sponge-layer damping active only where
    ``σ < sponge_top``; it HEATS the removed KE (energy-conserving), unlike a physical radiation
    sponge that removes GW energy from the domain;
  * ``conserves = ["energy"]`` is correct (no external launched wave — a direct resolved KE→heat
    conversion; contrast hines/lindzen = ["none"]); the KE→heat closure is definitional.
Dispatch hardening: an unknown GWD scheme raises ValueError.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.forcing.idealized.held_suarez import K_F, SIGMA_B
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    RayleighConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import get_gwd_fn
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _columns(ncol=3, nlev=20, u0=18.0, v0=7.0, p_top=300.0):
    """Deterministic column with sigma spanning the sponge (σ<0.02) to the BL (σ→1).

    Log-spaced pressure gives fine top resolution so the sponge region (σ < sponge_top = 0.02) is
    actually resolved.
    """
    p_half = jnp.broadcast_to(
        jnp.exp(jnp.linspace(jnp.log(p_top), jnp.log(1.0e5), nlev + 1))[None, :], (ncol, nlev + 1)
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


def _run(cfg, p_top=300.0, **kw):
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(p_top=p_top, **kw)
    out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)
    return out, (u, v, T, p_full, p_half, z_full, z_half, rho, lat)


def _k_bl_np(sigma, k_max, sigma_b):
    return k_max * np.clip((sigma - sigma_b) / (1.0 - sigma_b), 0.0, 1.0)


def _k_sponge_np(sigma, sponge_k, sponge_top):
    arg = np.clip((sponge_top - sigma) / sponge_top, 0.0, 1.0)
    return sponge_k * np.sin(0.5 * np.pi * arg) ** 2


def _sigma_np(inp):
    p_full, p_half = np.asarray(inp[3]), np.asarray(inp[4])
    return p_full / np.clip(p_half[:, -1:], 1.0, None)


def test_rayleigh_config_defaults_are_hs94():
    """Canary + cross-check: default k_max/σ_b EQUAL the model's own held_suarez K_F/SIGMA_B.

    Importing held_suarez.K_F/SIGMA_B and asserting equality is a real cross-module consistency
    check that this scheme's BL drag uses the same HS94 constants as the HS94 forcing.
    """
    c = RayleighConfig()
    assert c.k_max == K_F                       # == held_suarez.K_F (HS94 k_f = 1/day)
    assert c.sigma_b == SIGMA_B                 # == held_suarez.SIGMA_B (HS94 sigma_b = 0.7)
    assert c.k_max == 1.0 / 86400.0
    assert c.sigma_b == 0.7
    assert c.sponge_top == 0.02
    assert c.sponge_k == 1.0 / (0.5 * 86400.0)  # 2/day


def test_rayleigh_full_drag_form_exact():
    """IMPLEMENTATION pin (exact rtol 1e-12): du_dt/dv_dt == −(k_bl + k_sponge)·(u,v) vs NumPy.

    Pins the FULL closed form including the non-HS94 sponge (a regression pin, NOT a HS94-faithful
    claim — the HS94-faithful part is the BL-only test test_rayleigh_bl_matches_hs94_exactly).
    """
    cfg = RayleighConfig()
    out, inp = _run(cfg)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    sigma = _sigma_np(inp)
    k_drag = _k_bl_np(sigma, cfg.k_max, cfg.sigma_b) + _k_sponge_np(
        sigma, cfg.sponge_k, cfg.sponge_top
    )
    assert np.allclose(np.asarray(out.du_dt), -k_drag * u, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(out.dv_dt), -k_drag * v, rtol=1e-12, atol=0.0)
    assert np.any(np.abs(np.asarray(out.du_dt)) > 1e-12)   # non-vacuity


def test_rayleigh_bl_matches_hs94_exactly():
    """FAITHFUL: in a BL-only column (all σ > sponge_top) du_dt == −k_max·max(0,(σ−σ_b)/(1−σ_b))·u.

    The sponge is OFF (σ > sponge_top ⇒ sponge_arg = 0), so the tendency is exactly HS94's
    low-level Rayleigh friction.
    """
    cfg = RayleighConfig()
    out, inp = _run(cfg, p_top=3000.0)                     # σ_top ≈ 0.03 > sponge_top=0.02
    u = np.asarray(inp[0])
    sigma = _sigma_np(inp)
    assert np.all(sigma > cfg.sponge_top)                  # sponge genuinely off everywhere
    k_bl = _k_bl_np(sigma, cfg.k_max, cfg.sigma_b)
    assert np.allclose(np.asarray(out.du_dt), -k_bl * u, rtol=1e-12, atol=0.0)
    # the HS94 ramp: zero above the BL (σ ≤ σ_b), rising to k_max at the surface
    assert np.any(k_bl == 0.0) and np.any(k_bl > 0.0)      # non-vacuity: ramp actually varies


def test_rayleigh_sponge_sin2_profile():
    """DEPARTURE canary: the upper sponge is a sin² taper active ONLY where σ < sponge_top.

    In the sponge top (σ < sponge_top and σ < σ_b so k_bl = 0) the drag is exactly k_sponge, and
    it is identically zero below the sponge (σ ≥ sponge_top) — a non-HS94 numerical addition.
    """
    cfg = RayleighConfig()
    out, inp = _run(cfg, p_top=100.0)                      # reaches σ ≈ 0.001 (deep into sponge)
    u = np.asarray(inp[0])
    sigma = _sigma_np(inp)
    k_sponge = _k_sponge_np(sigma, cfg.sponge_k, cfg.sponge_top)
    in_sponge = sigma < cfg.sponge_top
    assert in_sponge.any()                                 # non-vacuity: reaches the sponge
    # below the sponge the sponge coefficient is exactly zero
    assert np.allclose(k_sponge[sigma >= cfg.sponge_top], 0.0, atol=0.0)
    # in the sponge-only top (also below σ_b ⇒ k_bl=0), du_dt == −k_sponge·u exactly
    sponge_only = in_sponge & (sigma < cfg.sigma_b)
    assert sponge_only.any()
    du = np.asarray(out.du_dt)
    assert np.allclose(du[sponge_only], (-k_sponge * u)[sponge_only], rtol=1e-12, atol=0.0)
    assert np.any(k_sponge[in_sponge] > 0.0)               # non-vacuity: sponge actually acts


def test_rayleigh_rest_state_zero():
    """Idealized: a resting column (u=v=0) has zero tendency."""
    out, _ = _run(RayleighConfig(), u0=0.0, v0=0.0)
    assert np.allclose(np.asarray(out.du_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dv_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dT_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.eps_gwd), 0.0, atol=1e-12)


def test_rayleigh_sign_is_a_sink():
    """Sign: du_dt·u ≤ 0 and dv_dt·v ≤ 0 componentwise (isotropic scalar drag) and eps_gwd ≥ 0."""
    out, inp = _run(RayleighConfig())
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    assert np.all(u * np.asarray(out.du_dt) <= 1e-14)
    assert np.all(v * np.asarray(out.dv_dt) <= 1e-14)
    assert np.any(u * np.asarray(out.du_dt) < -1e-12)      # non-vacuity
    assert np.all(np.asarray(out.eps_gwd) >= 0.0)
    assert np.any(np.asarray(out.eps_gwd) > 1e-12)


def test_rayleigh_energy_closure_ke_to_heat():
    """DESIGN (definitional): c_pd·Σρ·dT_dt·dz == eps_gwd ≥ 0 (KE→heat closure).

    Definitional (dT_dt is defined FROM the tendency), but here — with NO external wave source — it
    IS the full resolved energy budget, which is why conserves=["energy"] is correct for Rayleigh.
    """
    out, inp = _run(RayleighConfig())
    rho, z_half = np.asarray(inp[7]), np.asarray(inp[6])
    dz = np.abs(z_half[:, :-1] - z_half[:, 1:])
    col_heat = constants.c_pd * np.sum(rho * np.asarray(out.dT_dt) * dz, axis=1)
    eps = np.asarray(out.eps_gwd)
    assert np.any(eps > 1e-12)
    assert np.allclose(col_heat, eps, rtol=1e-9, atol=1e-12)


def test_rayleigh_frictional_heating_form():
    """DESIGN (regression pin): dT_dt == −(u·du_dt + v·dv_dt)/c_pd."""
    out, inp = _run(RayleighConfig())
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    assert np.any(np.abs(np.asarray(out.dT_dt)) > 1e-12)
    dT_exp = -(u * np.asarray(out.du_dt) + v * np.asarray(out.dv_dt)) / constants.c_pd
    assert np.allclose(np.asarray(out.dT_dt), dT_exp, rtol=1e-12, atol=0.0)


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))
