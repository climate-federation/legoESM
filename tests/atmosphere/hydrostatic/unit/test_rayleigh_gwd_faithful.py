"""Faithfulness tests for the Rayleigh-friction GWD scheme.

Oracle: Held & Suarez (1994) for the low-level sigma-drag Rayleigh friction. The BL form matches the
model's own ``held_suarez.py`` (``k_v = k_f·max(0,(σ−σ_b)/(1−σ_b))``, ``du_dt=−k_v·u``); the config
defaults are cross-checked by importing ``held_suarez.K_F``/``SIGMA_B`` directly.

FAITHFUL to HS94 (the MOMENTUM tendency): the BL drag ``k_bl = k_max·max(0,(σ−σ_b)/(1−σ_b))`` on
(u,v), defaults ``k_max = 1/day = k_f`` and ``σ_b = 0.7``, pinned to rtol 1e-12 vs an independent
NumPy reimpl. (The frictional HEATING is a departure — HS94 has no KE-compensating heating term.)
ADDITIONS / DEPARTURES (canaries):
  * the upper SPONGE is NOT HS94 — it is the Isca v1.0 damping_driver ``rayleigh`` (Polvani &
    Kushner 2002) sponge ``sponge_k ((p_sb - p)/p_sb)^2`` for ``p < p_sb`` on levels up to the one
    whose reference pressure is nearest ``2 p_sb``, with Isca's time-centred heating (oracle:
    tests/atmosphere/test_rayleigh_sponge_isca_oracle.py);
  * ``conserves = ["energy"]`` is correct: Rayleigh is a DIRECT drag (no wave at all) — a resolved
    KE→heat conversion, both tracked. Contrast the c≠0 launched-SPECTRUM schemes hines/
    prognostic_spectral = ["none"] (propagating waves carry off vertical energy flux the limiter
    discards); the c=0 orographic mcfarlane/lindzen ALSO conserve energy (zero wave-energy flux).
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
    """Deterministic column from ``p_top`` to the BL (σ→1), log-spaced in pressure."""
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


def _k_sponge_np(p_full, sigma, sponge_k, p_sb):
    """Isca damping_driver rayleigh, hard form, with its nlev_rayfric level cap."""
    pref = sigma * constants.p_atm_std
    cap = np.argmin(np.abs(pref - 2.0 * p_sb), axis=1)[:, None]
    in_cap = np.arange(p_full.shape[1])[None, :] <= cap
    return np.where(in_cap & (p_full < p_sb), sponge_k * ((p_sb - p_full) / p_sb) ** 2, 0.0)


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
    assert c.sponge_pbottom_Pa == 50.0          # Isca sponge_pbottom / PK02 0.5 hPa
    assert c.sponge_k == 1.0 / (0.5 * 86400.0)  # 2/day
    assert c.sponge_sharpness == 100.0          # C1 cutoff width 0.01


def test_rayleigh_full_drag_form_exact():
    """IMPLEMENTATION pin (exact rtol 1e-12): du_dt/dv_dt == −(k_bl + k_sponge)·(u,v) vs NumPy.

    Pins the FULL closed form including the non-HS94 sponge (a regression pin, NOT a HS94-faithful
    claim — the HS94-faithful part is the BL-only test test_rayleigh_bl_matches_hs94_exactly).
    """
    cfg = RayleighConfig(sponge_sharpness=float("inf"))
    out, inp = _run(cfg, p_top=1.0)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    sigma = _sigma_np(inp)
    k_drag = _k_bl_np(sigma, cfg.k_max, cfg.sigma_b) + _k_sponge_np(
        np.asarray(inp[3]), sigma, cfg.sponge_k, cfg.sponge_pbottom_Pa
    )
    assert np.any(k_drag[:, :3] > 0.0)                     # non-vacuity: sponge in the column
    assert np.allclose(np.asarray(out.du_dt), -k_drag * u, rtol=1e-12, atol=0.0)
    assert np.allclose(np.asarray(out.dv_dt), -k_drag * v, rtol=1e-12, atol=0.0)
    assert np.any(np.abs(np.asarray(out.du_dt)) > 1e-12)   # non-vacuity


def test_rayleigh_bl_matches_hs94_exactly():
    """FAITHFUL: in a BL-only column (all p >= p_sb) du_dt == −k_max·max(0,(σ−σ_b)/(1−σ_b))·u.

    The sponge is OFF (p >= sponge_pbottom_Pa everywhere), so the tendency is exactly HS94's
    low-level Rayleigh friction.
    """
    cfg = RayleighConfig()
    out, inp = _run(cfg, p_top=3000.0)
    u = np.asarray(inp[0])
    sigma = _sigma_np(inp)
    assert np.all(np.asarray(inp[3]) > cfg.sponge_pbottom_Pa)  # sponge genuinely off
    k_bl = _k_bl_np(sigma, cfg.k_max, cfg.sigma_b)
    assert np.allclose(np.asarray(out.du_dt), -k_bl * u, rtol=1e-12, atol=0.0)
    # the HS94 ramp: zero above the BL (σ ≤ σ_b), rising to k_max at the surface
    assert np.any(k_bl == 0.0) and np.any(k_bl > 0.0)      # non-vacuity: ramp actually varies


def test_rayleigh_sponge_isca_profile():
    """The sponge is Isca's ``sponge_k ((p_sb - p)/p_sb)^2``, zero where p >= p_sb (hard form)."""
    cfg = RayleighConfig(sponge_sharpness=float("inf"))
    out, inp = _run(cfg, p_top=1.0)
    u, p_full = np.asarray(inp[0]), np.asarray(inp[3])
    k_sponge = _k_sponge_np(p_full, _sigma_np(inp), cfg.sponge_k, cfg.sponge_pbottom_Pa)
    in_sponge = k_sponge > 0.0
    assert in_sponge.any()                                 # non-vacuity: sponge acts
    assert np.all(k_sponge[p_full >= cfg.sponge_pbottom_Pa] == 0.0)
    du = np.asarray(out.du_dt)
    assert np.allclose(du[in_sponge], (-k_sponge * u)[in_sponge], rtol=1e-12, atol=0.0)


def test_rayleigh_sponge_level_cap_binds():
    """Isca's nlev_rayfric cap: a low-p_s column (pref = σ·PSTD) loses sponge levels below the cap."""
    cfg = RayleighConfig(k_max=0.0, sponge_sharpness=float("inf"), sponge_pbottom_Pa=50.0)
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(p_top=1.0)
    p_half, p_full = p_half * 0.3, p_full * 0.3            # p_s = 30 kPa: pref = p/0.3
    out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg)
    pf = np.asarray(p_full)
    capped = (pf < cfg.sponge_pbottom_Pa) & (np.asarray(out.du_dt) == 0.0)
    assert capped.any()                                    # non-vacuity: the cap removes levels
    k_sponge = _k_sponge_np(pf, _sigma_np((u, v, T, p_full, p_half)), cfg.sponge_k, 50.0)
    assert np.allclose(np.asarray(out.du_dt), -k_sponge * np.asarray(u), rtol=1e-12, atol=0.0)


def test_rayleigh_sponge_smooth_cutoff_gradient():
    """RULE 5: d(du_dt)/d(p_sb) with a level INSIDE the 0.01 cutoff band: AD == FD, != hard AD."""
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(p_top=1.0)
    p_in = float(p_full[0, 6])                             # ~44 Pa
    p_sb0 = 1.005 * p_in                                   # (p_sb - p)/p_sb ~ 0.005 < 0.01

    def f(p_sb, s):
        cfg = RayleighConfig(k_max=0.0, sponge_pbottom_Pa=p_sb, sponge_sharpness=s)
        return jnp.sum(rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat,
                                    300.0, cfg).du_dt[:, 6])

    g = float(jax.grad(f)(p_sb0, 100.0))
    h = 1e-6 * p_sb0
    fd = (float(f(p_sb0 + h, 100.0)) - float(f(p_sb0 - h, 100.0))) / (2 * h)
    g_hard = float(jax.grad(f)(p_sb0, float("inf")))
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) <= 1e-5 * abs(fd)
    assert abs(g - g_hard) > 1e-3 * abs(g_hard)            # the smoothing is what is tested


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
    """eps_gwd = -Σρ(u du + v dv)dz (contract); c_pd·Σρ·dT·dz = eps_gwd − dt/2·Σρ(du_sp²+dv_sp²)dz.

    The extra term is the time-centred sponge heating (Isca do_conserve_energy): the heat equals
    the KE the explicit step removes, computed here independently from (u + dt·du)².
    """
    cfg, dt = RayleighConfig(), 300.0
    out, inp = _run(cfg, p_top=1.0)
    u, v, rho, z_half = (np.asarray(inp[i]) for i in (0, 1, 7, 6))
    dz = np.abs(z_half[:, :-1] - z_half[:, 1:])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    eps = np.asarray(out.eps_gwd)
    assert np.any(eps > 1e-12)
    assert np.allclose(eps, -np.sum(rho * (u * du + v * dv) * dz, axis=1), rtol=1e-12, atol=0.0)
    sp = _run(cfg._replace(k_max=0.0), p_top=1.0)[0]
    du_sp, dv_sp = np.asarray(sp.du_dt), np.asarray(sp.dv_dt)
    assert np.any(du_sp != 0.0)
    du_bl, dv_bl = du - du_sp, dv - dv_sp
    ke_step = -(((u + dt * du_sp) ** 2 - u ** 2) + ((v + dt * dv_sp) ** 2 - v ** 2)) / (2 * dt)
    want_heat = -np.sum(rho * (u * du_bl + v * dv_bl) * dz, axis=1) + np.sum(rho * ke_step * dz, axis=1)
    col_heat = constants.c_pd * np.sum(rho * np.asarray(out.dT_dt) * dz, axis=1)
    assert np.allclose(col_heat, want_heat, rtol=1e-10, atol=0.0)
    assert np.all(col_heat < eps)                          # the dt/2 term is present


def test_rayleigh_frictional_heating_form():
    """BL: −(u·du_bl + v·dv_bl)/c_pd; sponge: Isca −((u + dt/2 du_sp)·du_sp + ...)/c_pd."""
    cfg = RayleighConfig()
    out, inp = _run(cfg, p_top=1.0)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    sp = _run(cfg._replace(k_max=0.0), p_top=1.0)[0]
    du_sp, dv_sp = np.asarray(sp.du_dt), np.asarray(sp.dv_dt)
    du_bl, dv_bl = np.asarray(out.du_dt) - du_sp, np.asarray(out.dv_dt) - dv_sp
    assert np.any(du_sp != 0.0) and np.any(du_bl != 0.0)
    dt = 300.0
    dT_exp = (-(u * du_bl + v * dv_bl)
              - ((u + 0.5 * dt * du_sp) * du_sp + (v + 0.5 * dt * dv_sp) * dv_sp)) / constants.c_pd
    assert np.allclose(np.asarray(out.dT_dt), dT_exp, rtol=1e-12, atol=0.0)


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))
