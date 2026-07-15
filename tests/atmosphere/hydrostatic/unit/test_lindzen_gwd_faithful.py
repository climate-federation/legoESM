"""Faithfulness + scheme-design tests for the smoothed Lindzen-saturation orographic GWD.

Oracle: Lindzen (1981) for the SATURATION mechanism; the on-disk E3SM path ``e3sm_cam.py``
(McFarlane ``gw_oro_src`` launch + Lindzen ``gw_drag_prof`` saturation + WKB damping) as a faithful
sibling. ``lindzen.py`` is a Lindzen-style SINGLE-WAVE scheme with a McFarlane-1987 ``h²`` launch —
NOT a parameter-restricted subset of ``e3sm_cam.py`` (no ``sghmax`` cap, no WKB radiative damping).

FAITHFUL to Lindzen — the saturation core (pinned against an independent NumPy reimpl):
  * ``tau_sat = 0.5·ρ·k·|U_proj|³ / N`` (marginal-convective-instability saturation stress);
  * the saturation-breaking hypothesis: the sigmoid breaks the wave where the carried stress
    exceeds ``critical_Fr·tau_sat``, softly relaxing toward the UNSCALED ``tau_sat`` (bounded by
    ``min(tau_new, tau_carry)``), the shed stress deposited as
    ``accel = −(tau_carry−tau_new)/(ρ·dz) = +∂τ/∂z/ρ`` (a deceleration; τ ≥ 0 decreases upward).
DEPARTURE / DESIGN canaries:
  * ``critical_Fr`` is a LINEAR stress-ratio ACTIVATION threshold, NOT a Froude number (Lindzen
    scales with ``Fr_c²``); default 1.0 sets the exact-``tau_sat`` threshold but the saturation is
    SOFT (finite-sharpness sigmoid), not an exact cap, and the field name over-labels it;
  * the LAUNCH ``tau_0 = ρ_sfc·N_sfc·k·h²·U_ll`` is McFarlane (1987), not Lindzen; it has NO
    ``sghmax`` Froude cap, so it grows without bound as ``h²`` (departure from E3SM);
  * critical-level treatment is DESIGN: at ``c = 0`` (``U_proj`` reversal) the residual carried
    stress is radiated/discarded (nothing deposited on the reversed flow), NOT a deposition;
  * the antiparallel deceleration + the KE→heat closure ``dT_dt = −(u·du+v·dv)/c_pd`` are
    single-wave DESIGN choices; ``dT_dt`` is computed FROM the final tendency so
    ``c_pd·Σρ·dT·dz == eps_gwd`` holds BY CONSTRUCTION — and for a c=0 wave (zero vertical
    wave-energy flux) that IS the exact resolved-energy conservation (``conserves=["energy"]``);
  * ``crit_level_floor``/``Fr_sharpness``/``crit_level_sharpness`` are smoothing knobs; a HARD
    ``U_proj > 0`` mask forces the VECTOR sink ``u·du_dt + v·dv_dt ≤ 0`` strictly (componentwise
    ``du_dt·u`` can be >0 for oblique winds).
Dispatch hardening: an unknown GWD scheme raises ValueError.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    GravityWaveDragConfig,
    LindzenConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import get_gwd_fn
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import (
    __physics_contract__,
    _lindzen_launch_stress,
    _lindzen_saturation_stress,
    lindzen_gwd,
)

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# Limiters effectively OFF: isolates the Lindzen saturation-divergence / launch forms from the
# post-flux tndmax/umcfac magnitude caps (which break the exact stress-divergence balance).
def _no_limiter_config(**kw):
    base = dict(tndmax_per_day=1.0e12, umcfac=1.0e9, critical_Fr=1.0, Fr_sharpness=200.0)
    base.update(kw)
    return LindzenConfig(**base)


def _columns(ncol=3, nlev=16, u_sfc=25.0, u_top=25.0, v0=0.0, T_sfc=290.0, T_top=220.0,
             neutral=False):
    """Deterministic hydrostatic column; u varies linearly top->surface, v uniform.

    u_top can be < u_sfc (saturation shear) or negative (critical-level reversal). v0=0 keeps the
    source direction along +x so U_proj == u. ``neutral=True`` builds an ADIABATIC profile
    (T ∝ (p/p_sfc)^κ ⇒ θ ≈ const ⇒ N → ~0) to exercise the safe_divide(1/N) trap.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    if neutral:
        T = T_sfc * (p_full / p_full[:, -1:]) ** constants.kappa      # adiabatic ⇒ N → ~0
    else:
        T = jnp.broadcast_to(jnp.linspace(T_top, T_sfc, nlev)[None, :], (ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half_cumsum = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z_half = jnp.concatenate([z_half_cumsum, jnp.zeros((ncol, 1))], axis=1)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    # index 0 = top, -1 = surface
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.full((ncol, nlev), v0)
    lat = jnp.full((ncol,), 0.5)
    return u, v, T, p_full, p_half, z_full, z_half, rho, lat


def _run(cfg, dt=300.0, h_topo_col=None, **kw):
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(**kw)
    out = lindzen_gwd(
        u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, cfg, h_topo_col=h_topo_col
    )
    return out, (u, v, T, p_full, p_half, z_full, z_half, rho, lat)


def test_lindzen_config_defaults():
    """Canary: the Lindzen closure constants (critical_Fr=1 ⇒ threshold at exact tau_sat)."""
    c = LindzenConfig()
    assert c.critical_Fr == 1.0
    assert c.h_topo == 500.0
    assert c.k_wave == 2.0 * np.pi / 100e3
    assert c.Fr_sharpness == 20.0
    assert c.crit_level_sharpness == 10.0
    assert c.crit_level_floor == 0.5
    assert c.tndmax_per_day == 500.0
    assert c.umcfac == 0.5


# ===========================================================================
# FAITHFUL to Lindzen (1981): the saturation stress + McFarlane launch (exact helper pins)
# ===========================================================================
def test_lindzen_saturation_stress_form_exact():
    """FAITHFUL (exact to rtol 1e-12): ``_lindzen_saturation_stress`` == 0.5·ρ·k·|U_proj|³/N.

    Pins the production saturation-stress closed form directly against an INDEPENDENT NumPy reimpl
    (not the removed inline code), including the 0.1 m/s projected-wind floor and 1e-10 Pa floor.
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(u_sfc=30.0, u_top=8.0)
    N_full = brunt_vaisala_n_full(T, p_full, z_full)
    U_proj = u  # v0=0 ⇒ source direction +x
    k_wave = LindzenConfig().k_wave
    tau_sat = np.asarray(_lindzen_saturation_stress(rho, U_proj, k_wave, N_full))
    U_abs = np.clip(np.abs(np.asarray(U_proj)), 0.1, None)
    tau_sat_np = np.clip(
        0.5 * np.asarray(rho) * U_abs ** 3 * k_wave / np.asarray(N_full), 1e-10, None
    )
    assert np.allclose(tau_sat, tau_sat_np, rtol=1e-12, atol=0.0)
    # floor discriminator: a near-zero-wind column is held at the 0.1 m/s floor, not driven to ~0
    tau_floor = np.asarray(_lindzen_saturation_stress(rho, 0.0 * U_proj, k_wave, N_full))
    tau_floor_np = np.clip(
        0.5 * np.asarray(rho) * (0.1 ** 3) * k_wave / np.asarray(N_full), 1e-10, None
    )
    assert np.allclose(tau_floor, tau_floor_np, rtol=1e-12, atol=0.0)


def test_lindzen_launch_stress_form_exact():
    """FAITHFUL (exact to rtol 1e-12): ``_lindzen_launch_stress`` == clip(ρ·N·k·h²·U, 0)."""
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(u_sfc=25.0, u_top=25.0)
    N_full = brunt_vaisala_n_full(T, p_full, z_full)
    rho_sfc, N_sfc = np.asarray(rho)[:, -1], np.asarray(N_full)[:, -1]
    U_ll = np.abs(np.asarray(u)[:, -1])
    k_wave, h_sq = LindzenConfig().k_wave, 600.0 ** 2
    tau_0 = np.asarray(
        _lindzen_launch_stress(rho[:, -1], N_full[:, -1], k_wave, h_sq, jnp.asarray(U_ll))
    )
    tau_0_np = np.clip(rho_sfc * N_sfc * k_wave * h_sq * U_ll, 0.0, None)
    assert np.allclose(tau_0, tau_0_np, rtol=1e-12, atol=0.0)
    assert np.all(tau_0 > 0.0)                                          # non-vacuity


def test_lindzen_launch_stress_full_absorption():
    """FAITHFUL launch: with a near-critical top (u_top→0) the column absorbs ~all of tau_0.

    Approaching c=0 the saturation stress tau_sat ∝ |U|³ drops sharply (down to the 0.1 m/s U-floor,
    where it plateaus — not exactly 0), forcing near-total breaking so the column deposits ~all of
    the launched stress. Limiters OFF ⇒ Σ(ρ·|accel|·dz) ≈ tau_0 = ρ_sfc·N_sfc·k·h²·U_ll — pins the
    McFarlane orographic launch magnitude.
    """
    cfg = _no_limiter_config()
    h = 700.0
    out, inp = _run(cfg, h_topo_col=jnp.full((3,), h), u_sfc=25.0, u_top=0.4)
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = inp
    N_full = brunt_vaisala_n_full(T, p_full, z_full)
    u_sfc, v_sfc = np.asarray(u)[:, -1], np.asarray(v)[:, -1]
    U_ll = np.sqrt(u_sfc ** 2 + v_sfc ** 2 + 1e-10)
    tau_0 = np.asarray(rho)[:, -1] * np.asarray(N_full)[:, -1] * cfg.k_wave * (h ** 2) * U_ll

    dz = np.abs(np.asarray(z_half)[:, :-1] - np.asarray(z_half)[:, 1:])
    dz = np.clip(dz, 1.0, None)
    deposited = np.sum(np.asarray(rho) * np.abs(np.asarray(out.du_dt)) * dz, axis=1)
    assert np.all(deposited > 1e-6)                                     # non-vacuity
    # near-total absorption: within ~15% of the launched stress (residual reaches the top floor)
    assert np.all(np.abs(deposited - tau_0) / tau_0 < 0.15)


def test_lindzen_launch_scales_h_squared():
    """CANARY: tau_0 ∝ h² (McFarlane orographic launch) — 2× h_topo ⇒ ~4× total deposited stress.

    In the full-absorption regime the column deposits ~tau_0, so the h² launch law is observable as
    a factor-4 in the total column drag.
    """
    cfg = _no_limiter_config()
    kw = dict(u_sfc=25.0, u_top=0.4)
    out_1x, inp = _run(cfg, h_topo_col=jnp.full((3,), 500.0), **kw)
    out_2x, _ = _run(cfg, h_topo_col=jnp.full((3,), 1000.0), **kw)
    dz = np.abs(np.asarray(inp[6])[:, :-1] - np.asarray(inp[6])[:, 1:])
    dz = np.clip(dz, 1.0, None)
    rho = np.asarray(inp[7])
    d1 = np.sum(rho * np.abs(np.asarray(out_1x.du_dt)) * dz, axis=1)
    d2 = np.sum(rho * np.abs(np.asarray(out_2x.du_dt)) * dz, axis=1)
    assert np.all(d1 > 1e-6)
    ratio = d2 / d1
    assert np.all(np.abs(ratio - 4.0) < 0.4)                           # ~4× (h² law)


# ===========================================================================
# DESIGN: critical-level treatment (c = 0) + sign
# ===========================================================================
def test_lindzen_critical_level_absorption():
    """DESIGN: where U_proj reverses (c=0 critical level), NO drag on the reversed flow.

    A design choice (hard U_proj>0 mask + residual radiated/discarded), NOT a faithful deposition —
    the concept of critical-level absorption is physical, but this implementation under-deposits.

    u goes +20 (surface) -> −8 (top): the source-projected wind crosses zero mid-column. The hard
    U_proj>0 mask makes du_dt EXACTLY zero in every reversed layer (no spurious acceleration).
    """
    out, inp = _run(LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=20.0, u_top=-8.0)
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    reversed_layer = u <= 0.0
    assert reversed_layer.any()                                        # non-vacuity: crit level
    assert np.allclose(du[reversed_layer], 0.0, atol=1e-14)            # zero drag, reversed flow
    # non-vacuity: real drag DOES act on the positive-U side (else zero-everywhere would pass)
    assert np.any(np.abs(du[u > 0.0]) > 1e-10)
    # and the applied drag is a vector sink (u*du + v*dv <= 0) everywhere
    proj = u * du + v * dv
    assert np.all(proj <= 1e-14)


def test_lindzen_sign_and_sink():
    """DESIGN: the VECTOR sink u·du_dt + v·dv_dt ≤ 0 at every level and eps_gwd ≥ 0.

    Uses an OBLIQUE wind (v0≠0) so the test genuinely covers the vector invariant — only the
    projection along the source-wind direction is guaranteed ≤ 0, NOT componentwise du_dt·u.
    """
    out, inp = _run(
        LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=25.0, u_top=6.0, v0=9.0
    )
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    proj = u * du + v * dv                                             # vector projection
    assert np.all(proj <= 1e-14)
    assert np.any(proj < -1e-12)                                       # non-vacuity
    assert np.all(np.asarray(out.eps_gwd) >= 0.0)
    assert np.any(np.asarray(out.eps_gwd) > 1e-12)


# ===========================================================================
# KE->heat closure = the pointwise energy conservation (c=0 wave) + rest state
# ===========================================================================
def test_lindzen_energy_closure_ke_to_heat():
    """c_pd·Σρ·dT_dt·dz == eps_gwd ≥ 0 — the exact energy closure (conserves=["energy"]).

    ``dT_dt`` is computed FROM the final tendency, so the identity holds BY CONSTRUCTION; for a
    c=0 wave (zero vertical wave-energy flux, F_E = c·F_momentum = 0) that by-construction closure
    IS the exact resolved KE+internal energy conservation the contract declares. This is a
    regression pin of that closure against accidental desync, not an independent re-derivation.
    """
    out, inp = _run(LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=25.0, u_top=6.0)
    rho, z_half = np.asarray(inp[7]), np.asarray(inp[6])
    dz = np.clip(np.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    col_heat = constants.c_pd * np.sum(rho * np.asarray(out.dT_dt) * dz, axis=1)
    eps = np.asarray(out.eps_gwd)
    assert np.any(eps > 1e-12)
    assert np.allclose(col_heat, eps, rtol=1e-9, atol=1e-12)


def test_lindzen_conserves_energy():
    """``conserves == ["energy"]`` — a STATIONARY orographic wave (c=0) carries ZERO vertical
    wave-energy flux (F_E = c·F_momentum = 0, Eliassen–Palm), so all mean-flow KE removed is
    returned LOCALLY as heat and resolved KE+internal energy is conserved pointwise (see the
    KE→heat closure test above).

    The discriminator is "does the launched wave carry vertical energy flux?": c=0 orographic
    (this scheme, ``mcfarlane`` #1045, ``rayleigh`` direct-drag) ⇒ ``["energy"]``; c≠0 launched
    spectra (``hines``, ``prognostic_spectral``) whose waves carry energy the limiter discards ⇒
    ``["none"]``. MOMENTUM is not conserved (external source; critical-level radiation + limiter)
    — that breaks MOMENTUM, not energy, closure. (Corrects the prior ``["none"]`` ruling, which
    over-applied the hines c≠0 precedent to this c=0 scheme — codex #1045.)
    """
    assert __physics_contract__["conserves"] == ["energy"]


def test_lindzen_frictional_heating_form():
    """DESIGN (regression pin): dT_dt == −(u·du_dt + v·dv_dt)/c_pd."""
    out, inp = _run(
        LindzenConfig(), h_topo_col=jnp.full((3,), 800.0), u_sfc=25.0, u_top=6.0, v0=3.0
    )
    u, v = np.asarray(inp[0]), np.asarray(inp[1])
    assert np.any(np.abs(np.asarray(out.dT_dt)) > 1e-12)
    dT_exp = -(u * np.asarray(out.du_dt) + v * np.asarray(out.dv_dt)) / constants.c_pd
    assert np.allclose(np.asarray(out.dT_dt), dT_exp, rtol=1e-12, atol=0.0)


def test_lindzen_rest_state_zero_orography():
    """DESIGN idealized: zero subgrid orography (h=0) ⇒ zero launch ⇒ zero tendency."""
    out, _ = _run(LindzenConfig(), h_topo_col=jnp.zeros((3,)), u_sfc=25.0, u_top=6.0)
    assert np.allclose(np.asarray(out.du_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dv_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.dT_dt), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(out.eps_gwd), 0.0, atol=1e-12)


# ===========================================================================
# CANARIES: saturation threshold + departures
# ===========================================================================
def test_lindzen_critical_Fr_gates_breaking():
    """CANARY: critical_Fr gates saturation breaking — critical_Fr → ∞ ⇒ the wave never breaks.

    In a column with NO critical level (u > 0 throughout), the only stress sink is saturation
    breaking. critical_Fr = 1 deposits real drag; a huge critical_Fr (break only above
    critical_Fr·tau_sat, never reached) transmits the wave and deposits ≈ 0 — proving the
    saturation stress-ratio threshold, not the launch, gates the deposition.
    """
    kw = dict(h_topo_col=jnp.full((3,), 700.0), u_sfc=25.0, u_top=8.0)
    out_on, inp = _run(_no_limiter_config(critical_Fr=1.0), **kw)
    out_off, _ = _run(_no_limiter_config(critical_Fr=1.0e6), **kw)
    dz = np.clip(np.abs(np.asarray(inp[6])[:, :-1] - np.asarray(inp[6])[:, 1:]), 1.0, None)
    rho = np.asarray(inp[7])
    d_on = np.sum(rho * np.abs(np.asarray(out_on.du_dt)) * dz, axis=1)
    d_off = np.sum(rho * np.abs(np.asarray(out_off.du_dt)) * dz, axis=1)
    assert np.all(d_on > 1e-4)                                         # breaking deposits real drag
    assert np.all(d_off < 1e-6 * d_on)                                # unreachable ⇒ ~no drag


def test_lindzen_no_sghmax_cap_departure():
    """DEPARTURE canary: the launch has NO McFarlane sghmax Froude cap ⇒ tau_0 grows as h².

    10× h_topo ⇒ ~100× total deposited stress (unbounded in h), unlike E3SM's capped source.
    """
    cfg = _no_limiter_config()
    kw = dict(u_sfc=25.0, u_top=0.4)
    out_1x, inp = _run(cfg, h_topo_col=jnp.full((3,), 200.0), **kw)
    out_10x, _ = _run(cfg, h_topo_col=jnp.full((3,), 2000.0), **kw)
    dz = np.clip(np.abs(np.asarray(inp[6])[:, :-1] - np.asarray(inp[6])[:, 1:]), 1.0, None)
    rho = np.asarray(inp[7])
    d1 = np.sum(rho * np.abs(np.asarray(out_1x.du_dt)) * dz, axis=1)
    d10 = np.sum(rho * np.abs(np.asarray(out_10x.du_dt)) * dz, axis=1)
    assert np.all(d1 > 1e-6)
    assert np.all(d10 / d1 > 50.0)                                     # ~100x (h^2, no cap)


@pytest.mark.parametrize(
    "neutral,label",
    [(False, "stable + zero-top-wind (U->0 trap)"),
     (True, "near-neutral (N->0 trap: safe_divide(1,N) exercised)")],
)
def test_lindzen_ad_safe_at_traps(neutral, label):
    """NUMERICS: grad of column drag wrt a wind-scale is finite at the small-N and U→0 traps.

    The adiabatic column drives N down to its brunt floor (sqrt(1e-8)=1e-4), a large-but-finite
    1/N cotangent (the ordinary safe_divide branch, since 1e-4 > eps=1e-6); the stable column with
    u_top=0 exercises the |U_proj| floor / c=0 critical-level trap. (The masked safe_divide branch,
    N < eps, is only reachable by a direct helper call — see the next test.)
    """
    u, v, T, p_full, p_half, z_full, z_half, rho, lat = _columns(
        u_sfc=20.0, u_top=0.0, neutral=neutral
    )
    N_full = np.asarray(brunt_vaisala_n_full(T, p_full, z_full))
    if neutral:
        assert np.max(N_full) < 5e-3                     # near-neutral: small-N (1/N) trap stressed
    cfg = LindzenConfig()
    h_col = jnp.full((3,), 600.0)

    def col_drag(scale):
        out = lindzen_gwd(
            u * scale, v, T, p_full, p_half, z_full, z_half, rho, lat, 300.0, cfg,
            h_topo_col=h_col,
        )
        return jnp.sum(out.du_dt ** 2)

    g = jax.grad(col_drag)(1.0)
    assert np.isfinite(float(g))


def test_lindzen_saturation_stress_ad_safe_below_eps():
    """NUMERICS: _lindzen_saturation_stress is AD-safe when N < the safe_divide eps (1e-6).

    In-scheme N is floored at 1e-4 by brunt_vaisala, so this MASKED branch is only reachable by a
    direct call. With N=1e-8 (< eps) safe_divide returns fill=0, so tau_sat collapses to the 1e-10
    floor (NOT the raw 0.5·ρ·U³·k/1e-8 blow-up) and the grad wrt U is finite — proving the guard,
    not the raw 1/N, runs.
    """
    rho = jnp.full((2, 4), 1.0)
    U_proj = jnp.full((2, 4), 15.0)
    N_tiny = jnp.full((2, 4), 1e-8)                       # below safe_divide eps=1e-6 -> masked
    k_wave = LindzenConfig().k_wave
    val = np.asarray(_lindzen_saturation_stress(rho, U_proj, k_wave, N_tiny))
    assert np.allclose(val, 1e-10)                        # masked (fill=0) -> floor, not blow-up

    def s(scale):
        return jnp.sum(_lindzen_saturation_stress(rho, U_proj * scale, k_wave, N_tiny))
    assert np.isfinite(float(jax.grad(s)(1.0)))


def test_dispatch_unknown_gwd_raises():
    """Dispatch hardening: an unknown GWD scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown GWD scheme"):
        get_gwd_fn(GravityWaveDragConfig(scheme="not_a_scheme"))
