"""Scheme-level oracle-faithfulness tests for the EDMF turbulence closure.

EDMF (Siebesma, Soares & Teixeira 2007, "SST07") splits the turbulent flux into a down-gradient
eddy-diffusivity (ED) part and a nonlocal mass-flux (MF) updraft part. This implementation is an
SST07-STRUCTURED dry-CBL SINGLE-PLUME simplification — NOT a verbatim SST07, and NOT a Tan et al.
(2018) prognostic plume.

FAITHFUL to the EDMF framework (pinned here):
  * the conservative FLUX-FORM MF tendency ``∂φ/∂t = −(1/ρ)∂z[M(φ_u−φ)]`` (module
    ``_mass_flux_tendency``) — pinned against an independent NumPy reimpl on a STRETCHED grid
    (so the height-weighted interface interpolation is actually exercised, weights ≠ ½) AND its
    telescoping column conservation (mass-weighted column integral = the surface MF flux).

DEPARTURES / NUMERICS (locked + labeled):
  * the ED part is the ``tke`` MY-INSPIRED k-l closure (``Km = Ck·l·√TKE``, constant Ck/Pr_t, no
    stability functions ⇒ the DIAGNOSTIC ``Km ⊥ N²``), NOT SST07's z/z*-dependent Holtslag
    K-profile (Eq. 18-20). ``Km ⊥ N²`` holds for the instantaneous diagnostic only — N² still
    feeds ``tke_new`` and hence Km on the NEXT call;
  * the plume w-eq ``d(w²)/dz = 2(B−ε·w²)`` (β=0, b=1) is a simplification of SST07 Eq. 15
    (β=0.15, b=0.5); ``M = a_u·ρ·w_u`` is a compressible extension of SST07 ``M ≈ a_u·w_u``,
    then CFL-capped; constant ε (vs SST07 Eq. 16 ~c_ε[1/z+1/(z*−z)]); tuned surface init
    (w_u(0)=max(w_min, 2.5·u*) is a friction-velocity proxy, NOT the convective scale w*);
    the surface scalar excess is SST07 Eq. 17 (α·flux/σ_w) — exercised via the config canary
    and the live-vs-inactive MF contrast;
  * inactive-plume idealized case: no surface flux + non-buoyant column ⇒ the plume-environment
    anomaly φ_u−φ_env→0 ⇒ the scalar MF FLUX (and its tendency) vanishes — because the anomaly
    vanishes, NOT because the mass flux M→0 (M stays nonzero);
  * unknown turbulence scheme raises ValueError (dispatch hardening; sanity, not a pin).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import (
    TurbulenceConfig,
    TurbulentEDMFConfig,
)
from legoesm.atmosphere.physics.turbulence.edmf import (
    _EDMF_WSTAR_COEFF,
    _SIGMA_W_COEFF,
    _SIGMA_W_WSTAR3_COEFF,
    _mass_flux_tendency,
    _updraft_surface_excess,
    edmf_turbulence,
)
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.thermo import saturation_mixing_ratio

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _grid(ncol=2, nlev=6):
    """A FIXED (p, z) grid, decoupled from the test θ profile.

    z is set directly (linear top→surface) so changing θ (hence N²) between two states leaves
    the z geometry — and therefore the Blackadar length and the ED Km — bit-identical. Level 0
    is the top, nlev-1 the surface.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(7.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = jnp.broadcast_to(
        jnp.linspace(1500.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return p_full, p_half, z_full, z_half


def _stretched_grid(ncol=2, nlev=6):
    """A STRETCHED-in-height grid where interface heights are NOT midway between full levels.

    z_full is placed at a quadratic-in-index profile so ``z_half`` (the midpoint of z_half
    edges) is offset from ``½(z_full[k-1]+z_full[k])`` — the height-weighting interpolation
    weights are then ≠ ½, so a regression that hard-codes ½ would FAIL the flux-form pin.
    """
    idx = jnp.arange(nlev + 1)
    zt = 3000.0 * (idx / nlev) ** 1.7            # stretched half-level heights, surface→top
    z_half = jnp.broadcast_to(zt[::-1][None, :], (ncol, nlev + 1))  # top-first
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return z_full, z_half


def _state(grid, theta_top=300.0, theta_sfc=300.0, u_top=0.0, u_sfc=0.0, tke_val=0.5):
    """Build (u, v, T, q_v, tke, rho) on a fixed grid; θ linear top→surface, u linear."""
    p_full, p_half, z_full, z_half = grid
    ncol, nlev = p_full.shape
    theta = jnp.broadcast_to(
        jnp.linspace(theta_top, theta_sfc, nlev)[None, :], (ncol, nlev)
    )
    T = theta * (p_full / constants.p_ref) ** constants.kappa
    rho = p_full / (constants.R_d * T)
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.zeros((ncol, nlev))
    tke = jnp.full((ncol, nlev), tke_val)
    return u, v, T, q_v, tke, rho


def _run(cfg, grid=None, dt=300.0, no_surface_flux=False, dry_surface=False,
         **state_kw):
    grid = grid if grid is not None else _grid()
    p_full, p_half, z_full, z_half = grid
    u, v, T, q_v, tke, rho = _state(grid, **state_kw)
    theta = np.asarray(T) * (constants.p_ref / np.asarray(p_full)) ** constants.kappa
    if no_surface_flux:
        # Surface at the air value ⇒ (near) zero bulk surface heat/moisture flux.
        T_sfc = jnp.asarray(np.asarray(T)[:, -1])
        q_sfc = jnp.asarray(np.asarray(q_v)[:, -1])
    else:
        T_sfc = jnp.full((u.shape[0],), float(np.asarray(T)[0, -1]) + 2.0)
        q_sfc = (jnp.asarray(np.asarray(q_v)[:, -1]) if dry_surface
                 else saturation_mixing_ratio(T_sfc, p_half[:, -1]))
    out, tke_new = edmf_turbulence(
        u, v, T, q_v, tke, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, dt, cfg,
    )
    return out, tke_new, (u, v, T, q_v, tke, p_full, p_half, z_full, z_half, rho, theta)


def _N2_half0(inp):
    """Brunt-Väisälä N² at the top half-interface (col 0), mirroring production's N²."""
    u, v, T, q_v, tke, p_full, p_half, z_full, z_half, rho, theta = inp
    dz = np.clip(abs(np.asarray(z_full)[0, 0] - np.asarray(z_full)[0, 1]), 1.0, None)
    exner = 1.0 / np.asarray(exner_function(p_full))[0]
    thv = np.asarray(virtual_temperature(T, q_v))[0] * exner
    thv_bar = 0.5 * (thv[0] + thv[1])
    dthv_dz = (thv[0] - thv[1]) / dz
    return float(buoyancy_coefficient(jnp.clip(jnp.array(thv_bar), 1.0, None))) * dthv_dz


# ===========================================================================
# FAITHFUL to the EDMF framework (the MF flux-form transport + its conservation)
# ===========================================================================
def test_edmf_mass_flux_tendency_flux_form_stretched():
    """FAITHFUL: −(1/ρ)∂z[M(φ_u−φ)] on a STRETCHED grid vs an independent height-weighted reimpl.

    The grid is stretched so the interface heights are NOT midway between full levels ⇒ the
    interpolation weights are ≠ ½ (a regression hard-coding ½ would fail here).
    """
    ncol, nlev = 2, 6
    z_full, z_half = _stretched_grid(ncol, nlev)
    rng = np.random.default_rng(0)
    phi = jnp.asarray(rng.normal(size=(ncol, nlev)))
    phi_u = phi + jnp.asarray(rng.normal(size=(ncol, nlev)))
    M = jnp.asarray(np.abs(rng.normal(size=(ncol, nlev))))
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    rho = jnp.full((ncol, nlev), 1.1)
    got = np.asarray(_mass_flux_tendency(phi, phi_u, M, dz_layer, rho, z_full, z_half))
    # Independent NumPy reimpl of the height-weighted flux-form divergence.
    flux = np.asarray(M * (phi_u - phi))
    zf, zh = np.asarray(z_full), np.asarray(z_half)
    z_up, z_lo, z_if = zf[:, :-1], zf[:, 1:], zh[:, 1:-1]
    w = (z_up - z_if) / np.clip(z_up - z_lo, 1.0, None)
    flux_int = (1.0 - w) * flux[:, :-1] + w * flux[:, 1:]
    flux_iface = np.concatenate([np.zeros((ncol, 1)), flux_int, flux[:, -1:]], axis=1)
    dflux_dz = (flux_iface[:, :-1] - flux_iface[:, 1:]) / np.asarray(dz_layer)
    exp = -dflux_dz / np.clip(np.asarray(rho), 0.01, None)
    assert np.allclose(got, exp, rtol=1e-12, atol=0.0)
    # non-vacuity: the stretched grid genuinely has interface weights away from ½
    assert np.any(np.abs(w - 0.5) > 0.05)


def test_edmf_mass_flux_tendency_conserves():
    """FAITHFUL: the flux form telescopes — mass-weighted column integral = surface MF flux.

    ∑_k ρ_k·(∂φ/∂t)_k·dz_k = ∑_k −(F_k − F_{k+1}) = F_sfc − F_top = flux[:, -1] (F_top = 0).
    Proves the discrete MF operator conserves; NOT total EDMF conservation (ED + surface bulk
    fluxes are separate) — that is out of scope for this operator-level pin.
    """
    grid = _grid()
    p_full, p_half, z_full, z_half = grid
    ncol, nlev = p_full.shape
    rng = np.random.default_rng(1)
    phi = jnp.asarray(rng.normal(size=(ncol, nlev)))
    phi_u = phi + jnp.asarray(rng.normal(size=(ncol, nlev)))
    M = jnp.asarray(np.abs(rng.normal(size=(ncol, nlev))))
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    rho = jnp.asarray(1.0 + np.abs(rng.normal(size=(ncol, nlev))))
    tend = _mass_flux_tendency(phi, phi_u, M, dz_layer, rho, z_full, z_half)
    col_int = np.asarray(jnp.sum(rho * tend * dz_layer, axis=1))
    surface_flux = np.asarray(M * (phi_u - phi))[:, -1]   # F_sfc (F_top = 0)
    assert np.allclose(col_int, surface_flux, rtol=1e-10, atol=1e-12)
    assert np.any(np.abs(surface_flux) > 1e-6)            # non-vacuity: real transport


def test_edmf_config_defaults():
    """Canary: the EDMF closure constants (ED tke-style + MF plume knobs + w* proxy coeff)."""
    c = TurbulentEDMFConfig()
    # ED part = the tke MY-inspired k-l constants (identical to TKEConfig)
    assert c.Ck == 0.1
    assert c.Ce == 0.19
    assert c.Pr_t == 0.33
    assert c.l_mix_max == 100.0
    assert c.tke_min == 1e-6
    # MF part
    assert c.a_updraft == 0.1            # updraft area fraction
    assert c.entrainment_rate == 1e-3    # CONSTANT ε [1/m] (vs SST07 Eq. 16 ~c_ε[1/z+1/(z*−z)])
    assert c.surface_excess_prefactor == 1.0   # SST07 Eq. 17 α (LES ~1)
    assert (_SIGMA_W_COEFF, _SIGMA_W_WSTAR3_COEFF) == (1.3, 0.6)  # SST07 Eq. 7
    assert c.w_updraft_min == 0.1
    assert c.updraft_deactivation_sharpness == 20.0
    assert c.n_updrafts == 1             # single bulk plume (unused field; SST07 dry-CBL scope)
    assert _EDMF_WSTAR_COEFF == 2.5      # w_u(0) = max(w_min, 2.5·u*) — friction-velocity proxy


# ===========================================================================
# DEPARTURES / NUMERICS (ED tke-closure, live-vs-inactive MF, dispatch)
# ===========================================================================
def test_departure_edmf_ED_Km_kl_form():
    """DEPARTURE (ED): Km = Ck·l·√TKE — the tke k-l closure, NOT SST07's Holtslag K-profile.

    Pinned vs an independent reimpl (shared Blackadar ``mixing_length``); Kh = Km/Pr_t.
    """
    cfg = TurbulentEDMFConfig()
    out, _, inp = _run(cfg, tke_val=0.5)
    z_full = inp[7]
    Km = np.asarray(out.Km)
    Kh = np.asarray(out.Kh)
    l_mix = np.asarray(mixing_length(z_full, cfg.l_mix_max))
    Km_exp = cfg.Ck * l_mix * np.sqrt(0.5)
    assert np.all(Km > 0.0)
    assert np.allclose(Km, Km_exp, rtol=1e-12, atol=0.0)
    assert np.allclose(Kh, Km / cfg.Pr_t, rtol=1e-12, atol=0.0)


def test_departure_edmf_ED_Km_independent_of_stratification():
    """DEPARTURE: the DIAGNOSTIC ED Km is the tke constant-Ck closure — INDEPENDENT of N².

    On a FIXED grid with the SAME TKE, a strongly STABLE and a strongly UNSTABLE column give
    BIT-IDENTICAL diagnostic ``out.Km`` (Km = Ck·l·√TKE) — the diagnostic ED K has NO buoyancy /
    stability input at all. This pins the tke-closure DEPARTURE (SST07 instead uses a Holtslag
    ``z/z*``, ``w*``, ``u*`` K-profile, Eq. 18-20 — itself not a local-N² function, so this is a
    structural-form difference, not an N²-sensitivity one). Instantaneous diagnostic ONLY — N²
    still feeds ``tke_new`` (via the buoyancy production), so it changes Km on the NEXT call.
    """
    cfg = TurbulentEDMFConfig()
    grid = _grid()
    out_stable, _, inp_stable = _run(cfg, grid=grid, theta_top=320.0, theta_sfc=300.0)
    out_unstable, _, inp_unstable = _run(cfg, grid=grid, theta_top=290.0, theta_sfc=310.0)
    assert _N2_half0(inp_stable) > 0.0        # warm aloft ⇒ stable
    assert _N2_half0(inp_unstable) < 0.0      # warm surface ⇒ unstable
    assert np.array_equal(np.asarray(out_stable.Km), np.asarray(out_unstable.Km))


def test_edmf_mass_flux_scalar_transport_live_vs_inactive():
    """The MF scalar transport is LIVE under surface heating and OFF without it.

    SST07 Eq. 17: the updraft starts at φ(z1) + α·(w'φ')_s/σ_w.  With a heated surface the
    excess (and so the plume's heat transport) scales with α; ED diffusivities do not depend on
    α.  With no surface flux the excess is zero, the plume-environment ANOMALY φ_u−φ_env
    vanishes, and the tendency no longer depends on the plume (M stays nonzero, so the updraft
    area fraction changes M but not the result).
    """
    grid = _grid()
    out_a, _, _ = _run(TurbulentEDMFConfig(surface_excess_prefactor=1.0), grid=grid,
                       u_top=8.0, u_sfc=5.0)
    out_b, _, _ = _run(TurbulentEDMFConfig(surface_excess_prefactor=1.5), grid=grid,
                       u_top=8.0, u_sfc=5.0)
    assert np.all(np.asarray(out_a.shflx) > 1.0)              # surface genuinely heats
    assert np.allclose(np.asarray(out_a.Km), np.asarray(out_b.Km), rtol=1e-12)
    assert not np.allclose(np.asarray(out_a.dT_dt), np.asarray(out_b.dT_dt))
    # unforced: the plume carries no anomaly, so its area fraction cannot matter
    off = [_run(TurbulentEDMFConfig(a_updraft=a), grid=grid, no_surface_flux=True)[0]
           for a in (0.05, 0.3)]
    assert np.max(np.abs(np.asarray(off[0].shflx))) < 1e-6
    assert np.max(np.abs(np.asarray(off[0].lhflx))) < 1e-6
    for f in ("dT_dt", "dq_v_dt"):
        x0, x1 = np.asarray(getattr(off[0], f)), np.asarray(getattr(off[1], f))
        scale = np.max(np.abs(np.asarray(getattr(out_a, f))))   # heated-run tendency
        assert np.max(np.abs(x0 - x1)) <= 1e-9 * scale, f


def test_edmf_inactive_plume_idealized():
    """Idealized (contract): neutral, no-flux column ⇒ zero scalar MF transport, tiny tendency."""
    cfg = TurbulentEDMFConfig()
    out, _, _ = _run(cfg, theta_top=300.0, theta_sfc=300.0, u_top=0.0, u_sfc=0.0,
                     no_surface_flux=True)
    assert np.max(np.abs(np.asarray(out.shflx))) < 1e-6   # genuinely no surface heat flux
    assert np.max(np.abs(np.asarray(out.lhflx))) < 1e-6
    assert np.all(np.asarray(out.Km) >= 0.0)              # ED down-gradient
    assert np.all(np.asarray(out.Kh) >= 0.0)
    # a non-buoyant, unforced column produces a negligible MF heat/moisture tendency
    assert np.max(np.abs(np.asarray(out.dT_dt))) < 1e-4
    assert np.max(np.abs(np.asarray(out.dq_v_dt))) < 1e-6


def test_dispatch_unknown_turbulence_raises():
    """Dispatch hardening: an unknown turbulence scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown turbulence scheme"):
        get_turbulence_fn(TurbulenceConfig(scheme="not_a_scheme"))


def test_edmf_plume_surface_flux_is_part_of_the_surface_flux():
    """ED + MF at the surface equals the physical surface flux (SST07 Eq. 5).
    Heated, dry surface (no moisture flux, so no latent enthalpy term): the
    column's mass-weighted (rho*dz, the scheme's own weights) theta tendency
    equals shflx/(c_pd*Pi_1) to round-off, with an active plume (prefactor 1.5
    vs 1 changes the profile but not the column total).  Before, the plume's
    bottom flux a*rho*w*(theta_u - theta) was added on top of the surface flux."""
    totals = []
    for alpha in (1.0, 1.5):
        out, _, inp = _run(TurbulentEDMFConfig(surface_excess_prefactor=alpha),
                           grid=_grid(), dry_surface=True,
                           u_top=8.0, u_sfc=5.0)
        p_full, z_half, rho = (np.asarray(inp[5]), np.asarray(inp[8]),
                               np.asarray(inp[9]))
        assert np.max(np.abs(np.asarray(out.lhflx))) < 1e-9
        shflx = np.asarray(out.shflx)
        assert np.all(shflx > 1.0)
        exner_1 = (p_full[:, -1] / constants.p_ref) ** constants.kappa
        w = rho * np.clip(np.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
        dtheta = np.asarray(out.dT_dt) / (p_full / constants.p_ref) ** constants.kappa
        budget = (w * dtheta).sum(-1)
        expected = shflx / (constants.c_pd * exner_1)
        assert np.allclose(budget, expected, rtol=1e-9, atol=0.0), (budget, expected)
        totals.append(dtheta)
    assert not np.allclose(totals[0], totals[1])   # the plume is active


def test_updraft_surface_excess_is_sst07_eq17_with_buoyancy_wstar():
    """SST07 Eq. 17 + Eq. 7 (Holtslag & Moeng 1991), transcribed independently in NumPy.

    theta_u - theta = alpha (w'theta')_s / sigma_w, q_u - q = alpha (w'q')_s / sigma_w,
    sigma_w = max(w_min, 1.3 (u*^3 + 0.6 w*^3 z/z*)^(1/3) (1 - z/z*)^(1/2)),
    w*^3 = (g/theta_v) (w'theta_v')_s z* with the BUOYANCY flux (user decision 2026-10-16).
    Columns: dry heating, moist heating (moisture raises w*), shallow PBL (sigma_w floor),
    surface cooling (no excess)."""
    cfg = TurbulentEDMFConfig(surface_excess_prefactor=1.7)
    wth = np.array([0.1, 0.1, 0.1, -0.02])
    wq = np.array([0.0, 5e-5, 5e-5, 5e-5])
    th = np.full(4, 300.0)
    thv = th * (1.0 + (1.0 / constants.epsilon - 1.0) * 0.01)
    ust = np.array([0.3, 0.3, 0.05, 0.3])
    z1 = np.full(4, 30.0)
    zs = np.array([1000.0, 1000.0, 30.0, 1000.0])
    d_th, d_q = (np.asarray(x) for x in _updraft_surface_excess(
        jnp.asarray(wth), jnp.asarray(wq), jnp.asarray(th), jnp.asarray(thv),
        jnp.asarray(ust), jnp.asarray(z1), jnp.asarray(zs), cfg))
    wthv = wth + (constants.R_v / constants.R_d - 1.0) * th * wq
    wstar3 = constants.g / thv * np.maximum(wthv, 0.0) * zs
    r = np.minimum(z1 / zs, 1.0)
    sw = 1.3 * np.cbrt(ust ** 3 + 0.6 * wstar3 * r) * np.sqrt(np.maximum(1.0 - r, 1e-6))
    sw = np.maximum(sw, cfg.w_updraft_min)
    on = wth > 0.0
    np.testing.assert_allclose(d_th, np.where(on, 1.7 * wth / sw, 0.0), rtol=1e-12)
    np.testing.assert_allclose(d_q, np.where(on, 1.7 * wq / sw, 0.0), rtol=1e-12)
    assert d_th[1] < d_th[0]                       # moisture flux raises w*, shrinks the excess
    assert sw[2] == cfg.w_updraft_min              # PBL top at z1: the floor binds
    assert d_th[3] == 0.0 and d_q[3] == 0.0        # surface cooling: no excess


def test_edmf_plume_heat_leak_below_one_percent_of_surface_flux():
    """The MF part transports theta and converts with each layer's Exner factor, so the
    column HEAT it receives differs from the surface flux.  Cap (user decision 2026-10-16):
    under 1 % of the surface flux at the default prefactor; a heat-carrying plume is a
    follow-up.  Dry heated surface, so no latent enthalpy term."""
    out, _, inp = _run(TurbulentEDMFConfig(), grid=_grid(), dry_surface=True,
                       u_top=8.0, u_sfc=5.0)
    z_half, rho = np.asarray(inp[8]), np.asarray(inp[9])
    w = rho * np.clip(np.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)   # scheme's weights
    heat = (w * np.asarray(out.dT_dt)).sum(-1) * constants.c_pd
    shflx = np.asarray(out.shflx)
    assert np.all(shflx > 1.0)
    assert np.all(np.abs(heat - shflx) < 0.01 * shflx), (heat, shflx)
