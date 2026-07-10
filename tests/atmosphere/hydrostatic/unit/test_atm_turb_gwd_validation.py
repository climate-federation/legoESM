"""Validation suite for the atm-turb-gwd physics group.

Covers, per the physics-validator truth tiers, the named schemes in scope:
  smagorinsky, louis, holtslag_boville, ysu  (turbulence)
  mcfarlane (orographic GWD), hines (non-orographic GWD)

Truth tiers exercised here (units are covered by the constants/saturation
ratchets + static review; this file asserts the numerically-checkable tiers):

  * SIGN          — surface sensible-heat flux sign (warm surface -> +up);
                    GWD momentum sink (du/dt * u <= 0); GWD frictional
                    heating (dT/dt >= 0); GWD column dissipation eps >= 0.
  * CONSERVATION  — in a CLOSED column (zero surface fluxes) every
                    turbulence scheme conserves mass-weighted theta, total
                    water, and momentum to machine precision (the flux-form
                    invariant ``sum(rho*dz*phi)``).  The HB/YSU nonlocal
                    countergradient is divergence-form, so it redistributes
                    without changing the column total.  GWD: KE lost by the
                    mean flow equals frictional heating gained
                    (``sum(rho*cp*dT/dt*dz) == eps_gwd``).
  * DIFFERENTIABILITY — jax.grad of a scalar tendency loss w.r.t. every
                    tunable scheme parameter is finite and matches a centred
                    finite difference; jit and vmap reproduce eager outputs.
  * IDEALIZED     — Smagorinsky Lilly cutoff (Km->0 in strongly stable);
                    Louis stable suppression vs unstable enhancement;
                    HB/YSU convective K-profile peaks inside the PBL and
                    vanishes at the top; all K >= 0.

These complement (do not replace) the existing per-backend unit tests in
``test_turbulence.py`` / ``test_gravity_wave_drag.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.turbulence.config import (
    SurfaceLayerConfig,
    SmagorinskyConfig,
    LouisConfig,
    HoltslagBovilleConfig,
    YSUConfig,
)
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    McFarlaneConfig,
    HinesConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd


DT = 600.0

TURB_SCHEMES = {
    "smagorinsky": (smagorinsky_turbulence, lambda: SmagorinskyConfig(surface=SurfaceLayerConfig())),
    "louis": (louis_turbulence, lambda: LouisConfig(surface=SurfaceLayerConfig())),
    "holtslag_boville": (holtslag_boville_turbulence, lambda: HoltslagBovilleConfig(surface=SurfaceLayerConfig())),
    "ysu": (ysu_turbulence, lambda: YSUConfig(surface=SurfaceLayerConfig())),
}


# ---------------------------------------------------------------------------
# Column builders (level 0 = model top, level nlev-1 = surface)
# ---------------------------------------------------------------------------

def _base_column(ncol, nlev, theta_top, theta_sfc, u_top, u_sfc):
    p_half = jnp.broadcast_to(
        jnp.linspace(2000.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    theta = jnp.broadcast_to(jnp.linspace(theta_top, theta_sfc, nlev)[None, :], (ncol, nlev))
    exner = (p_full / constants.p_ref) ** constants.kappa
    T = theta * exner
    q_v = jnp.broadcast_to(jnp.linspace(2e-3, 12e-3, nlev)[None, :], (ncol, nlev))
    u = jnp.broadcast_to(jnp.linspace(u_top, u_sfc, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * p_mid))
    z_half = jnp.concatenate(
        [jnp.zeros((ncol, 1)), jnp.cumsum(dz[:, ::-1], axis=1)], axis=1
    )[:, ::-1]
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    return dict(
        u=u, v=v, T=T, q_v=q_v, p_full=p_full, p_half=p_half,
        z_full=z_full, z_half=z_half, rho=rho,
    )


def unstable_column(ncol=3, nlev=24):
    """Strong surface heating: shflx>0, active convective mixing."""
    d = _base_column(ncol, nlev, theta_top=320.0, theta_sfc=300.0, u_top=4.0, u_sfc=18.0)
    d["T_sfc"] = d["T"][:, -1] + 8.0
    d["q_sfc"] = saturation_mixing_ratio(d["T_sfc"], d["p_full"][:, -1])
    return d


def closed_column(ncol=2, nlev=24):
    """Calm surface (zero wind at the bottom level), T_sfc=T_bot, q_sfc=q_bot.

    All bulk surface fluxes vanish, so the only tendencies are internal
    redistribution -> the column-integrated invariants must be conserved.
    """
    d = _base_column(ncol, nlev, theta_top=320.0, theta_sfc=300.0, u_top=15.0, u_sfc=12.0)
    d["u"] = d["u"].at[:, -1].set(0.0)  # calm at surface -> |V|~0 -> fluxes~0
    d["T_sfc"] = d["T"][:, -1]
    d["q_sfc"] = d["q_v"][:, -1]
    return d


def sheared_neutral_column(ncol=2, nlev=24):
    """Near-neutral stratification + strong shear so the gradient-Ri stays
    below Pr_t -> the shear/Ri-driven schemes (Smagorinsky in particular)
    have an ACTIVE interior eddy viscosity, exercising their tunables.
    """
    # theta nearly constant (weakly stable) -> small N2; strong wind shear
    # from a deep jet -> large S2 -> Ri < 1.
    d = _base_column(ncol, nlev, theta_top=302.0, theta_sfc=300.0, u_top=40.0, u_sfc=2.0)
    d["T_sfc"] = d["T"][:, -1] + 4.0
    d["q_sfc"] = saturation_mixing_ratio(d["T_sfc"], d["p_full"][:, -1])
    return d


def gwd_breaking_column(ncol=2, nlev=40):
    """Stable stratification + strong low-level jet -> the wave breaks."""
    d = _base_column(ncol, nlev, theta_top=500.0, theta_sfc=290.0, u_top=4.0, u_sfc=30.0)
    d["lat"] = jnp.zeros(ncol)
    return d


def _run_turb(fn, cfg, d):
    return fn(
        d["u"], d["v"], d["T"], d["q_v"], d["p_full"], d["p_half"],
        d["z_full"], d["z_half"], d["T_sfc"], d["q_sfc"], d["rho"], DT, cfg,
    )


# ===========================================================================
# SIGN
# ===========================================================================

@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_sign_shflx_warm_surface_positive(name):
    fn, mk = TURB_SCHEMES[name]
    out = _run_turb(fn, mk(), unstable_column())
    # Warm surface (T_sfc > T_air) -> positive (upward) sensible heat flux.
    assert jnp.all(out.shflx > 0.0), f"{name}: warm surface gave shflx<=0"
    # Cooler surface flips the sign.
    d = unstable_column()
    d["T_sfc"] = d["T"][:, -1] - 8.0
    d["q_sfc"] = saturation_mixing_ratio(d["T_sfc"], d["p_full"][:, -1])
    out_c = _run_turb(fn, mk(), d)
    assert jnp.all(out_c.shflx < 0.0), f"{name}: cold surface gave shflx>=0"


@pytest.mark.parametrize("gwd,mk", [
    (mcfarlane_gwd, McFarlaneConfig),
    (hines_gwd, HinesConfig),
])
def test_sign_gwd_momentum_sink(gwd, mk):
    d = gwd_breaking_column()
    out = gwd(
        d["u"], d["v"], d["T"], d["p_full"], d["p_half"],
        d["z_full"], d["z_half"], d["rho"], d["lat"], DT, mk(),
    )
    # Drag opposes the wind: du/dt * u <= 0 everywhere (strict momentum sink).
    assert jnp.max(out.du_dt * d["u"]) <= 1e-9
    assert jnp.max(out.dv_dt * d["v"]) <= 1e-9
    # Frictional heating is non-negative (KE -> heat).
    assert jnp.min(out.dT_dt) >= -1e-12
    # Column dissipation is positive-definite.
    assert jnp.all(out.eps_gwd >= -1e-9)
    # And the wave actually broke (non-trivial test).
    assert jnp.any(jnp.abs(out.du_dt) > 0.0)


# ===========================================================================
# CONSERVATION
# ===========================================================================

@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_conservation_closed_column(name):
    """Closed column: mass-weighted theta, water, momentum conserved."""
    fn, mk = TURB_SCHEMES[name]
    d = closed_column()
    out = _run_turb(fn, mk(), d)
    # Surface fluxes must be ~zero (closed column precondition).
    assert jnp.allclose(out.shflx, 0.0, atol=1e-8)
    assert jnp.allclose(out.lhflx, 0.0, atol=1e-8)

    exner = (d["p_full"] / constants.p_ref) ** constants.kappa
    rdz = d["rho"] * jnp.abs(d["z_half"][:, :-1] - d["z_half"][:, 1:])

    theta0 = d["T"] / exner
    theta_new = (d["T"] + DT * out.dT_dt) / exner
    mtheta0 = jnp.sum(rdz * theta0, axis=1)
    mtheta_drift = jnp.sum(rdz * (theta_new - theta0), axis=1) / mtheta0
    assert jnp.max(jnp.abs(mtheta_drift)) < 1e-12, f"{name}: theta not conserved"

    q0 = jnp.sum(rdz * d["q_v"], axis=1)
    dq = jnp.sum(rdz * out.dq_v_dt, axis=1) * DT / q0
    assert jnp.max(jnp.abs(dq)) < 1e-12, f"{name}: water not conserved"

    du = jnp.sum(rdz * out.du_dt, axis=1) * DT
    assert jnp.max(jnp.abs(du)) < 1e-9, f"{name}: momentum not conserved"


@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_conservation_moisture_equals_surface_flux(name):
    """Open column: column water tendency == surface moisture flux."""
    fn, mk = TURB_SCHEMES[name]
    d = unstable_column()
    out = _run_turb(fn, mk(), d)
    rdz = d["rho"] * jnp.abs(d["z_half"][:, :-1] - d["z_half"][:, 1:])
    dq_int = jnp.sum(rdz * out.dq_v_dt, axis=1)
    sfc_qflux = out.lhflx / constants.L_v
    np.testing.assert_allclose(np.array(dq_int), np.array(sfc_qflux), rtol=1e-6)


@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_conservation_momentum_equals_surface_stress(name):
    fn, mk = TURB_SCHEMES[name]
    d = unstable_column()
    out = _run_turb(fn, mk(), d)
    rdz = d["rho"] * jnp.abs(d["z_half"][:, :-1] - d["z_half"][:, 1:])
    du_int = jnp.sum(rdz * out.du_dt, axis=1)
    Cd = SurfaceLayerConfig().Cd_neutral
    wind = jnp.sqrt(d["u"][:, -1] ** 2 + d["v"][:, -1] ** 2 + 1e-4)
    tau_x = -d["rho"][:, -1] * Cd * wind * d["u"][:, -1]
    np.testing.assert_allclose(np.array(du_int), np.array(tau_x), rtol=1e-6)


@pytest.mark.parametrize("gwd,mk", [
    (mcfarlane_gwd, McFarlaneConfig),
    (hines_gwd, HinesConfig),
])
def test_conservation_gwd_ke_to_heat(gwd, mk):
    """Column KE lost by the mean flow == frictional heating deposited."""
    d = gwd_breaking_column()
    out = gwd(
        d["u"], d["v"], d["T"], d["p_full"], d["p_half"],
        d["z_full"], d["z_half"], d["rho"], d["lat"], DT, mk(),
    )
    dz = jnp.abs(d["z_half"][:, :-1] - d["z_half"][:, 1:])
    heat_int = jnp.sum(d["rho"] * constants.c_pd * out.dT_dt * dz, axis=1)
    np.testing.assert_allclose(np.array(heat_int), np.array(out.eps_gwd), rtol=1e-8, atol=1e-10)


# ===========================================================================
# DIFFERENTIABILITY  (AD finite, nonzero, matches FD; jit & vmap consistent)
# ===========================================================================

TURB_GRAD_PARAMS = {
    "smagorinsky": ["C_s", "l_mix_max", "Pr_t"],
    "louis": ["b_louis", "b_heat_ratio", "l_mix_max", "Ri_crit"],
    "holtslag_boville": ["fak", "fakn", "betam", "betah", "sffrac", "Ri_crit"],
    "ysu": ["entrainment_ratio", "countergrad_coeff", "Pr_t", "louis_b",
            "entrainment_width_frac"],
}


@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_diff_turb_params_finite_match_fd(name):
    fn, mk = TURB_SCHEMES[name]
    cfg = mk()
    # Smagorinsky's eddy viscosity is purely shear/Ri-driven (no surface-flux
    # nonlocal term), so it is identically zero in a strongly stable interior;
    # exercise its tunables in a sheared near-neutral column where Ri < Pr_t.
    # The nonlocal K-profile schemes need an active convective BL instead.
    d = sheared_neutral_column(ncol=2, nlev=24) if name == "smagorinsky" \
        else unstable_column(ncol=2, nlev=24)

    def make_loss(param):
        def loss(pv):
            cfg_in = cfg._replace(**{param: pv})
            out = _run_turb(fn, cfg_in, d)
            return jnp.sum(out.dT_dt ** 2 + out.du_dt ** 2 + out.dq_v_dt ** 2)
        return loss

    saw_nonzero = False
    for param in TURB_GRAD_PARAMS[name]:
        p0 = float(getattr(cfg, param))
        loss = make_loss(param)
        g = float(jax.grad(loss)(p0))
        assert np.isfinite(g), f"{name}.{param}: non-finite grad"
        eps = max(abs(p0) * 1e-5, 1e-7)
        fd = float((loss(p0 + eps) - loss(p0 - eps)) / (2 * eps))
        if abs(fd) > 1e-14 or abs(g) > 1e-14:
            saw_nonzero = True
            np.testing.assert_allclose(g, fd, rtol=2e-3, atol=1e-10)
    assert saw_nonzero, f"{name}: all tunable grads dead in its active regime"


@pytest.mark.parametrize("gwd,mk,params", [
    (mcfarlane_gwd, McFarlaneConfig, ["G_0", "h_topo", "efficiency", "fcrit2"]),
    (hines_gwd, HinesConfig, ["total_rms_wind", "Fmax", "m_star"]),
])
def test_diff_gwd_params_finite_match_fd(gwd, mk, params):
    d = gwd_breaking_column(ncol=2, nlev=40)
    cfg = mk()

    def make_loss(param):
        def loss(pv):
            out = gwd(
                d["u"], d["v"], d["T"], d["p_full"], d["p_half"],
                d["z_full"], d["z_half"], d["rho"], d["lat"], DT,
                cfg._replace(**{param: pv}),
            )
            return jnp.sum(out.du_dt ** 2 + out.dT_dt ** 2)
        return loss

    saw_nonzero = False
    for param in params:
        p0 = float(getattr(cfg, param))
        loss = make_loss(param)
        g = float(jax.grad(loss)(p0))
        assert np.isfinite(g), f"GWD.{param}: non-finite grad"
        eps = max(abs(p0) * 1e-5, 1e-7)
        fd = float((loss(p0 + eps) - loss(p0 - eps)) / (2 * eps))
        if abs(fd) > 1e-14 or abs(g) > 1e-14:
            saw_nonzero = True
            np.testing.assert_allclose(g, fd, rtol=5e-3, atol=1e-12)
    assert saw_nonzero, f"GWD {gwd.__name__}: all tunable grads dead"


@pytest.mark.parametrize("name", list(TURB_SCHEMES))
def test_jit_vmap_consistency_turb(name):
    fn, mk = TURB_SCHEMES[name]
    cfg = mk()
    d = unstable_column(ncol=3, nlev=20)
    eager = _run_turb(fn, cfg, d)
    jitted = jax.jit(lambda dd: _run_turb(fn, cfg, dd))(d)
    np.testing.assert_allclose(np.array(eager.dT_dt), np.array(jitted.dT_dt), rtol=1e-10, atol=1e-12)

    # vmap over an ensemble axis prepended to every array input.
    d_ens = {k: jnp.stack([v, v + 0.0]) for k, v in d.items()}
    vfn = jax.vmap(lambda dd: _run_turb(fn, cfg, dd))
    vout = vfn(d_ens)
    np.testing.assert_allclose(np.array(vout.dT_dt[0]), np.array(eager.dT_dt), rtol=1e-10, atol=1e-12)


def test_gwd_no_nan_in_reversed_flow():
    """McFarlane hard U_proj>0 mask must not leak NaN gradients at reversal."""
    d = _base_column(2, 40, theta_top=500.0, theta_sfc=290.0, u_top=-10.0, u_sfc=25.0)
    d["lat"] = jnp.zeros(2)

    def loss(g0):
        out = mcfarlane_gwd(
            d["u"], d["v"], d["T"], d["p_full"], d["p_half"],
            d["z_full"], d["z_half"], d["rho"], d["lat"], DT,
            McFarlaneConfig(G_0=g0),
        )
        return jnp.sum(out.du_dt ** 2)
    out = mcfarlane_gwd(
        d["u"], d["v"], d["T"], d["p_full"], d["p_half"],
        d["z_full"], d["z_half"], d["rho"], d["lat"], DT, McFarlaneConfig(),
    )
    assert not bool(jnp.any(jnp.isnan(out.du_dt)))
    g = jax.grad(loss)(0.5)
    assert np.isfinite(float(g))
    # The deposited drag never accelerates a reversed layer.
    assert jnp.max(out.du_dt * d["u"]) <= 1e-9


# ===========================================================================
# IDEALIZED FIDELITY
# ===========================================================================

def test_idealized_smagorinsky_lilly_cutoff():
    cfg = SmagorinskyConfig(surface=SurfaceLayerConfig())
    d_stab = _base_column(1, 30, 360.0, 300.0, 4.0, 18.0)  # strongly stable
    d_stab["T_sfc"] = d_stab["T"][:, -1]
    d_stab["q_sfc"] = d_stab["q_v"][:, -1]
    d_unst = _base_column(1, 30, 300.0, 305.0, 4.0, 18.0)
    d_unst["T_sfc"] = d_unst["T"][:, -1] + 4.0
    d_unst["q_sfc"] = saturation_mixing_ratio(d_unst["T_sfc"], d_unst["p_full"][:, -1])
    o_s = _run_turb(smagorinsky_turbulence, cfg, d_stab)
    o_u = _run_turb(smagorinsky_turbulence, cfg, d_unst)
    assert jnp.all(o_s.Km >= 0.0)
    # Strong static stability (Ri >= Pr_t) shuts mixing off.
    assert jnp.mean(o_s.Km) < jnp.mean(o_u.Km)
    assert float(jnp.max(o_s.Km)) < 1e-6


def test_idealized_louis_stability_suppression():
    cfg = LouisConfig(surface=SurfaceLayerConfig())
    d_stab = _base_column(1, 30, 360.0, 300.0, 4.0, 18.0)
    d_stab["T_sfc"] = d_stab["T"][:, -1]
    d_stab["q_sfc"] = d_stab["q_v"][:, -1]
    d_unst = _base_column(1, 30, 300.0, 305.0, 4.0, 18.0)
    d_unst["T_sfc"] = d_unst["T"][:, -1] + 4.0
    d_unst["q_sfc"] = saturation_mixing_ratio(d_unst["T_sfc"], d_unst["p_full"][:, -1])
    o_s = _run_turb(louis_turbulence, cfg, d_stab)
    o_u = _run_turb(louis_turbulence, cfg, d_unst)
    assert jnp.all(o_s.Km > 0.0)  # Louis stays positive (never identically off)
    assert jnp.mean(o_s.Km) < jnp.mean(o_u.Km)


@pytest.mark.parametrize("name,fn,mk", [
    ("holtslag_boville", holtslag_boville_turbulence, lambda: HoltslagBovilleConfig(surface=SurfaceLayerConfig())),
    ("ysu", ysu_turbulence, lambda: YSUConfig(surface=SurfaceLayerConfig())),
])
def test_idealized_convective_kprofile_shape(name, fn, mk):
    d = _base_column(1, 30, 305.0, 300.0, 4.0, 12.0)
    d["T_sfc"] = d["T"][:, -1] + 12.0  # strong surface heating
    d["q_sfc"] = saturation_mixing_ratio(d["T_sfc"], d["p_full"][:, -1])
    out = _run_turb(fn, mk(), d)
    Km = out.Km[0]
    assert jnp.all(Km >= 0.0), f"{name}: negative Km"
    # K-profile peaks in the interior (inside PBL), not at the model top.
    assert int(jnp.argmax(Km)) not in (0, 1), f"{name}: Km peaks at top"
    # K small near the model top (free atmosphere).
    assert float(Km[0]) < float(jnp.max(Km))


def test_idealized_ysu_pbl_deepens_with_surface_heating():
    """YSU bulk-Ri PBL height must respond to surface heat flux.

    Regression for the codex atm-turb-gwd finding A: the bulk-Ri numerator now
    carries the Troen-Mahrt / Hong et al. (2006) unstable surface-excess parcel
    θ_T = b·(w'θ')_0/w_s, so a convective surface heat flux deepens the
    diagnosed PBL.  Uses a hydrostatically-consistent well-mixed-then-capped
    profile so the PBL stays bounded by the inversion (the first-crossing
    diagnosis is robust to a sharp inversion).
    """
    cfg = YSUConfig(surface=SurfaceLayerConfig())
    ncol, nlev = 1, 50
    p_half = jnp.linspace(2000.0, 1.0e5, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    # Iterate to a hydrostatic column: theta mixed to ~2 km then a strong
    # capping inversion above.
    T = jnp.full((ncol, nlev), 300.0) * exner
    for _ in range(5):
        dp = p_half[:, 1:] - p_half[:, :-1]
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        dz = jnp.abs(constants.R_d * T * dp / (constants.g * p_mid))
        z_half = jnp.concatenate(
            [jnp.zeros((ncol, 1)), jnp.cumsum(dz[:, ::-1], axis=1)], axis=1
        )[:, ::-1]
        z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
        theta = jnp.where(z_full < 2000.0, 300.0, 300.0 + 0.008 * (z_full - 2000.0))
        T = theta * exner
    rho = p_full / (constants.R_d * T)
    q_v = jnp.linspace(2e-3, 10e-3, nlev)[None, :]
    u = jnp.linspace(10.0, 4.0, nlev)[None, :]
    v = jnp.zeros((ncol, nlev))

    def run(off):
        T_sfc = T[:, -1] + off
        q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
        return ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                              T_sfc, q_sfc, rho, DT, cfg)

    o_neutral = run(0.0)   # zero surface flux
    o_warm = run(8.0)      # strong surface heating
    # PBL deepens with surface heating (the bug was h_pbl pinned at the floor).
    assert float(o_warm.h_pbl[0]) > float(o_neutral.h_pbl[0]) + 50.0
    # And it stays physically bounded by the inversion (not the domain mean).
    assert float(o_warm.h_pbl[0]) < 3000.0
    # K-profile magnitude also responds to the deeper convective BL.
    assert float(jnp.max(o_warm.Km)) > float(jnp.max(o_neutral.Km))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
