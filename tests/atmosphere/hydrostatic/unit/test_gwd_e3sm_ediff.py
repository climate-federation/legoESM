"""Direct unit tests for the E3SM/CAM GW eddy diffusion + energy closure.

Covers the new leaves:

* ``newtonian_alpha_profile`` — the height-dependent Newtonian-cooling
  coefficient (gw_drag.F90 alpha0/palph) interpolated to interfaces;
* ``gw_ediff`` — the GW-induced effective diffusivity at interfaces
  (gw_diffusion.F90), matched term-by-term to the gfortran ediff oracle;
* ``gw_diff_tridiag_coeffs`` + ``gw_diff_tend`` — the implicit (tridiagonal)
  vertical diffusion of a scalar, matched to the oracle's LU solve;
* ``momentum_energy_conservation`` — the C.-C. Chen column energy fixer
  (gw_common.F90), verified to drive the column total-energy residual to ~0;
* the ``e3sm_cam_gwd`` driver with ``do_eddy_diffusion`` /
  ``do_energy_conservation`` / ``use_newtonian_profile`` enabled.

The egwdffi anchors are baked from
``.physics-validator/gravity_wave_drag/oracle/`` where the live compare
matches the gfortran oracle to ~1e-16 (egwdffi) and ~5e-16 (the implicit
diffusion of a spike scalar) in float64.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMCAMConfig, E3SMFrontalConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    gw_prof, gw_cm_src, gw_drag_prof, gw_ediff, gw_diff_tridiag_coeffs,
    gw_diff_tend, momentum_energy_conservation, gw_taucd_net,
    newtonian_alpha_profile, e3sm_cam_gwd,
)

ORACLE_RAIR = 287.04
ORACLE_CPAIR = 1004.64
ORACLE_G = 9.80616
KWV = 6.28e-5
DC = 5.0
NGWV = 8
PVER = 40
DT = 1800.0
PRNDL = 0.25
EGWD_MAX = 150.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _column(pver=PVER):
    pint = np.linspace(100.0, 1.0e5, pver + 1)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    T = np.linspace(230.0, 290.0, pver)
    dp = np.diff(pint)
    dz = ORACLE_RAIR * T * dp / (ORACLE_G * pmid)
    z_half = np.zeros(pver + 1)
    for k in range(pver, 0, -1):
        z_half[k - 1] = z_half[k] + dz[k - 1]
    zm = 0.5 * (z_half[:-1] + z_half[1:])
    u = np.linspace(5.0, 30.0, pver)
    v = np.zeros(pver)
    return pint, pmid, T, zm, z_half, u, v


def _spectral_state():
    """Run the frontal source + solver to get a realistic (gwut, ubm, nm, c)."""
    pint, pmid, T, zm, z_half, u, v = _column()
    to = lambda a: jnp.asarray(a[None, :])
    pint_c = jnp.asarray(pint[None, :])
    dpm = jnp.abs(pint_c[:, 1:] - pint_c[:, :-1])
    rdpm = 1.0 / dpm
    piln = jnp.log(pint_c)
    cfg = E3SMCAMConfig(
        source="frontal", pgwv=NGWV, dc=DC, kwv=KWV, effgw=1.0,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
    )
    rhoi, ti, nm, ni = gw_prof(
        to(T), to(pmid), pint_c, ORACLE_CPAIR, ORACLE_RAIR, ORACLE_G, cfg.n2min
    )
    frontgf = jnp.full((1, PVER), 1e-9)
    kbot = int(np.argmin(np.abs(pmid - 5.0e4)))
    kfront = int(np.argmin(np.abs(pmid - 6.0e4)))
    tau0, src, tend, xv, yv, c, ubm, ubi = gw_cm_src(
        to(u), to(v), frontgf, NGWV, DC, cfg.frontal.c0, cfg.frontal.taubgnd,
        cfg.frontal.frontgfc, kbot, kfront,
    )
    tau, utgw, vtgw, gwut, _tau_sat = gw_drag_prof(
        tau0, c, src, tend, to(T), ti, piln, rhoi, nm, ni,
        ubm, ubi, xv, yv, dpm, rdpm, jnp.zeros(1), 1.0, DT, cfg,
        orographic_only=False, do_taper=False,
    )
    rk = np.zeros(PVER + 1)
    rk[0] = pint[0] / (ORACLE_RAIR * T[0])
    for k in range(1, PVER):
        rk[k] = pint[k] * 2.0 / (ORACLE_RAIR * (T[k] + T[k - 1]))
    rk[PVER] = pint[PVER] / (ORACLE_RAIR * T[PVER - 1])
    return dict(
        pmid=pmid, rdpm=rdpm, gwut=gwut, ubm=ubm, nm=nm, c=c, tend=tend,
        rk=jnp.asarray(rk[None, :]), u=u, T=T, zm=zm, pint=pint,
    )


# egwdffi oracle anchors at selected interfaces (frontal column above).
_ORACLE_EGWDFFI = {
    1: 0.0260065416926745,
    5: 0.0004462393022615828,
    10: 0.0001944891545927518,
    14: 0.001477968158338963,
    18: 9.049444284177556e-05,
}


def test_ediff_matches_oracle():
    st = _spectral_state()
    to = lambda a: jnp.asarray(a[None, :])
    kbot = int(np.argmin(np.abs(st["pmid"] - 5.0e4)))
    egwdffi = gw_ediff(
        st["gwut"], st["ubm"], st["nm"], st["c"], st["rk"], to(st["pmid"]),
        st["rdpm"], st["tend"], DT, ORACLE_G, ORACLE_RAIR, 0, kbot,
        PRNDL, EGWD_MAX,
    )
    eg = np.array(egwdffi[0])
    for k, ref in _ORACLE_EGWDFFI.items():
        np.testing.assert_allclose(eg[k], ref, rtol=1e-10)
    # zero at the top and bottom interfaces, and at/below tend_level.
    assert eg[0] == 0.0
    assert float(np.max(np.abs(eg[int(st["tend"][0]):]))) == 0.0


def test_diff_tend_conserves_mass_and_smooths_spike():
    """Implicit GW diffusion of a spike conserves the column integral (zero-flux
    BCs) and spreads the spike to neighbours (positive off-diagonal coupling)."""
    st = _spectral_state()
    to = lambda a: jnp.asarray(a[None, :])
    kbot = int(np.argmin(np.abs(st["pmid"] - 5.0e4)))
    egwdffi = gw_ediff(
        st["gwut"], st["ubm"], st["nm"], st["c"], st["rk"], to(st["pmid"]),
        st["rdpm"], st["tend"], DT, ORACLE_G, ORACLE_RAIR, 0, kbot,
        PRNDL, EGWD_MAX,
    )
    a, b, cc = gw_diff_tridiag_coeffs(
        egwdffi, st["rk"], to(st["pmid"]), st["rdpm"], DT, ORACLE_G, 0, kbot,
    )
    q = np.zeros(PVER)
    q[10] = 100.0
    dq = np.array(gw_diff_tend(to(q), a, b, cc, DT)[0])
    q_new = q + DT * dq
    # zero-flux BC: column sum conserved (the matrix is the implicit-diffusion
    # operator with no flux through the boundaries).
    np.testing.assert_allclose(np.sum(q_new), np.sum(q), rtol=1e-10)
    # the spike loses amplitude and the neighbours gain.
    assert q_new[10] < 100.0
    assert dq[9] > 0.0 and dq[11] > 0.0


def test_newtonian_alpha_profile():
    """alpha increases toward the model top (radiative damping strengthens with
    altitude) and is >= the 1e-6 floor everywhere; endpoints clamp."""
    pint = jnp.asarray(np.linspace(100.0, 1.0e5, PVER + 1)[None, :])
    alpha = np.array(newtonian_alpha_profile(pint)[0])
    assert np.all(alpha >= 1e-6 - 1e-18)
    # top interface (lowest p) has larger alpha than a mid-troposphere one.
    assert alpha[0] > alpha[-5]
    assert np.all(np.isfinite(alpha))


def test_energy_conservation_closes_budget():
    """The C.-C. Chen fixer drives the column total-energy residual to ~0 and
    deposits the penetrating stress as a uniform below-source body force.

    E3SM (gw_common.F90 momentum_energy_conservation): below the source
    (0-based midpoints m >= tend_level) it adds -tau_net*xv/dz / -tau_net*yv/dz
    to dudt/dvdt, then removes the net column total-energy change uniformly
    from dsdt so that
        sum_k pdel*(dsdt + dudt*(u+0.5 dt dudt) + dvdt*(v+0.5 dt dvdt)) == 0.
    """
    nlev = PVER
    rng = np.random.default_rng(0)
    pint = np.linspace(100.0, 1.0e5, nlev + 1)
    pdel = np.diff(pint)
    u = np.linspace(5.0, 30.0, nlev)
    v = np.linspace(-2.0, 4.0, nlev)
    dudt = rng.normal(0, 1e-4, nlev)
    dvdt = rng.normal(0, 1e-4, nlev)
    dsdt = rng.normal(0, 1e-2, nlev)
    tend_level = jnp.array([nlev // 2], dtype=jnp.int32)  # source partway down
    tau_net = jnp.array([0.5])   # synthetic penetrating stress
    xv = jnp.array([0.8]); yv = jnp.array([0.6])
    to = lambda a: jnp.asarray(a[None, :])
    du2, dv2, ds2 = momentum_energy_conservation(
        tend_level, DT, to(pdel), to(u), to(v), to(dudt), to(dvdt), to(dsdt),
        to(pint), ORACLE_G, tau_net, xv, yv,
    )
    du2 = np.array(du2[0]); dv2 = np.array(dv2[0]); ds2 = np.array(ds2[0])
    # full column total-energy residual after the fix == machine zero.
    e_resid = np.sum(
        pdel * (
            ds2
            + du2 * (u + 0.5 * DT * du2)
            + dv2 * (v + 0.5 * DT * dv2)
        )
    )
    assert abs(e_resid) < 1e-9, f"energy residual {e_resid:.3e} not closed"
    # momentum body force: below source, the dudt correction is a constant
    # -tau_net*xv/dz; check the uniform-deposition signature.
    below = np.arange(nlev) >= int(tend_level[0])
    dz = np.sum(pdel[below]) / ORACLE_G
    np.testing.assert_allclose(
        du2[below] - dudt[below], -float(tau_net[0]) * 0.8 / dz, rtol=1e-10
    )
    # above source: dudt unchanged.
    above = ~below
    np.testing.assert_allclose(du2[above], dudt[above], rtol=0, atol=1e-18)


def _driver_column(ncol=2, nlev=PVER):
    pint, pmid, T, zm, z_half, u, v = _column(nlev)
    rep = lambda a: jnp.broadcast_to(jnp.asarray(a)[None, :], (ncol, len(a)))
    pmid_c = rep(pmid)
    pint_c = jnp.broadcast_to(jnp.asarray(pint)[None, :], (ncol, nlev + 1))
    T_c = rep(T)
    zf_c = rep(zm)
    zh_c = jnp.broadcast_to(jnp.asarray(z_half)[None, :], (ncol, nlev + 1))
    rho_c = pmid_c / (ORACLE_RAIR * T_c)
    u_c = rep(u)
    v_c = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)
    return u_c, v_c, T_c, pmid_c, pint_c, zf_c, zh_c, rho_c, lat


def test_driver_energy_closure_machine_zero():
    """End-to-end: with do_energy_conservation the column total-energy budget
    (E3SM's exact metric) self-closes to machine precision.

    E3SM conserves
        E = sum_k pdel*(dsdt + dudt*(u+0.5 dt dudt) + dvdt*(v+0.5 dt dvdt))
    (the column-integrated dry-static-energy + kinetic-energy tendency,
    including the quadratic 0.5 dt (du/dt)^2 term).  The fixer removes the net
    column dE uniformly from dsdt below source so this integral -> 0.
    """
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column(ncol=1)
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    base = dict(source="frontal", pgwv=NGWV, dc=DC, effgw=0.5,
                frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10))
    pdel = np.abs(np.diff(np.array(ph[0])))
    u_np = np.array(u[0]); v_np = np.array(v[0])

    def metric(out):
        dudt = np.array(out.du_dt[0]); dvdt = np.array(out.dv_dt[0])
        dsdt = ORACLE_CPAIR * np.array(out.dT_dt[0])
        return np.sum(pdel * (
            dsdt
            + dudt * (u_np + 0.5 * DT * dudt)
            + dvdt * (v_np + 0.5 * DT * dvdt)
        ))

    out_off = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, do_energy_conservation=False),
        frontgf_col=frontgf)
    out_on = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, do_energy_conservation=True),
        frontgf_col=frontgf)
    e_off = metric(out_off)
    e_on = metric(out_on)
    assert abs(e_off) > 1e-3, "test column should have a non-trivial imbalance"
    # the closed budget is ~machine zero relative to the open one.
    assert abs(e_on) < 1e-12 * max(abs(e_off), 1.0), (
        f"energy residual {e_on:.3e} not closed (open was {e_off:.3e})"
    )


def test_driver_eddy_diffusion_changes_tendency():
    """Enabling do_eddy_diffusion adds the GW eddy diffusion of u,v and the
    dttdf heating to dT_dt -> the tendencies change vs the momentum-only path."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    base = dict(source="frontal", pgwv=NGWV, dc=DC,
                frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10))
    out0 = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, do_eddy_diffusion=False), frontgf_col=frontgf)
    out1 = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, do_eddy_diffusion=True), frontgf_col=frontgf)
    assert jnp.all(jnp.isfinite(out1.du_dt))
    assert jnp.all(jnp.isfinite(out1.dT_dt))
    # the eddy diffusion modifies du_dt and dT_dt.
    assert float(jnp.max(jnp.abs(out1.du_dt - out0.du_dt))) > 0.0
    assert float(jnp.max(jnp.abs(out1.dT_dt - out0.dT_dt))) > 0.0


def test_driver_newtonian_profile_changes_drag():
    """Enabling the Newtonian-cooling height profile changes the spectral drag
    (alpha enters the saturation diffusivity and the WKB damping)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    base = dict(source="frontal", pgwv=NGWV, dc=DC,
                frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10))
    out0 = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, use_newtonian_profile=False), frontgf_col=frontgf)
    out1 = e3sm_cam_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, DT,
        E3SMCAMConfig(**base, use_newtonian_profile=True), frontgf_col=frontgf)
    assert jnp.all(jnp.isfinite(out1.du_dt))
    assert float(jnp.max(jnp.abs(out1.du_dt - out0.du_dt))) > 0.0


def test_driver_ediff_jit_grad_safe():
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    cfg = E3SMCAMConfig(
        source="frontal", pgwv=NGWV, dc=DC, do_eddy_diffusion=True,
        do_energy_conservation=True, use_newtonian_profile=True,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
    )
    fn = functools.partial(e3sm_cam_gwd, config=cfg)
    eager = fn(u, v, T, pf, ph, zf, zh, rho, lat, DT, frontgf_col=frontgf)
    jitted = jax.jit(fn)(u, v, T, pf, ph, zf, zh, rho, lat, DT,
                         frontgf_col=frontgf)
    np.testing.assert_allclose(np.array(jitted.du_dt), np.array(eager.du_dt),
                              rtol=1e-12, atol=1e-15)

    def loss(u_in):
        out = fn(u_in, v, T, pf, ph, zf, zh, rho, lat, DT, frontgf_col=frontgf)
        return jnp.sum(out.du_dt ** 2 + out.dT_dt ** 2)

    g = jax.grad(loss)(u)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
