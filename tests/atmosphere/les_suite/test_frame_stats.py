"""Moist LES reference statistics + column budgets (``les_suite.frame_stats``) and the
moist terms of the spectral core's step budget. Run with JAX_ENABLE_X64=1."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
from legoesm.atmosphere.les_suite.emit import (
    faces_to_centres,
    sgs_vertical_scalar_flux_faces,
)
from legoesm.atmosphere.les_suite.frame_stats import (
    SPLIT_TERMS,
    FrameRecorder,
    MoistRunEmitter,
    column_contents,
    frame_statistics,
    make_frame_fields_fn,
    moist_step_budget,
)
from legoesm.atmosphere.les_suite.scm_coupling import liquid_water_theta

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_NT = 9   # standard tracer layout (q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r, N_i)


def _moist_state(nx=10, nz=16, sgs_buoyancy=True):
    """A small sheared, stratified, cloudy moist column on the monotone scalar path."""
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=nx, nz=nz, Lx=1000.0, Ly=1000.0, Lz=1600.0, z0=1e-4, dealias=True,
        smagorinsky_dynamic=False, sgs_model="smagorinsky", time_scheme="rk3",
        buoyancy=True, theta_ref0=300.0, moist=True, n_tracers=_NT,
        monotone_scalars=True, sgs_buoyancy=sgs_buoyancy)
    g = sl.make_grid(cfg, dtype=jnp.float64)
    k = jax.random.split(jax.random.PRNGKey(3), 6)
    shp = (nx, nx, nz)
    z = g.z_c
    th = 298.0 + 0.004 * z + 0.3 * jax.random.normal(k[0], shp)
    u = -6.0 + 2e-3 * z + 0.8 * jax.random.normal(k[1], shp)
    v = 0.8 * jax.random.normal(k[2], shp)
    w = 0.5 * jax.random.normal(k[3], (nx, nx, nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    u, v, w = sl.project(u, v, w, 1.0, g)
    tr = jnp.zeros(shp + (_NT,))
    tr = tr.at[..., 0].set(0.016 - 5e-6 * z + 5e-4 * jax.random.uniform(k[4], shp))
    cloud = ((z > 500.0) & (z < 1100.0)).astype(jnp.float64)
    tr = tr.at[..., 1].set(4e-4 * cloud * jax.random.uniform(k[5], shp))
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=u * 0, rhs_v_prev=v * 0,
                             rhs_w_prev=w * 0, theta=th, rhs_theta_prev=th * 0,
                             tracers=tr, rhs_tracers_prev=tr * 0)
    return g, st


def test_moist_step_budget_closes_and_needs_the_fv_terms():
    g, st = _moist_state()
    q0, qv0, dt = 0.01, 5e-5, 0.5
    step = jax.jit(lambda s: sl.step(s, g, dt, (-6.0, 0.0), 4e-5, sfc_theta_flux=q0,
                                     sfc_qv_flux=qv0, return_budget=True))
    c0 = np.asarray(column_contents(st, g))
    acc = np.zeros(len(sl.BUDGET_TERMS))
    for _ in range(10):
        st, _us, b = step(st)
        acc += np.asarray(b)
    c1 = np.asarray(column_contents(st, g))
    t = dict(zip(sl.BUDGET_TERMS, acc))
    d = c1 - c0
    tol = 1e-13 * np.abs(c1) + 1e-15
    r_th = d[0] - (t["sfc_heat_flux"] + t["fv_theta"])
    r_w = d[3] - (t["sfc_qv_flux"] + t["fv_water"])
    assert abs(r_th) <= tol[0] and abs(r_w) <= tol[3], (r_th, r_w, tol)
    # non-vacuous: the non-flux-form advection correction is far above round-off,
    # so a budget that dropped it would not close
    assert abs(t["fv_theta"]) > 1e3 * tol[0]
    assert abs(t["fv_water"]) > 1e3 * tol[3]
    np.testing.assert_allclose(t["sfc_qv_flux"], 10 * dt * qv0, rtol=1e-12)


def test_dry_step_has_no_moist_budget_terms():
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=400.0, Ly=400.0, Lz=400.0,
                               buoyancy=True)
    g = sl.make_grid(cfg, dtype=jnp.float64)
    th = 300.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8))
    z = jnp.zeros((8, 8, 8))
    st = sl.SpectralLESState(u=z + 1.0, v=z, w=jnp.zeros((8, 8, 9)), rhs_u_prev=z,
                             rhs_v_prev=z, rhs_w_prev=jnp.zeros((8, 8, 9)),
                             theta=th, rhs_theta_prev=z)
    _s, _u, b = sl.step(st, g, 0.5, (1.0, 0.0), 0.0, sfc_theta_flux=0.02,
                        sfc_qv_flux=1e-4, return_budget=True)
    t = dict(zip(sl.BUDGET_TERMS, np.asarray(b)))
    assert t["sfc_qv_flux"] == 0.0 and t["fv_theta"] == 0.0 and t["fv_water"] == 0.0


def test_effective_viscosity_is_what_rhs_applies_and_carries_lilly():
    g, st = _moist_state()
    nu_eff = sl.effective_eddy_viscosity(st.u, st.v, st.w, g, theta=st.theta,
                                         tracers=st.tracers)
    nu_raw = sl.eddy_viscosity(st.u, st.v, st.w, g)
    assert float(jnp.max(jnp.abs(nu_eff - nu_raw))) > 1e-3 * float(jnp.max(nu_raw))
    # the scalar SGS tendency rhs applies equals the divergence of the emitted face
    # flux built from the effective viscosity (planar mean; horizontal SGS integrates
    # to zero): rhs(ν_eff) − rhs(ν=0) is the SGS part alone
    phi = st.tracers[..., 0]
    full = sl.scalar_rhs_monotone(phi, st.u, st.v, st.w, nu_eff, g, 0.0)
    adv_only = sl.scalar_rhs_monotone(phi, st.u, st.v, st.w, nu_eff * 0.0, g, 0.0)
    sgs_tend = np.asarray((full - adv_only).mean((0, 1)))
    faces = sgs_vertical_scalar_flux_faces(np.asarray(phi), np.asarray(nu_eff),
                                           float(g.dz), pr_sgs=g.cfg.pr_sgs)
    np.testing.assert_allclose(-(faces[1:] - faces[:-1]) / g.dz, sgs_tend,
                               rtol=1e-10, atol=1e-18)
    raw = sgs_vertical_scalar_flux_faces(np.asarray(phi), np.asarray(nu_raw),
                                         float(g.dz), pr_sgs=g.cfg.pr_sgs)
    assert not np.allclose(-(raw[1:] - raw[:-1]) / g.dz, sgs_tend, rtol=1e-3)


def test_emitted_sgs_flux_is_what_rhs_applies(monkeypatch):
    """The frame's SGS θ flux (from the emitted ν_t) has exactly the divergence of
    the SGS part of the θ tendency ``rhs`` itself returns; with ``rhs`` planted to
    use the unsuppressed viscosity the identity fails."""
    g, st = _moist_state()
    fields = make_frame_fields_fn(g, (-6.0, 0.0), 4e-5)(st, 0.0, 0.0)
    faces = sgs_vertical_scalar_flux_faces(np.asarray(st.theta), np.asarray(fields["nu_t"]),
                                           float(g.dz), pr_sgs=g.cfg.pr_sgs)
    emitted = -(faces[1:] - faces[:-1]) / g.dz
    adv = np.asarray(sl.scalar_rhs_monotone(st.theta, st.u, st.v, st.w,
                                            fields["nu_t"] * 0.0, g, 0.0).mean((0, 1)))

    def rhs_sgs():
        r = sl.rhs(st.u, st.v, st.w, g, (-6.0, 0.0), 4e-5, theta=st.theta,
                   tracers=st.tracers)
        return np.asarray(r[4].mean((0, 1))) - adv

    np.testing.assert_allclose(emitted, rhs_sgs(), rtol=1e-10, atol=1e-18)
    monkeypatch.setattr(sl, "effective_eddy_viscosity",
                        lambda u, v, w, g, theta=None, tracers=None:
                        sl.eddy_viscosity(u, v, w, g))
    assert not np.allclose(emitted, rhs_sgs(), rtol=1e-3)


def test_frame_fields_use_the_effective_viscosity():
    g, st = _moist_state()
    f = make_frame_fields_fn(g, (-6.0, 0.0), 4e-5)(st, 0.01, 5e-5)
    nu_eff = sl.effective_eddy_viscosity(st.u, st.v, st.w, g, theta=st.theta,
                                         tracers=st.tracers)
    # jit vs eager: same expression, rounding-level reassociation only
    np.testing.assert_allclose(np.asarray(f["nu_t"]), np.asarray(nu_eff), rtol=1e-12)


def _exners(g):
    # any smooth reference Exner works for the identities
    return 1.0 - 1e-4 * np.asarray(g.z_c), 1.0 - 1e-4 * np.asarray(g.z_f)


def test_moist_frame_statistics_are_theta_l_and_total_water():
    g, st = _moist_state()
    q0, qv0 = 0.01, 5e-5
    fields = make_frame_fields_fn(g, (-6.0, 0.0), 4e-5)(st, q0, qv0)
    ex_c, ex_f = _exners(g)
    prof, srf = frame_statistics(st, g, fields, g.cfg.pr_sgs, exner_c=ex_c,
                                 exner_f=ex_f)
    th = np.asarray(st.theta)
    tr = np.asarray(st.tracers)
    thl = np.asarray(liquid_water_theta(th, tr[..., 1], ex_c[None, None, :]))
    np.testing.assert_allclose(prof["theta"], thl.mean((0, 1)), rtol=1e-14)
    assert np.all(prof["theta"] <= th.mean((0, 1)) + 1e-12)
    np.testing.assert_allclose(prof["qt"], (tr[..., 0] + tr[..., 1] + tr[..., 2])
                               .mean((0, 1)), rtol=1e-14)
    nu = np.asarray(fields["nu_t"])
    # θ_l SGS flux from the applied θ and q_c face fluxes (surface: the θ flux)
    j_th = sgs_vertical_scalar_flux_faces(th, nu, g.dz, pr_sgs=g.cfg.pr_sgs,
                                          surface_flux=q0)
    j_qc = sgs_vertical_scalar_flux_faces(tr[..., 1], nu, g.dz, pr_sgs=g.cfg.pr_sgs)
    expect = faces_to_centres(j_th - constants.L_v / constants.c_pd * j_qc / ex_f)
    np.testing.assert_allclose(prof["wtheta_sgs"], expect, rtol=1e-12)
    # surface face carries the applied heat / moisture flux
    j_qv = sgs_vertical_scalar_flux_faces(tr[..., 0], nu, g.dz, surface_flux=qv0)
    assert j_qv[0] == qv0 and srf["w_qv_s"] == qv0 and srf["w_theta_s"] == q0
    assert np.any(np.abs(prof["wqt_sgs"]) > 0.0)
    assert np.all((prof["cloud_frac"] >= 0.0) & (prof["cloud_frac"] <= 1.0))
    assert np.max(prof["cloud_frac"]) > 0.0


def test_theta_l_sgs_flux_is_the_applied_one():
    """The emitted θ_l SGS flux carries exactly what the core applied to θ and q_c:
    (a) with θ and q_c uniform in height the core applies no SGS flux, so neither
    may the diagnostic (a gradient flux −K∂θ_l/∂z would, since Π varies);
    (b) its divergence plus the documented variable-Exner source equals the θ_l
    SGS tendency the core's scalar operator applied, R_θ − (L_v/(c_pd Π_c)) R_qc."""
    from legoesm.atmosphere.les_suite.frame_stats import theta_l_sgs_faces
    g, st = _moist_state()
    ex_c, ex_f = _exners(g)
    nu = sl.effective_eddy_viscosity(st.u, st.v, st.w, g, theta=st.theta,
                                     tracers=st.tracers)
    nu_np, pr, dz = np.asarray(nu), g.cfg.pr_sgs, float(g.dz)
    lc = constants.L_v / constants.c_pd
    # (a) uniform θ and q_c, varying Π: interior faces carry nothing
    shp = st.theta.shape
    flat = theta_l_sgs_faces(np.full(shp, 300.0), np.full(shp, 4e-4), nu_np, dz, pr,
                             0.0, ex_f)
    assert np.max(np.abs(flat)) == 0.0
    thl_flat = np.asarray(liquid_water_theta(np.full(shp, 300.0), np.full(shp, 4e-4),
                                             ex_c[None, None, :]))
    grad = sgs_vertical_scalar_flux_faces(thl_flat, nu_np, dz, pr_sgs=pr)
    assert np.max(np.abs(grad)) > 1e-8                      # the rejected form is not zero
    # (b) tendency identity against the core's own operator (SGS part only)
    def sgs_tend(phi):
        full = sl.scalar_rhs_monotone(phi, st.u, st.v, st.w, nu, g, 0.0)
        adv = sl.scalar_rhs_monotone(phi, st.u, st.v, st.w, nu * 0.0, g, 0.0)
        return np.asarray((full - adv).mean((0, 1)))
    r_th, r_qc = sgs_tend(st.theta), sgs_tend(st.tracers[..., 1])
    applied = r_th - lc * r_qc / ex_c
    f = theta_l_sgs_faces(np.asarray(st.theta), np.asarray(st.tracers[..., 1]), nu_np,
                          dz, pr, 0.0, ex_f)
    j_qc = sgs_vertical_scalar_flux_faces(np.asarray(st.tracers[..., 1]), nu_np, dz,
                                          pr_sgs=pr)
    source = lc / dz * (j_qc[1:] * (1.0 / ex_c - 1.0 / ex_f[1:])
                        - j_qc[:-1] * (1.0 / ex_c - 1.0 / ex_f[:-1]))
    np.testing.assert_allclose(-(f[1:] - f[:-1]) / dz + source, applied,
                               rtol=1e-9, atol=1e-16)
    assert np.max(np.abs(source)) > 1e-6 * np.max(np.abs(applied))


def test_moist_frame_needs_reference_exner():
    g, st = _moist_state()
    fields = make_frame_fields_fn(g, (-6.0, 0.0), 4e-5)(st, 0.01, 5e-5)
    with pytest.raises(ValueError):
        frame_statistics(st, g, fields, g.cfg.pr_sgs)


def test_recorder_stacks_frames_and_series():
    g, st = _moist_state()
    fn = make_frame_fields_fn(g, (-6.0, 0.0), 4e-5)
    ex_c, ex_f = _exners(g)
    rec = FrameRecorder(g, g.cfg.pr_sgs, exner_c=ex_c, exner_f=ex_f)
    for k in range(3):
        rec.record(10.0 * k, st, fn(st, 0.01, 5e-5),
                   series={"w_qv_s_cum": 5e-5 * 10.0 * k, "precip_s": 0.0})
    kw = rec.artifact_kwargs()
    assert kw["theta"].shape == (3, g.cfg.nz) and kw["qt"].shape == (3, g.cfg.nz)
    np.testing.assert_allclose(kw["w_qv_s_cum"], [0.0, 5e-4, 1e-3])
    np.testing.assert_allclose(kw["times_s"], [0.0, 10.0, 20.0])


def test_moist_step_budget_vector_layout():
    dyc = jnp.arange(len(sl.BUDGET_TERMS), dtype=jnp.float64)
    vec = moist_step_budget(dyc, ls=jnp.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
                            micro=jnp.array([5.0, 0.0, 0.0, 6.0, 11.0, 12.0]),
                            positivity=jnp.array([0.0, 0.0, 0.0, 7.0, 0.0, 0.0]),
                            sponge=jnp.array([0.0, 8.0, 9.0, 0.0, 0.0, 0.0]),
                            precip_mass=10.0)
    tail = np.asarray(vec[len(sl.BUDGET_TERMS):])
    # ls_theta ls_water rad_theta micro_theta micro_water positivity sponge_u sponge_v
    # precip micro_water_rho micro_enthalpy_rho
    np.testing.assert_array_equal(
        tail, [1.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert len(tail) == len(SPLIT_TERMS)


def test_kessler_conserves_water_with_precipitation_as_the_only_sink():
    from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    nlev, dt = 30, 2.0
    z = jnp.linspace(3000.0, 15.0, nlev)[None, :]          # top-down, as the scheme expects
    dz = jnp.full((1, nlev), 100.0)
    p = 1e5 * jnp.exp(-z / 8000.0)
    p_half = jnp.concatenate([p[:, :1] * 0.99, 0.5 * (p[:, 1:] + p[:, :-1]),
                              p[:, -1:] * 1.01], axis=1)
    T = 298.0 - 6.5e-3 * z
    rho = p / (constants.R_d * T)
    q_v = 0.9 * 0.017 * jnp.exp(-z / 2500.0)
    cloud = ((z > 800.0) & (z < 2000.0)).astype(jnp.float64)
    q_c = 2.5e-3 * cloud                                    # above the autoconversion threshold
    q_r = 1e-3 * (z < 2000.0)                    # rain shaft down to the surface layer
    zero = jnp.zeros_like(q_v)
    hyd = HydrometeorState(q_c=q_c, q_r=q_r, q_i=zero, q_s=zero, q_g=zero,
                           N_c=zero, N_r=zero, N_i=zero)
    # The default evaporation coefficient evaporates all sub-saturated rain within
    # one step (its rate clamps at q_r/dt), so no rain reaches the surface; with
    # evaporation off the shaft does precipitate. Water closes in both.
    for cfg, rains_out in ((KesslerConfig(), False),
                           (KesslerConfig(evaporation_coeff=0.0), True)):
        out = kessler_microphysics(T, q_v, hyd, p, p_half, rho, dz, dt, cfg)
        w0 = jnp.sum(rho * (q_v + q_c + q_r) * dz)
        w1 = jnp.sum(rho * (q_v + dt * out.dq_v_dt + q_c + dt * out.dq_c_dt
                            + q_r + dt * out.dq_r_dt) * dz)
        precip = float(jnp.sum(out.precipitation)) * dt
        if rains_out:
            assert precip > 1e-6, (cfg, precip)
        else:
            assert precip == 0.0, (cfg, precip)
        np.testing.assert_allclose(float(w1 - w0), -precip, rtol=1e-10, atol=1e-14)
        # latent heating matches the net phase change (condensation − evaporation)
        net_vapour_loss = -jnp.sum(out.dq_v_dt * rho * dz)
        heat = jnp.sum(out.dT_dt * rho * dz) * constants.c_pd / constants.L_v
        np.testing.assert_allclose(float(heat), float(net_vapour_loss), rtol=1e-10)


def _kessler_les(g, rain_dt):
    """Reference state + the LES Kessler adapter on ``_moist_state``'s grid."""
    from types import SimpleNamespace

    from legoesm.atmosphere.dynamics.les.spectral_les_moist import make_les_microphysics_fn
    from legoesm.atmosphere.physics._shared import exner_function
    from legoesm.atmosphere.physics.microphysics.config import (
        KesslerConfig,
        MicrophysicsConfig,
    )
    z_c, z_f = np.asarray(g.z_c), np.asarray(g.z_f)
    p_c, p_f = 1.0e5 * np.exp(-z_c / 8000.0), 1.0e5 * np.exp(-z_f / 8000.0)
    exner_c = np.asarray(exner_function(p_c))
    rho_c = p_c / (constants.R_d * 298.0 * exner_c)
    ref = SimpleNamespace(p_c=jnp.asarray(p_c), p_f=jnp.asarray(p_f),
                          rho_c=jnp.asarray(rho_c), exner_c=jnp.asarray(exner_c))
    # evaporation off so the rain reaches the surface (the default coefficient
    # evaporates sub-saturated rain within one step)
    cfg = MicrophysicsConfig(scheme="kessler", kessler=KesslerConfig(
        autoconversion_threshold=1e-4, autoconversion_rate=5e-2, evaporation_coeff=0.0))
    return ref, make_les_microphysics_fn(cfg, ref, float(g.dz), rain_dt)


def test_driver_microphysics_closure_fails_without_the_precipitation_sink():
    """The emitter's density-weighted microphysics closures, fed exactly as a driver
    feeds them (``column_contents`` difference across the adapter + its precip):
    water mass changes only by the precipitation removed, moist enthalpy not at
    all; a driver that dropped the precipitation would fail the water closure."""
    g, st = _moist_state()
    dt = 2.0
    ref, micro = _kessler_les(g, dt)
    tr = st.tracers.at[..., 2].set(2e-3 * (np.asarray(g.z_c) < 400.0))  # rain at the floor
    st = st._replace(tracers=tr)

    def budget_for(state, keep_precip):
        c0 = column_contents(state, g, ref)
        dth, dtr, pr = micro(state.theta, state.tracers)
        new = state._replace(theta=state.theta + dt * dth,
                             tracers=state.tracers + dt * dtr)
        dyc = jnp.zeros(len(sl.BUDGET_TERMS))
        b = moist_step_budget(dyc, micro=column_contents(new, g, ref) - c0,
                              precip_mass=(dt * jnp.mean(pr) if keep_precip else 0.0))
        return new, b, float(dt * jnp.mean(pr))

    def residuals(keep_precip):
        em = MoistRunEmitter.__new__(MoistRunEmitter)
        em.g, em.ref = g, ref
        em.terms = tuple(sl.BUDGET_TERMS) + SPLIT_TERMS
        em.col0 = np.asarray(column_contents(st, g, ref))
        new, b, precip = budget_for(st, keep_precip)
        em.acc = b
        # transport-norm rows see no dycore here, so only the micro rows are tested
        res = em.close_budgets(new)
        return res, precip, np.asarray(column_contents(new, g, ref))

    res, precip, col = residuals(True)
    assert precip > 1e-8                                   # rain does leave the column
    assert abs(res["micro_water"]) <= 1e-13 * abs(col[4]) + 1e-15, res
    assert abs(res["micro_enthalpy"]) <= 1e-12 * abs(col[5]), res
    bad, _p, _c = residuals(False)
    assert abs(bad["micro_water"]) > 1e3 * (1e-13 * abs(col[4]) + 1e-15), bad
