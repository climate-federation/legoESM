"""MPAS-A scalar transport on the prescribed-wind MPAS tracer model.

Targets ``horizontal_scheme="mpas_fct"`` of
``legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas`` (``mpas_rk3_fct_step``)
and the MPAS edge-reconstruction / streamfunction helpers of
``legoesm.core.operators_voronoi``:

* the Skamarock-Gassmann (2011) stencil: sum rules and its 4th-order accuracy
  for cell means (point-sampled, the 4th-order edge value differs from the
  midpoint value by -1/3 of the centred average's error, to O(dx^4));
* the streamfunction edge wind is discretely non-divergent;
* RULE 0: global tracer mass ``sum areaCell q dp`` is conserved to round-off
  with a DIVERGENT horizontal wind (fails for the advective centred2 form);
* a constant mixing ratio stays constant (divergent wind + vertical motion);
* RULE 5: at the hard limit the step matches a literal loop transcription of
  MPAS ``atm_srk3`` / ``atm_advance_scalars`` / ``atm_advance_scalars_mono``
  and is monotone; at the smooth default the gap to it is pinned per case;
  ``check_grads`` through the limiter at the default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    edge_normal_velocity_from_streamfunction,
    gradient_edge_3d,
    mpas_advection_coefficients,
    mpas_edge_reconstruction,
)
from legoesm.core.state import TracerState
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
    TracerAirMassState,
    TracerTransportMPASConfig,
    TracerTransportMPASModel,
    air_mass_state,
    mpas_rk3_fct_step,
    validate_tracer_transport_mpas_config,
)

NLEV = 3
DT = 1800.0
P_S = 1.0e5


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(2, lloyd_iterations=5)


@pytest.fixture(scope="module")
def coefs(mesh):
    return mpas_advection_coefficients(mesh)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(NLEV)


def _cfg(width=1.0e-4, uwidth=1.0e-3, coef3=0.25):
    return TracerTransportMPASConfig(time_integrator="mpas_rk3", horizontal_scheme="mpas_fct",
                                     coef_3rd_order=coef3, fct_smoothing_width=width,
                                     fct_upwind_width_ms=uwidth)


def _rot_wind(mesh, amp=2.0e7, seed=0):
    psi = amp * jax.random.normal(jax.random.PRNGKey(seed), (mesh.nVertices,))
    return edge_normal_velocity_from_streamfunction(psi[:, None] * jnp.ones(NLEV), mesh)


def _div_wind(mesh, amp=1.0e7, seed=3):
    phi = amp * jax.random.normal(jax.random.PRNGKey(seed), (mesh.nCells,))
    return gradient_edge_3d(phi[:, None] * jnp.ones(NLEV), mesh)


def _const_wind(u_edge, sigma_dot):
    return lambda t, m, s: (u_edge, sigma_dot)


def _state(q, sigma):
    st = TracerState(tracers=Field(data=q, name="tracers", dims=("cell", "level", "tracer")),
                     time=Field(data=jnp.asarray(0.0), name="time", dims=()))
    return air_mass_state(st, P_S, sigma)


def _blocky(mesh, seed=1, ntr=2):
    """Discontinuous positive field (two levels of values) plus noise."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    base = jnp.where(jax.random.uniform(k1, (mesh.nCells, NLEV, ntr)) > 0.6, 1.0, 0.1)
    return base + 0.05 * jax.random.uniform(k2, (mesh.nCells, NLEV, ntr))


def _mass(st, mesh):
    return jnp.sum(mesh.areaCell[:, None, None] * st.air_mass.data[..., None] * st.tracers.data,
                   axis=(0, 1))


def _run(st, n, mesh, sigma, wind, cfg, coefs):
    for _ in range(n):
        st = mpas_rk3_fct_step(st, DT, mesh, sigma, wind, cfg, coefs)
    return st


# ---------------------------------------------------------------------------
# Reconstruction / wind helpers
# ---------------------------------------------------------------------------

def test_adv_coefs_sum_rules(coefs):
    # a constant reconstructs exactly; the 3rd-order correction of a constant is 0
    np.testing.assert_allclose(np.sum(np.asarray(coefs.adv_coefs), axis=0), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.sum(np.asarray(coefs.adv_coefs_3rd), axis=0), 0.0, atol=1e-13)


def test_reconstruction_is_fourth_order_for_cell_means():
    """Point-sampled q: the 4th-order edge value = 0.5(q1+q2) - dc^2/6 q''
    misses the midpoint value by -dc^2/24 q'' = -1/3 of the centred average's
    miss (dc^2/8 q''), the rest being O(dc^4).  Edges whose normal is nearly
    meridional on a zonally symmetric field (1-D along the normal); the
    residual falls 15.7x from level 3 to 4 (measured; 4th order = 16x)."""
    f = lambda lat: np.sin(3 * lat) + np.exp(np.sin(lat))
    res = []
    for lev in (3, 4):
        m = create_voronoi_mesh(lev)
        q = jnp.asarray(f(np.asarray(m.latCell, np.float64)))
        c1, c2 = np.asarray(m.cellsOnEdge)
        x = np.stack([np.asarray(getattr(m, k), np.float64) for k in ("xCell", "yCell", "zCell")], 1)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        mid = x[c1] + x[c2]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        qe = f(np.arcsin(mid[:, 2]))
        sel = ((np.abs(np.sin(np.asarray(m.angleEdge, np.float64))) > 0.97)
               & (np.abs(np.arcsin(mid[:, 2])) < 1.0))
        q4, _ = mpas_edge_reconstruction(q, mpas_advection_coefficients(m))
        cen = 0.5 * (np.asarray(q)[c1] + np.asarray(q)[c2])
        res.append(np.sqrt(np.mean((np.asarray(q4)[sel] - qe[sel] + (cen[sel] - qe[sel]) / 3) ** 2)))
    assert res[0] / res[1] > 12.0, res


def test_third_order_correction_sign():
    """For a flow c1 -> c2 the 3rd-order value adds dc^2/12 (D2 - D1), D the
    second derivative along the normal at each cell (1-D: (-q_{i-1} + 5 q_i +
    2 q_{i+1})/6).  Checked against the analytic D of a zonally symmetric
    field on near-meridional edges: regression slope ~ 1, independent of the
    stencil code."""
    f1 = lambda lat: 3 * np.cos(3 * lat) + np.exp(np.sin(lat)) * np.cos(lat)
    f2 = lambda lat: -9 * np.sin(3 * lat) + np.exp(np.sin(lat)) * (np.cos(lat) ** 2 - np.sin(lat))
    f = lambda lat: np.sin(3 * lat) + np.exp(np.sin(lat))
    m = create_voronoi_mesh(4)
    lat = np.asarray(m.latCell, np.float64)
    _, q3 = mpas_edge_reconstruction(jnp.asarray(f(lat)), mpas_advection_coefficients(m))
    c1, c2 = np.asarray(m.cellsOnEdge)
    ang = np.asarray(m.angleEdge, np.float64)
    sel = (np.abs(np.sin(ang)) > 0.97) & (np.abs(lat[c1]) < 1.0)
    R = float(m.radius)
    d = lambda la: (f2(la) * np.sin(ang) ** 2 - f1(la) * np.tan(la) * np.cos(ang) ** 2) / R ** 2
    expect = (np.asarray(m.dcEdge, np.float64) ** 2 / 12.0 * (d(lat[c2]) - d(lat[c1])))[sel]
    got = np.asarray(q3)[sel]
    slope = np.dot(got, expect) / np.dot(expect, expect)
    assert 0.9 < slope < 1.1, slope


def test_dcmip_edge_winds_match_analytic_projection():
    """The DCMIP edge winds (streamfunction curl + projected divergent part)
    equal the analytic wind projected on the edge normal at the edge point to
    the discretisation error, for each test (sign of the curl included)."""
    import tests.atmosphere.hydrostatic.test_cases.dcmip_transport as D
    m = create_voronoi_mesh(4)
    sc = D.create_dcmip_sigma(8)
    for tn in (11, 12, 13):
        ue, _ = D.dcmip_wind_mpas(tn, 2.0e4, m, sc)
        u, v, _ = getattr(D, f"dcmip{tn}_wind_geo")(2.0e4, m.lonEdge, m.latEdge, sc)
        ang = m.angleEdge[:, None]
        ref = u * jnp.cos(ang) + v * jnp.sin(ang)
        rel = float(jnp.sqrt(jnp.mean((ue - ref) ** 2) / jnp.mean(ref ** 2)))
        assert rel < 3e-3, (tn, rel)


def test_dcmip13_air_mass_balances_the_analytic_vertical_flux():
    """DCMIP 1-3 air mass ``dp = p0 exp(-z_s/H) dsigma``: the initial air-mass
    tendency (horizontal mass divergence of the streamfunction wind + the
    analytic ``p_s sigma_dot``) is truncation error, falling with resolution
    (rms 2.97e-6 -> 8.81e-7 /s of dp from ico4 to ico5, measured); with a flat
    ``p_s = p0`` it is O(1) and does not fall (5.06e-6 at both)."""
    import tests.atmosphere.hydrostatic.test_cases.dcmip_transport as D
    sc = D.create_dcmip_sigma(30)

    def rms_tendency(lev, flat):
        m = create_voronoi_mesh(lev)
        st = D.dcmip13_init_mpas(m, sc)
        st = air_mass_state(st, D._p0, sc) if flat else D.dcmip_air_mass_state_mpas(13, st, m, sc)
        u, sd = D.dcmip_wind_mpas(13, 0.0, m, sc)
        dp = st.air_mass.data
        fm = u * 0.5 * (dp[m.cellsOnEdge[0]] + dp[m.cellsOnEdge[1]])
        ps = dp.sum(-1, keepdims=True) / jnp.sum(sc.dsigma)
        t = -divergence_cell_3d(fm, m) - ps * (sd[:, 1:] - sd[:, :-1])
        return float(jnp.sqrt(jnp.mean((t / dp) ** 2)))

    assert rms_tendency(4, False) / rms_tendency(5, False) > 3.0
    assert rms_tendency(4, True) / rms_tendency(5, True) < 1.1


def test_streamfunction_wind_is_discretely_nondivergent(mesh):
    u = _rot_wind(mesh)
    div = divergence_cell_3d(u, mesh)
    scale = float(jnp.max(jnp.abs(u))) / float(jnp.min(mesh.dcEdge))
    assert float(jnp.max(jnp.abs(div))) < 1e-12 * scale


# ---------------------------------------------------------------------------
# RULE 0 and consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,uwidth", [(1.0e-4, 1.0e-3), (0.0, 0.0)])
def test_tracer_mass_conserved_divergent_wind(mesh, sigma, coefs, width, uwidth):
    """RULE 0: one flux per edge -> sum areaCell q dp conserved to round-off
    with a divergent horizontal wind (sigma_dot = 0)."""
    u = _rot_wind(mesh) + _div_wind(mesh)
    wind = _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1)))
    st0 = _state(_blocky(mesh), sigma)
    st = _run(st0, 10, mesh, sigma, wind, _cfg(width, uwidth), coefs)
    m0, m1 = _mass(st0, mesh), _mass(st, mesh)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m0), rtol=1e-13)
    # and the air mass itself
    am0 = jnp.sum(mesh.areaCell[:, None] * st0.air_mass.data)
    am1 = jnp.sum(mesh.areaCell[:, None] * st.air_mass.data)
    np.testing.assert_allclose(float(am1), float(am0), rtol=1e-13)


def test_centred2_advective_form_fails_the_mass_check(mesh, sigma):
    """Non-vacuity of the RULE 0 test: the same wind and check with the
    advective centred2 form (the pre-mpas_fct scheme) does NOT conserve."""
    u = _rot_wind(mesh) + _div_wind(mesh)
    model = TracerTransportMPASModel(mesh, sigma, _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))),
                                     TracerTransportMPASConfig())
    q0 = _blocky(mesh)
    st = TracerState(tracers=Field(data=q0, name="tracers", dims=("cell", "level", "tracer")),
                     time=Field(data=jnp.asarray(0.0), name="time", dims=()))
    for _ in range(10):
        st = model.step(st, DT)
    w = mesh.areaCell[:, None, None]
    rel = float(jnp.max(jnp.abs(jnp.sum(w * st.tracers.data, (0, 1)) / jnp.sum(w * q0, (0, 1)) - 1)))
    assert rel > 1e-6, rel


@pytest.mark.parametrize("width,uwidth", [(1.0e-4, 1.0e-3), (0.0, 0.0)])
def test_constant_mixing_ratio_preserved(mesh, sigma, coefs, width, uwidth):
    """Air mass and tracer mass use the same fluxes: q = const stays const with
    a divergent wind and a vertical motion."""
    u = _rot_wind(mesh) + _div_wind(mesh)
    sd = 1.0e-6 * jnp.sin(jnp.linspace(0.0, np.pi, NLEV + 1))[None, :] \
        * jnp.cos(mesh.latCell)[:, None]
    st0 = _state(0.7 * jnp.ones((mesh.nCells, NLEV, 2)), sigma)
    st = _run(st0, 5, mesh, sigma, _const_wind(u, sd), _cfg(width, uwidth), coefs)
    np.testing.assert_allclose(np.asarray(st.tracers.data), 0.7, rtol=1e-13)
    assert float(jnp.max(jnp.abs(st.air_mass.data - st0.air_mass.data))) > 1.0   # dp did evolve


def test_three_d_nondivergent_flow_keeps_air_mass(mesh, sigma, coefs):
    """Horizontal mass divergence balanced by the vertical one (sigma_dot from
    the column continuity with p = sigma p_s, model top sigma_top > 0) leaves
    dp unchanged: the vertical air-mass flux is p_s sigma_dot, p_s = sum(dp) /
    sum(dsigma) (not sum(dp))."""
    assert float(sigma.sigma_half[0]) > 0.0
    dsig = np.asarray(sigma.dsigma, np.float64)
    c = np.zeros(NLEV)
    c[0], c[1] = 1.0, -dsig[0] / dsig[1]              # column-integrated divergence 0
    u = _div_wind(mesh)[:, :1] * jnp.asarray(c)[None, :]
    st0 = _state(_blocky(mesh), sigma)
    dp0 = st0.air_mass.data
    fm = u * 0.5 * (dp0[mesh.cellsOnEdge[0]] + dp0[mesh.cellsOnEdge[1]])
    div = divergence_cell_3d(fm, mesh)                # (nC, nlev) [Pa/s]
    sd = jnp.concatenate([jnp.zeros((mesh.nCells, 1)), -jnp.cumsum(div, axis=1) / P_S], axis=1)
    assert float(jnp.max(jnp.abs(sd[:, -1]))) < 1e-18
    st = mpas_rk3_fct_step(st0, DT, mesh, sigma, _const_wind(u, sd), _cfg(), coefs)
    np.testing.assert_allclose(np.asarray(st.air_mass.data), np.asarray(dp0), rtol=1e-12)


# ---------------------------------------------------------------------------
# Oracle: literal loop transcription of MPAS-A (hard limiter, sigma_dot = 0)
# ---------------------------------------------------------------------------

def _mpas_oracle(q0, dp0, u, mesh, coefs, coef3, dt):
    """atm_srk3 scalar stages, transcribed loop by loop from
    mpas_atm_time_integration.F (atm_advance_scalars_work for rk_step 1-2 with
    advance_density, atm_advance_scalars_mono_work for rk_step 3, eps=1e-20),
    horizontal only, without the max(0, .) copy-back (see KNOWN DIFFERENCES)."""
    q0, dp0, u = (np.asarray(x, np.float64) for x in (q0, dp0, u))
    coe = np.asarray(mesh.cellsOnEdge)
    eoc, sgn = np.asarray(mesh.edgesOnCell), np.asarray(mesh.edgeSignOnCell, np.float64)
    coc, neoc = np.asarray(mesh.cellsOnCell), np.asarray(mesh.nEdgesOnCell)
    dv, area = np.asarray(mesh.dvEdge, np.float64), np.asarray(mesh.areaCell, np.float64)
    cells, a4, a3 = (np.asarray(x) for x in coefs)
    a3 = coef3 * a3
    nC, nE, nl = q0.shape[0], coe.shape[1], q0.shape[1]

    def uh_of(dp):
        return u * 0.5 * (dp[coe[0]] + dp[coe[1]])

    def hflux(qs, uh):   # flux_arr incl. dvEdge, per MPAS adv_coefs*dvEdge
        f = np.zeros((nE, nl))
        for e in range(nE):
            for k in range(nl):
                w = a4[:, e] + np.sign(uh[e, k]) * a3[:, e]
                f[e, k] = uh[e, k] * dv[e] * np.dot(w, qs[cells[:, e], k])
        return f

    def cell_sum(fe):    # sum_i sign * fe / area
        out = np.zeros((nC, nl))
        for c in range(nC):
            for i in range(neoc[c]):
                out[c] += sgn[i, c] * fe[eoc[i, c]]
            out[c] /= area[c]
        return out

    qs, dps = q0, dp0
    for tau in (dt / 3.0, dt / 2.0):
        uh = uh_of(dps)
        rho_new = dp0 - tau * cell_sum(uh * dv[:, None])
        qs = (q0 * dp0 - tau * cell_sum(hflux(qs, uh))) / rho_new
        dps = rho_new
    uh = uh_of(dps)
    flux_arr = hflux(qs, uh)
    rho_int = dp0 - dt * cell_sum(uh * dv[:, None])
    s_max, s_min = q0.copy(), q0.copy()
    for c in range(nC):
        for i in range(neoc[c]):
            s_max[c] = np.maximum(s_max[c], q0[coc[i, c]])
            s_min[c] = np.minimum(s_min[c], q0[coc[i, c]])
    f_up = np.zeros((nE, nl))
    for e in range(nE):
        c1, c2 = coe[:, e]
        f_up[e] = dv[e] * dt * (np.maximum(0.0, uh[e]) * q0[c1] + np.minimum(0.0, uh[e]) * q0[c2])
    flux_tmp = dt * flux_arr - f_up
    snew = q0 * dp0 - cell_sum(f_up)
    s_in, s_out = np.zeros((nC, nl)), np.zeros((nC, nl))
    for c in range(nC):
        for i in range(neoc[c]):
            ft = sgn[i, c] * flux_tmp[eoc[i, c]]
            s_out[c] -= np.maximum(0.0, ft) / area[c]
            s_in[c] -= np.minimum(0.0, ft) / area[c]
    eps = 1.0e-20
    sc_in = np.clip((s_max * rho_int - snew) / (s_in + eps), 0.0, 1.0)
    sc_out = np.clip((s_min * rho_int - snew) / (s_out - eps), 0.0, 1.0)
    lim = np.zeros((nE, nl))
    for e in range(nE):
        c1, c2 = coe[:, e]
        f = flux_tmp[e]
        lim[e] = (np.maximum(0.0, f) * np.minimum(sc_out[c1], sc_in[c2])
                  + np.minimum(0.0, f) * np.minimum(sc_in[c1], sc_out[c2]))
    return (snew - cell_sum(lim)) / rho_int, rho_int


_CASES = {"rot": (2.0e7, 0.0), "rot+div": (2.0e7, 1.0e7)}


def _case(mesh, name):
    a_rot, a_div = _CASES[name]
    u = _rot_wind(mesh, a_rot) + (_div_wind(mesh, a_div) if a_div else 0.0)
    return u, _blocky(mesh, ntr=1)[..., 0]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_hard_limit_matches_mpas_transcription(mesh, sigma, coefs, name):
    u, q = _case(mesh, name)
    st0 = _state(q[..., None], sigma)
    st = mpas_rk3_fct_step(st0, DT, mesh, sigma, _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))),
                           _cfg(0.0, 0.0), coefs)
    q_ref, dp_ref = _mpas_oracle(q, st0.air_mass.data, u, mesh, coefs, 0.25, DT)
    np.testing.assert_allclose(np.asarray(st.air_mass.data), dp_ref, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(st.tracers.data[..., 0]), q_ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("width,uwidth", [(1.0e-4, 1.0e-3), (0.0, 0.0)])
def test_local_bounds_held_at_default_and_hard(mesh, sigma, coefs, width, uwidth):
    """Non-divergent wind, blocky field with fluxes of mixed magnitude and
    sign: after one step every cell stays within the HARD local bounds
    (extremes of q0 over the cell and its neighbours) at the smooth default
    too (smooth extrema inside, smooth in/out budgets from above)."""
    u, q = _case(mesh, "rot")
    q = q * jnp.where(jax.random.uniform(jax.random.PRNGKey(9), q.shape) > 0.5, 1.0, 1e-5)
    st = mpas_rk3_fct_step(_state(q[..., None], sigma), DT, mesh, sigma,
                           _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))), _cfg(width, uwidth), coefs)
    coc = np.asarray(mesh.cellsOnCell)
    qn = np.asarray(q)
    nb = np.where((coc >= 0)[..., None], qn[np.maximum(coc, 0)], qn[None])
    lo, hi = np.minimum(qn, nb.min(0)), np.maximum(qn, nb.max(0))
    qnew = np.asarray(st.tracers.data[..., 0])
    # round-off only (measured: 5.6e-19 below at the default, 2.3e-18 hard)
    tol = 1e-14 * float(np.max(np.abs(qn)))
    assert np.all(qnew >= lo - tol) and np.all(qnew <= hi + tol)


def test_hard_limit_is_monotone(mesh, sigma, coefs):
    """Non-divergent wind: q_new within the extremes of q0 over the cell and
    its neighbours (the s_min/s_max bounds)."""
    u, q = _case(mesh, "rot")
    st = mpas_rk3_fct_step(_state(q[..., None], sigma), DT, mesh, sigma,
                           _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))), _cfg(0.0, 0.0), coefs)
    coc = np.asarray(mesh.cellsOnCell)
    qn = np.asarray(q)
    nb = np.where((coc >= 0)[..., None], qn[np.maximum(coc, 0)], qn[None])
    lo, hi = np.minimum(qn, nb.min(0)), np.maximum(qn, nb.max(0))
    qnew = np.asarray(st.tracers.data[..., 0])
    assert np.all(qnew >= lo - 1e-12) and np.all(qnew <= hi + 1e-12)
    # unlimited high order (coef huge limiter room) is NOT monotone here: the test can fail
    assert float(jnp.max(jnp.abs(_unlimited(q, u, mesh, sigma, coefs) - qnew))) > 1e-3


def test_smooth_default_never_exceeds_hard_bounds(mesh, sigma, coefs):
    """Default widths: a plateau (step function 0/1) stays within [0, 1].  The
    smooth extrema are taken from inside the hard interval; with max_rel /
    min_rel (from outside) the plateau crept up every step (DCMIP 1-3 q3, 12
    days: max 1.0188, measured)."""
    u = _rot_wind(mesh)
    k = jax.random.PRNGKey(5)
    q = jnp.where(jax.random.uniform(k, (mesh.nCells, NLEV, 1)) > 0.5, 1.0, 0.0)
    st = _run(_state(q, sigma), 20, mesh, sigma,
              _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))), _cfg(), coefs)
    assert float(jnp.max(st.tracers.data)) <= 1.0 + 1e-13
    assert float(jnp.min(st.tracers.data)) >= -1e-13


def _unlimited(q, u, mesh, sigma, coefs):
    """The same RK3 without the limiter (stage 3 = stage 1-2 form)."""
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_mpas as tm
    cfg = _cfg(0.0, 0.0)
    st0 = _state(q[..., None], sigma)
    q0, dp0 = st0.tracers.data, st0.air_mass.data
    qs, dps = q0, dp0
    zero = jnp.zeros((mesh.nCells, NLEV + 1))
    for tau in (DT / 3.0, DT / 2.0, DT):
        fm, a, f_ho, vm, vt = tm._mpas_mass_fluxes(qs, dps, u, zero, mesh, sigma, coefs, cfg)
        div = lambda f: divergence_cell_3d(f.reshape(mesh.nEdges, -1), mesh).reshape(qs.shape[0], NLEV, -1)
        dps = dp0 - tau * div(fm)[..., 0]
        qs = (q0 * dp0[..., None] - tau * div(f_ho)) / dps[..., None]
    return qs[..., 0]


# Gap of the smooth default (width 1e-4, upwind width 1e-3 m/s) to the hard
# limiter, max |q_default - q_hard| after one step, per case.  Measured
# 2026-11-23 (fp64, CPU) after masking padded (pentagon) neighbour slots out
# of the smooth bounds (was 2.61e-5 with them); re-measure if the smooth
# forms or defaults change.
_DEFAULT_GAP = {"rot": 1.0850146481278244e-06, "rot+div": 1.095331088496465e-06}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_smooth_default_gap_per_case(mesh, sigma, coefs, name):
    u, q = _case(mesh, name)
    wind = _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1)))
    st0 = _state(q[..., None], sigma)
    hard = mpas_rk3_fct_step(st0, DT, mesh, sigma, wind, _cfg(0.0, 0.0), coefs)
    soft = mpas_rk3_fct_step(st0, DT, mesh, sigma, wind, _cfg(), coefs)
    gap = float(jnp.max(jnp.abs(soft.tracers.data - hard.tracers.data)))
    np.testing.assert_allclose(gap, _DEFAULT_GAP[name], rtol=1e-6, atol=1e-12)


def test_rk3_stage_wind_times(mesh, sigma, coefs):
    """atm_srk3: each stage's fluxes come from the previous stage's state, so
    the prescribed wind is sampled at t0, t0 + dt/3, t0 + dt/2."""
    u = _rot_wind(mesh)
    times = []

    def wind(t, m, s):
        times.append(float(t))
        return u, jnp.zeros((mesh.nCells, NLEV + 1))

    st0 = _state(_blocky(mesh), sigma)
    st0 = st0._replace(time=st0.time.replace(data=jnp.asarray(7200.0)))
    mpas_rk3_fct_step(st0, DT, mesh, sigma, wind, _cfg(), coefs)
    np.testing.assert_allclose(times, [7200.0, 7200.0 + DT / 3.0, 7200.0 + DT / 2.0])


def test_positive_with_wind_inside_upwind_band(mesh, sigma, coefs):
    """An empty donor next to a positive cell with |u| ~ 1e-4 m/s (inside the
    fct_upwind_width_ms band) stays non-negative: the smooth |fm| is taken
    from above, so the low-order flux never draws from an empty cell."""
    u = _rot_wind(mesh, amp=2.0e2)                       # |u| ~ 1e-4 m/s
    assert float(jnp.max(jnp.abs(u))) < 1.0e-3
    q = jnp.zeros((mesh.nCells, NLEV, 1)).at[::7].set(1.0)
    st = _run(_state(q, sigma), 3, mesh, sigma,
              _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))), _cfg(), coefs)
    assert float(jnp.min(st.tracers.data)) >= -1e-15


def test_check_grads_through_limiter(mesh, sigma, coefs):
    u, q = _case(mesh, "rot+div")
    wind = _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1)))
    dp0 = _state(q[..., None], sigma).air_mass.data

    def f(qq):
        st = TracerAirMassState(
            tracers=Field(data=qq[..., None], name="tracers", dims=("cell", "level", "tracer")),
            air_mass=Field(data=dp0, name="air_mass", dims=("cell", "level")),
            time=Field(data=jnp.asarray(0.0), name="time", dims=()))
        out = mpas_rk3_fct_step(st, DT, mesh, sigma, wind, _cfg(), coefs).tracers.data
        return jnp.sum(jnp.sin(3.0 * out))

    g = jax.grad(f)(q)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0
    check_grads(f, (q,), order=1, modes=("rev",), eps=1e-6, atol=1e-5, rtol=1e-4)


_FP32_GRAD_SCRIPT = r'''
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.core.operators_voronoi import (mpas_advection_coefficients,
                                            edge_normal_velocity_from_streamfunction)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.vertical import create_sigma_coordinate
import legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas as TM
m = create_voronoi_mesh(2, lloyd_iterations=5); sg = create_sigma_coordinate(3)
co = mpas_advection_coefficients(m)
psi = 2e7 * jax.random.normal(jax.random.PRNGKey(0), (m.nVertices,))
u = edge_normal_velocity_from_streamfunction(psi[:, None] * jnp.ones(3), m).astype(jnp.float32)
wind = lambda t, mm, s: (u, jnp.zeros((m.nCells, 4), jnp.float32))
cfg = TM.TracerTransportMPASConfig(time_integrator="mpas_rk3", horizontal_scheme="mpas_fct")
def f(qq):
    st = TM.air_mass_state(TracerState(tracers=Field(data=qq, name="t", dims=()),
                                       time=Field(data=jnp.float32(0), name="t", dims=())), 1e5, sg)
    return jnp.sum(TM.mpas_rk3_fct_step(st, 1800.0, m, sg, wind, cfg, co).tracers.data ** 2)
g = jax.grad(f)(0.5 * jnp.ones((m.nCells, 3, 1), jnp.float32))   # zero antidiffusive budget
assert g.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(g))), "non-finite fp32 gradient"
print("OK")
'''


def test_fp32_gradient_finite_at_zero_antidiffusive_budget():
    """fp32 (x64 off, separate process): a constant field has p_in = p_out =
    0 and the MPAS ratio room / (p + 1e-20); its gradient stays finite only
    because the division is a stable_div (plain division: NaN, measured)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, JAX_ENABLE_X64="0", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, "-c", _FP32_GRAD_SCRIPT], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


# ---------------------------------------------------------------------------
# Config / dispatch
# ---------------------------------------------------------------------------

def test_config_validation():
    with pytest.raises(ValueError, match="unknown horizontal_scheme"):
        validate_tracer_transport_mpas_config(TracerTransportMPASConfig(horizontal_scheme="ppm"))
    with pytest.raises(ValueError, match="mpas_rk3"):
        validate_tracer_transport_mpas_config(TracerTransportMPASConfig(horizontal_scheme="mpas_fct"))
    with pytest.raises(ValueError, match="mpas_rk3"):
        validate_tracer_transport_mpas_config(TracerTransportMPASConfig(time_integrator="mpas_rk3"))
    with pytest.raises(ValueError, match="coef_3rd_order"):
        validate_tracer_transport_mpas_config(_cfg(coef3=1.5))
    with pytest.raises(ValueError, match="finite"):
        validate_tracer_transport_mpas_config(_cfg(uwidth=float("inf")))
    with pytest.raises(ValueError, match="hyperdiff"):
        validate_tracer_transport_mpas_config(_cfg()._replace(hyperdiff_coeff=1.0))
    validate_tracer_transport_mpas_config(TracerTransportMPASConfig())
    validate_tracer_transport_mpas_config(_cfg())


@pytest.mark.parametrize("nlev", [2, NLEV])
def test_sigma_dot_must_be_on_interfaces(mesh, coefs, nlev):
    """A (nCells, nlev) sigma_dot is refused; at nlev = 2 it would otherwise
    broadcast silently (nlev - 1 = 1)."""
    sg = create_sigma_coordinate(nlev)
    q = jnp.ones((mesh.nCells, nlev, 1))
    st = air_mass_state(TracerState(tracers=Field(data=q, name="tracers", dims=("cell", "level", "tracer")),
                                    time=Field(data=jnp.asarray(0.0), name="time", dims=())), P_S, sg)
    psi = 2.0e7 * jax.random.normal(jax.random.PRNGKey(0), (mesh.nVertices,))
    u = edge_normal_velocity_from_streamfunction(psi[:, None] * jnp.ones(nlev), mesh)
    with pytest.raises(ValueError, match="nlev\\+1 interfaces"):
        mpas_rk3_fct_step(st, DT, mesh, sg, _const_wind(u, jnp.zeros((mesh.nCells, nlev))), _cfg(), coefs)


def test_centred2_tendency_refuses_mpas_fct(mesh, sigma):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import tracer_tendencies_mpas
    st = _state(_blocky(mesh), sigma)
    with pytest.raises(ValueError, match="centred2"):
        tracer_tendencies_mpas(TracerState(tracers=st.tracers, time=st.time), mesh, sigma,
                               _const_wind(_rot_wind(mesh), jnp.zeros((mesh.nCells, NLEV + 1))), _cfg())


def test_model_step_mpas_fct_and_state_type(mesh, sigma):
    u = _rot_wind(mesh)
    model = TracerTransportMPASModel(mesh, sigma, _const_wind(u, jnp.zeros((mesh.nCells, NLEV + 1))), _cfg())
    st0 = _state(_blocky(mesh), sigma)
    st = model.step(st0, DT)
    assert isinstance(st, TracerAirMassState)
    np.testing.assert_allclose(float(st.time.data), DT)
    with pytest.raises(TypeError, match="TracerAirMassState"):
        model.step(TracerState(tracers=st0.tracers, time=st0.time), DT)
