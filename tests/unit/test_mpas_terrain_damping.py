"""MPAS terrain options and per-edge damping on the shared PE core (ENSO fork
port, Isma Di Carlo): flux-form Zalesak FCT tracer transport, K_h temperature
diffusion on pressure surfaces, and per-edge damping scaled by mesh density.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    damping_reference_stiffness,
    edge_damping_scales,
)
from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
    fct_in_out_sums,
    tracer_horizontal_flux_fct,
)
from legoesm.core.field import Field
from legoesm.core.operators_voronoi import divergence_cell_3d
from legoesm.core.state import MPASHydrostaticState
from legoesm.grids.vertical import make_cam6_l32_levels, pressure_from_hybrid
from legoesm.grids.voronoi import create_voronoi_mesh

MESH = create_voronoi_mesh(2, lloyd_iterations=5)
DT = 300.0
C1, C2 = MESH.cellsOnEdge[0], MESH.cellsOnEdge[1]


def _her_fct(q, u_edge, dp, dp_edge, mesh, dt):
    """Reference: the fork's ``tracer_horizontal_flux_fct`` (e0226a4a8b),
    verbatim but for the helper inlined."""
    def cell_net_outflow(flux_edge):
        eoc = mesh.edgesOnCell
        sign = jnp.where(eoc >= 0, mesh.edgeSignOnCell, 0).astype(flux_edge.dtype)
        g = flux_edge[jnp.maximum(eoc, 0)]
        sign = sign.reshape(sign.shape + (1,) * (g.ndim - 2))
        return jnp.sum(sign * g, axis=0)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    m = mesh.areaCell[:, None, None] * jnp.maximum(dp, 1e-10)[..., None]
    Fm = (u_edge * dp_edge * mesh.dvEdge[:, None])[..., None]
    q1, q2 = q[c1], q[c2]
    F_low = Fm * jnp.where(Fm >= 0, q1, q2)
    F_high = Fm * (0.5 * (q1 + q2))
    divFm = cell_net_outflow(Fm)
    q_td = q + dt * (q * divFm - cell_net_outflow(F_low)) / m
    coc = mesh.cellsOnCell
    ok = (coc >= 0)[..., None, None]
    coc_safe = jnp.maximum(coc, 0)
    lo, hi = jnp.minimum(q, q_td), jnp.maximum(q, q_td)
    q_min = jnp.minimum(jnp.min(jnp.where(ok, lo[coc_safe], jnp.inf), axis=0), lo)
    q_max = jnp.maximum(jnp.max(jnp.where(ok, hi[coc_safe], -jnp.inf), axis=0), hi)
    A = F_high - F_low
    eoc = mesh.edgesOnCell
    s = jnp.where(eoc >= 0, mesh.edgeSignOnCell, 0).astype(q.dtype)[..., None, None]
    out = s * A[jnp.maximum(eoc, 0)]
    p_in = jnp.sum(jnp.maximum(-out, 0.0), axis=0)
    p_out = jnp.sum(jnp.maximum(out, 0.0), axis=0)
    room_in = jnp.maximum(q_max - q_td, 0.0) * m / dt
    room_out = jnp.maximum(q_td - q_min, 0.0) * m / dt
    r_plus = jnp.where(p_in > 0, jnp.minimum(1.0, room_in / jnp.where(p_in > 0, p_in, 1.0)), 0.0)
    r_minus = jnp.where(p_out > 0, jnp.minimum(1.0, room_out / jnp.where(p_out > 0, p_out, 1.0)), 0.0)
    C = jnp.where(A >= 0, jnp.minimum(r_minus[c1], r_plus[c2]),
                  jnp.minimum(r_plus[c1], r_minus[c2]))
    return (q * divFm - cell_net_outflow(F_low + C * A)) / m


def _fct_case(seed, nlev=4, ntr=2, band=False, rest=False):
    """Random tracer (30 % empty cells), winds and layer thicknesses.
    ``band``: a third of the edges at |u| ~ the upwind width and a patch of
    near-equal tracer (relative spread 1e-5 < the default width), so the
    smooth switches act; ``rest``: u = 0."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 1e-2, (MESH.nCells, nlev, ntr))
    q[rng.random(q.shape) < 0.3] = 0.0                      # empty cells next to full
    u = 30.0 * rng.standard_normal((MESH.nEdges, nlev))
    if band:
        slow = rng.random(MESH.nEdges) < 0.33
        u[slow] = 2e-3 * rng.standard_normal((int(slow.sum()), nlev))
        patch = np.asarray(MESH.latCell) > 0.6
        q[patch] = 5e-3 * (1.0 + 1e-5 * rng.standard_normal(q[patch].shape))
    if rest:
        u[:] = 0.0
    dp = 1000.0 + 200.0 * rng.random((MESH.nCells, nlev))
    dp, q, u = jnp.asarray(dp), jnp.asarray(q), jnp.asarray(u)
    return q, u, dp, 0.5 * (dp[C1] + dp[C2])


def _fct(case, width=1.0e-4, upwind=1.0e-3):
    q, u, dp, dpe = case
    return tracer_horizontal_flux_fct(q, u, dp, dpe, MESH, DT, width, upwind)


# Measured relative gap max|dq(default) - dq(hard)| / max|dq(hard)| per case
# (RULE 5; re-measure if a smooth form or a default width changes), and the
# measured bound excursion (min of q_new - q_min_hard_global, below 0 = undershoot,
# relative to max q) at the default.
_CASES = {"r0": dict(seed=0), "r1": dict(seed=1), "b2": dict(seed=2, band=True),
          "b3": dict(seed=3, band=True)}
_FCT_GAPS = {"r0": 0.003462604087337507, "r1": 0.003449479153346413,
             "b2": 0.00689659509245864, "b3": 0.0037763303124857556}
# all from the smooth upwind: the smooth limiter with the hard upwind gives 0.0
_FCT_UNDERSHOOT = {"r0": -8.097241276928989e-19, "r1": -5.697561895741597e-19,
                   "b2": -1.9647838550013313e-08, "b3": -2.0417052776945792e-08}


def _rel_gap(name):
    case = _fct_case(**_CASES[name])
    hard = _fct(case, 0.0, 0.0)
    return float(jnp.max(jnp.abs(_fct(case) - hard)) / jnp.max(jnp.abs(hard)))


def _undershoot(name):
    case = _fct_case(**_CASES[name])
    q = case[0]
    return float((q + DT * _fct(case)).min() / q.max())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fct_hard_limit_matches_the_fork(seed):
    q, u, dp, dpe = case = _fct_case(seed, band=seed == 2)
    ours = _fct(case, 0.0, 0.0)
    ref = _her_fct(q, u, dp, dpe, MESH, DT)
    # same arithmetic up to the order of the divisions: 6.3e-15 of the largest
    # tendency measured on case 0
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12 * float(jnp.max(jnp.abs(ref))))


@pytest.mark.parametrize("name", list(_CASES))
def test_fct_default_gap_per_case(name):
    assert _rel_gap(name) == pytest.approx(_FCT_GAPS[name], rel=1e-6, abs=1e-12)
    assert _undershoot(name) == pytest.approx(_FCT_UNDERSHOOT[name], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("kw", [dict(seed=3, band=True), dict(seed=4, rest=True)])
def test_fct_gradient_at_the_default(kw):
    q, u, dp, dpe = _fct_case(nlev=2, ntr=1, **kw)
    # strictly positive: at an exactly zero operand the relative bands vanish
    # and the switch is the hard one (KNOWN DIFFERENCE in the docstring)
    q = jnp.maximum(q, 1e-4)
    r = jnp.asarray(np.random.default_rng(9).standard_normal(q.shape))

    def f(q, v, width=1.0e-4, upwind=1.0e-3):   # v = u / 1e4: a 1e-6 m/s step at eps 1e-10
        return jnp.sum(r * tracer_horizontal_flux_fct(q, 1e4 * v, dp, dpe, MESH, DT, width, upwind))

    u = u / 1e4

    assert float(jnp.abs(jax.grad(f, argnums=1)(q, u)).max()) > 0.0
    # the step must stay inside the bands (relative 1e-4 of q ~ 5e-7, upwind
    # 1e-3 m/s) for the difference quotient to resolve the C1 forms
    check_grads(f, (q, u), order=1, modes=("fwd", "rev"), eps=1e-11, atol=1e-6, rtol=1e-4)
    if kw.get("band"):
        # non-vacuous: the hard limiter's kinks inside the step fail it here
        # (measured: AD 0.0256 vs difference quotient 0.0336)
        with pytest.raises(AssertionError):
            check_grads(lambda q, u: f(q, u, 0.0, 0.0), (q, u), order=1, modes=("fwd",),
                        eps=1e-11, atol=1e-6, rtol=1e-4)


def test_fct_in_out_sums_bound_the_hard_sums_from_above():
    rng = np.random.default_rng(11)
    out = jnp.asarray(rng.standard_normal((6, 50)) * np.where(rng.random((6, 50)) < 0.5, 1.0, 1e-5))
    p_in, p_out = fct_in_out_sums(out, 1.0e-4)
    h_in, h_out = fct_in_out_sums(out, 0.0)
    assert bool(jnp.all(p_in >= h_in)) and bool(jnp.all(p_out >= h_out))
    assert bool(jnp.any(p_in > h_in))                                    # the band acts
    np.testing.assert_array_equal(h_in, jnp.sum(jnp.maximum(-out, 0.0), axis=0))
    # codex's case: out = [1, 5e-5, -1e-5] has inflow 1e-5, never less
    p_in, _ = fct_in_out_sums(jnp.asarray([[1.0], [5e-5], [-1e-5]]), 1.0e-4)
    assert float(p_in[0]) >= 1e-5


def test_fct_in_out_sums_are_c1_where_a_flux_reverses():
    """One flux crosses zero while another sits inside its band: left and right
    derivatives agree (an L1 band scale sum |out| kinks here, codex review)."""
    base = jnp.asarray([[1.0], [3e-5], [0.0]])
    t = jnp.asarray([[0.0], [0.0], [1.0]])

    def f(h, sums=fct_in_out_sums):
        return float(sums(base + h * t, 1.0e-4)[0][0])

    def kink(sums):
        h = 1e-11
        right, left = (f(h, sums) - f(0.0, sums)) / h, (f(0.0, sums) - f(-h, sums)) / h
        return abs(right - left) / max(abs(right), abs(left))

    assert kink(fct_in_out_sums) < 1e-5

    def l1_sums(out, width):            # planted regression: L1 band scale
        band = width * jnp.sum(jnp.abs(out), axis=0)
        inside = jnp.abs(-out) < band
        wb = jnp.where(inside, band, 1.0)
        part = jnp.where(inside, (-out + wb) ** 2 / (4.0 * wb), jnp.maximum(-out, 0.0))
        return jnp.sum(part, axis=0), None

    assert kink(l1_sums) > 1e-5


def test_fct_smooth_limiter_only_limits_more():
    """Each smooth piece of the limiter errs toward limiting: with the hard
    upwind, the default widths keep the update inside the hard bounds."""
    for name in ("b2", "b3"):
        q, u, dp, dpe = case = _fct_case(**_CASES[name])
        assert float((q + DT * _fct(case, 1.0e-4, 0.0)).min()) >= -1e-15 * float(q.max())


def test_fct_uniform_stays_uniform_and_mass_closes():
    q, u, dp, dpe = case = _fct_case(4, band=True)
    one = jnp.full_like(q, 3.0e-3)
    assert float(jnp.abs(_fct((one, u, dp, dpe))).max()) < 1e-18
    dq = _fct(case)
    div_fm = divergence_cell_3d(u * dpe, MESH)[..., None]
    # d(q dp)/dt = dp dq/dt + q d(dp)/dt, d(dp)/dt = -div(u dp_e): sums to 0
    total = jnp.sum(MESH.areaCell[:, None, None] * (dp[..., None] * dq - q * div_fm), axis=0)
    scale = jnp.sum(MESH.areaCell[:, None, None] * jnp.abs(dp[..., None] * dq), axis=0)
    assert float(jnp.max(jnp.abs(total) / scale)) < 1e-12
    assert float(jnp.abs(_fct((q, 0.0 * u, dp, dpe))).max()) == 0.0   # no flux at rest


def test_fct_hard_limit_keeps_the_low_order_bounds():
    q, u, dp, dpe = case = _fct_case(5)
    q_new = q + DT * _fct(case, 0.0, 0.0)
    assert float(q_new.min()) >= -1e-15 and float(q_new.max()) <= float(q.max()) + 1e-15
    # non-vacuous: the unlimited centred flux form undershoots on this case
    fm = u * dpe
    div_fm = divergence_cell_3d(fm, MESH)
    centred = jnp.stack([
        q[..., i] + DT * (q[..., i] * div_fm
                          - divergence_cell_3d(fm * 0.5 * (q[C1, :, i] + q[C2, :, i]), MESH)) / dp
        for i in range(q.shape[-1])], axis=-1)
    assert float(centred.min()) < -1e-12


@pytest.mark.parametrize("depth,match", [(3, True), (2, False)])
def test_fct_partition_parity_needs_two_complete_rings(depth, match):
    from legoesm.parallel.voronoi_mpi import make_voronoi_partition_layout
    # sparse spikes: a halo cell's low-order update is an extremum of its
    # neighbour's bounds only next to an isolated value (measured: depth 2 off
    # by 2.8e-4 of the largest tendency at 16 ranks, 0 at 4 and 8)
    q, u, dp, dpe = _fct_case(6)
    q = jnp.asarray(1e-2 * (np.random.default_rng(6).random(q.shape) < 0.08))
    case = (q, u, dp, dpe)
    glob = _fct(case)
    worst = 0.0
    for rank in range(16):
        lay = make_voronoi_partition_layout(MESH, rank, 16, halo_depth=depth)
        lc, le = np.asarray(lay.partition.local_cells), np.asarray(lay.partition.local_edges)
        loc = tracer_horizontal_flux_fct(q[lc], u[le], dp[lc], dpe[le], lay.local_mesh, DT)
        own = np.asarray(lay.owned_mask_cells)
        worst = max(worst, float(jnp.abs(loc[own] - glob[lc][own]).max()))
    scale = float(jnp.abs(glob).max())
    assert (worst <= 1e-12 * scale) == match, worst / scale


def test_fct_jitted_model_step():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel
    cfg = MPASPrimitiveEquationConfig(vertical_form="as83", tracer_horizontal_scheme="flux_fct")
    model = MPASPrimitiveEquationModel(MESH, SIGMA, cfg)
    st = model.step(_state(), dt=DT)
    assert bool(jnp.all(jnp.isfinite(st.tracers["q_v"].data)))


def test_fct_refuses_nonpositive_dt():
    q, u, dp, dpe = _fct_case(0)
    with pytest.raises(ValueError, match="dt > 0"):
        tracer_horizontal_flux_fct(q, u, dp, dpe, MESH, 0.0)


# ---------------------------------------------------------------- shared core
SIGMA = make_cam6_l32_levels()
NLEV = SIGMA.B_full.shape[0]


def _state(terrain_m=2000.0, tracers=True):
    lat, lon = jnp.asarray(MESH.latCell), jnp.asarray(MESH.lonCell)
    z_s = terrain_m * jnp.exp(-((lat - 0.3) ** 2 + (lon - 1.0) ** 2) / 0.2)
    phis = constants.g * z_s
    p_s = 1.0e5 * jnp.exp(-phis / (constants.R_d * 280.0))
    p_full = pressure_from_hybrid(SIGMA, p_s)
    T = 288.0 * (p_full / 1.0e5) ** 0.19                    # a function of p alone
    rng = np.random.default_rng(7)
    u = jnp.asarray(5.0 * rng.standard_normal((MESH.nEdges, NLEV)))
    tr = None
    if tracers:
        qv = jnp.asarray(1e-2 * rng.random((MESH.nCells, NLEV)))
        tr = {"q_v": Field(data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")}
    return MPASHydrostaticState(
        u=Field(data=u, name="u", dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=T, name="T", dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=p_s, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis, name="phis", dims=("nCells",), units="m2/s2"),
        tracers=tr)


def _tend(cfg, state=None, dt=DT):
    return mpas_pe_tendencies(state if state is not None else _state(), MESH, SIGMA,
                              cfg, dt=dt, vertical_form="as83")


def test_flux_fct_reaches_the_tracers_only():
    base = MPASPrimitiveEquationConfig(vertical_form="as83")
    a, b = _tend(base), _tend(base._replace(tracer_horizontal_scheme="flux_fct"))
    assert jnp.array_equal(a.dT_dt.data, b.dT_dt.data)
    assert not jnp.allclose(a.tracer_tendencies["q_v"].data, b.tracer_tendencies["q_v"].data)
    with pytest.raises(ValueError, match="dt > 0"):
        _tend(base._replace(tracer_horizontal_scheme="flux_fct"), dt=0.0)
    with pytest.raises(ValueError, match="tracer_horizontal_scheme"):
        _tend(base._replace(tracer_horizontal_scheme="fct"))


def test_kh_on_pressure_surfaces_removes_the_terrain_diffusion():
    base = MPASPrimitiveEquationConfig(vertical_form="as83")
    k = base._replace(K_h=1.0e5)
    d0 = _tend(base).dT_dt.data
    model = _tend(k).dT_dt.data - d0
    press = _tend(k._replace(K_h_pressure_surfaces=True)).dT_dt.data - d0
    ratio = float(jnp.abs(press).max() / jnp.abs(model).max())
    assert float(jnp.abs(model).max()) > 0.0
    assert ratio == pytest.approx(_KH_RATIO, rel=1e-6)
    with pytest.raises(ValueError, match="K_h > 0"):
        _tend(base._replace(K_h_pressure_surfaces=True))


# Measured: on pressure surfaces the K_h tendency of a T(p) column over a
# 2000 m mountain is 4.6 % of the model-surface one (leading term only).
_KH_RATIO = 0.04611595575219175


def _damped(cfg):
    return cfg._replace(nu_del2=1.0e5, nu_del4=1.0e15, nu_div4=1.0e15,
                        nu_del4_ps=1.0e15, K_h=1.0e5)


def test_edge_damping_is_bitwise_off_on_a_uniform_mesh():
    base = _damped(MPASPrimitiveEquationConfig(vertical_form="as83"))
    on = base._replace(edge_scaled_damping=True,
                       damping_ref_stiffness=damping_reference_stiffness(MESH))
    f2, f4 = edge_damping_scales(MESH, on)
    assert bool(jnp.all(f2 == 1.0)) and bool(jnp.all(f4 == 1.0))
    a, b = _tend(base), _tend(on)
    for x, y in ((a.du_dt, b.du_dt), (a.dT_dt, b.dT_dt), (a.dp_s_dt, b.dp_s_dt)):
        assert jnp.array_equal(x.data, y.data)


def _variable_mesh():
    lat = np.asarray(MESH.latCell)
    rho = np.minimum(1.0, 0.05 + 1.9 * np.exp(-((lat - 0.5) / 0.4) ** 2))
    return MESH._replace(meshDensity=jnp.asarray(rho))


def test_edge_damping_scales_follow_density_under_the_cap():
    mesh = _variable_mesh()
    cfg = MPASPrimitiveEquationConfig(edge_scaled_damping=True,
                                      damping_ref_stiffness=damping_reference_stiffness(mesh))
    f2, f4 = edge_damping_scales(mesh, cfg)
    rho = np.asarray(mesh.meshDensity)
    rho_e = 0.5 * (rho[np.asarray(C1)] + rho[np.asarray(C2)])
    # the mesh is geometrically uniform, so the stiffness cap (not the nominal
    # density spacing) sets the coarse edges' factor
    assert float(np.max(np.asarray(f2))) > 1.2                      # coarse edges damp more
    assert np.all(np.asarray(f2) <= rho_e ** -0.5 * (1 + 1e-6))
    assert np.all(np.asarray(f4) <= rho_e ** -1.0 * (1 + 1e-6))
    np.testing.assert_allclose(np.asarray(f4), np.asarray(f2) ** 2, rtol=1e-6)
    with pytest.raises(ValueError, match="damping_ref_stiffness"):
        edge_damping_scales(mesh, cfg._replace(damping_ref_stiffness=0.0))
    off = _damped(MPASPrimitiveEquationConfig(vertical_form="as83"))
    on = off._replace(edge_scaled_damping=True,
                      damping_ref_stiffness=cfg.damping_ref_stiffness)
    a, b = (mpas_pe_tendencies(_state(), mesh, SIGMA, c, dt=DT) for c in (off, on))
    assert not jnp.allclose(a.du_dt.data, b.du_dt.data)
    assert not jnp.allclose(a.dT_dt.data, b.dT_dt.data)
    assert not jnp.allclose(a.dp_s_dt.data, b.dp_s_dt.data)


def test_edge_damping_cap_binds_on_a_stiff_edge():
    cfg = MPASPrimitiveEquationConfig(edge_scaled_damping=True,
                                      damping_ref_stiffness=0.5 * damping_reference_stiffness(MESH))
    f2, _ = edge_damping_scales(_variable_mesh(), cfg)
    assert float(jnp.min(f2)) < 1.0                                  # cap below density factor


@pytest.mark.parametrize("field", ["tracer_horizontal_scheme", "K_h_pressure_surfaces",
                                   "edge_scaled_damping"])
def test_legacy_vertical_form_refuses_the_new_options(field):
    val = {"tracer_horizontal_scheme": "flux_fct"}.get(field, True)
    cfg = MPASPrimitiveEquationConfig(vertical_form="legacy", K_h=1.0e5,
                                      damping_ref_stiffness=1.0, **{field: val})
    with pytest.raises(ValueError, match="shared PE core only"):
        mpas_pe_tendencies(_state(), MESH, SIGMA, cfg, dt=DT)


# ---------------------------------------------------------------- MPI
def test_mpi_halo_depth_and_refusal():
    from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_step, required_halo_depth
    assert required_halo_depth("flux_fct") == 3 and required_halo_depth("advective") == 2
    cfg = MPASPrimitiveEquationConfig(tracer_horizontal_scheme="flux_fct")
    with pytest.raises(ValueError, match="complete halo rings"):
        make_voronoi_mpi_step(SimpleNamespace(config=cfg),
                              SimpleNamespace(complete_cell_rings=1), SIGMA)


def test_spmd_step_refuses_flux_fct():
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
    cfg = MPASPrimitiveEquationConfig(tracer_horizontal_scheme="flux_fct")
    with pytest.raises(ValueError, match="not certified"):
        make_voronoi_sharded_step(SimpleNamespace(config=cfg, step=None),
                                  SimpleNamespace(n_devices=2, mesh=object()))


# ---------------------------------------------------------------- decks
_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_DECKS = {
    "amip_production_flux_fct.yaml": ("amip_production.yaml",
                                      {"mpas_tracer_horizontal_scheme": "flux_fct"}),
    "amip_production_kh_pressure.yaml": ("amip_production.yaml",
                                         {"mpas_kh_pressure_surfaces": True}),
    "amip_production_edge_damping.yaml": ("amip_production_saved_mesh.yaml",
                                          {"mpas_edge_scaled_damping": True}),
}


def _cfg(deck):
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


def _errors(cfg):
    try:
        cfg.validate_strict()
    except ValueError as exc:
        return set(str(exc).splitlines())
    return set()


@pytest.mark.parametrize("name", _DECKS)
def test_deck_differs_from_its_base_only_in_its_row(name):
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    base_name, rows = _DECKS[name]
    base, deck = read_yaml_with_includes(_AMIP / base_name), read_yaml_with_includes(_AMIP / name)
    assert {k for k in base.keys() | deck.keys() if base.get(k) != deck.get(k)} == set(rows)
    assert all(deck[k] == v for k, v in rows.items())


@pytest.mark.parametrize("name", _DECKS)
def test_deck_reaches_the_dycore_and_validates(name):
    from legoesm.driver.component_factory import create_atmosphere_dycore
    base_name, rows = _DECKS[name]
    base, cfg = _cfg(_AMIP / base_name), _cfg(_AMIP / name)
    assert _errors(cfg) <= _errors(base)
    m = create_atmosphere_dycore(cfg, MESH, SIGMA).config
    assert m.vertical_form == "as83"
    if "mpas_tracer_horizontal_scheme" in rows:
        assert m.tracer_horizontal_scheme == "flux_fct"
        assert (m.fct_smoothing_width, m.fct_upwind_width_ms) == (1.0e-4, 1.0e-3)
    elif "mpas_kh_pressure_surfaces" in rows:
        assert m.K_h_pressure_surfaces and m.K_h > 0.0
    else:
        assert m.edge_scaled_damping and m.damping_ref_stiffness > 0.0
        with pytest.raises(ValueError, match="meshDensity"):
            create_atmosphere_dycore(
                cfg, MESH._replace(meshDensity=0.5 * jnp.ones(MESH.nCells)), SIGMA)


def test_validate_strict_refuses_bad_settings():
    prod = _cfg(_AMIP / "amip_production.yaml")
    d = prod.dycore
    for repl, word in (({"mpas_tracer_horizontal_scheme": "fct"}, "mpas_tracer_horizontal_scheme"),
                       ({"mpas_fct_smoothing_width": -1.0}, "mpas_fct_smoothing_width"),
                       ({"mpas_fct_upwind_width_ms": float("nan")}, "mpas_fct_upwind_width_ms"),
                       ({"mpas_damping_exp_del4": -1.0}, "mpas_damping_exp_del4"),
                       ({"mpas_kh_pressure_surfaces": True, "vertical_form": "legacy"},
                        "shared PE core only")):
        errs = _errors(prod._replace(dycore=d._replace(**repl))) - _errors(prod)
        assert any(word in e for e in errs), (repl, errs)


def test_cli_flags_round_trip():
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    cfg = run_amip.build_config_from_args(p.parse_args([
        "--mpas-tracer-horizontal-scheme", "flux_fct", "--mpas-fct-smoothing-width", "0",
        "--mpas-fct-upwind-width-ms", "0.01",
        "--mpas-kh-pressure-surfaces", "--mpas-edge-scaled-damping",
        "--mpas-damping-exp-del2", "1.5", "--mpas-damping-exp-del4", "3.0"]))
    d = cfg.dycore
    assert d.mpas_fct_upwind_width_ms == 0.01
    assert (d.mpas_tracer_horizontal_scheme, d.mpas_fct_smoothing_width,
            d.mpas_kh_pressure_surfaces, d.mpas_edge_scaled_damping,
            d.mpas_damping_exp_del2, d.mpas_damping_exp_del4) == (
        "flux_fct", 0.0, True, True, 1.5, 3.0)
