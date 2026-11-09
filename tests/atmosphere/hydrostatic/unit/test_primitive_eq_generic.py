"""Shared hydrostatic PE core (``primitive_eq_generic``) vs the two old cores.

(a) terms whose form did not change match the old lat-lon / MPAS tendencies;
    the changed terms are measured (printed) against the old solver.
(b) #1029 rest over a mountain on a hybrid coordinate, and the #930 dry-adiabat
    vertical residual (a 2dz checkerboard), on BOTH grids.
(c) JW baroclinic wave, old vs shared core in the SAME model time loop.
(d) mass and total-energy budgets over that window.
(e) jit parity, check_grads order 2, gradient through 100 steps.
Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="dycore science test needs float64 (run with JAX_ENABLE_X64=1)")

NLEV = 8


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _coord(kind, nlev=NLEV):
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    return create_sigma_coordinate(nlev) if kind == "sigma" else standard_hybrid_levels(nlev)


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(b)), 1e-300))


# ---------------------------------------------------------------------------
# lat-lon fixtures
# ---------------------------------------------------------------------------

def _ll_grid():
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(16, 32)


def _ll_state(grid, nlev=NLEV, seed=0):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
    )
    r = np.random.RandomState(seed)
    nj, ni = grid.n_lat, grid.n_lon
    v = r.randn(nj + 1, ni, nlev) * 8.0
    v[0] = v[-1] = 0.0
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(r.randn(nj, ni + 1, nlev) * 10.0),
        v=jnp.asarray(v),
        T=jnp.asarray(250.0 + 20.0 * r.rand(nj, ni, nlev)),
        p_s=jnp.asarray(1.0e5 + 2000.0 * r.randn(nj, ni)),
        phis=jnp.asarray(3000.0 * r.rand(nj, ni)),
        tracers={"q": jnp.asarray(1e-3 * r.rand(nj, ni, nlev))})


def _ll_pair(state, grid, coord, cfg, dt=600.0, form="sb81_theta"):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import cgrid_latlon_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_latlon_hydrostatic_tendencies,
    )
    return (cgrid_latlon_hydrostatic_tendencies(state, grid, coord, cfg, dt=dt),
            cgrid_latlon_pe_tendencies(state, grid, coord, cfg, dt=dt, vertical_form=form))


# ---------------------------------------------------------------------------
# MPAS fixtures
# ---------------------------------------------------------------------------

def _mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(2, lloyd_iterations=5)


def _mpas_state(mesh, nlev=NLEV, seed=0, tracers=True):
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    r = np.random.RandomState(seed)
    nC, nE = mesh.nCells, mesh.nEdges
    tr = ({"q_v": Field(data=jnp.asarray(1e-3 * r.rand(nC, nlev)), name="q_v",
                      dims=("nCells", "nlev"), units="kg/kg")} if tracers else None)
    return MPASHydrostaticState(
        u=Field(data=jnp.asarray(r.randn(nE, nlev) * 10.0), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=jnp.asarray(250.0 + 20.0 * r.rand(nC, nlev)), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=jnp.asarray(1.0e5 + 2000.0 * r.randn(nC)), name="p_s",
                  dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.asarray(3000.0 * r.rand(nC)), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=tr)


def _mpas_pair(state, mesh, coord, cfg, dt=600.0, form="sb81_theta"):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    return (mpas_hydrostatic_tendencies(state, mesh, coord, cfg, dt=dt),
            mpas_pe_tendencies(state, mesh, coord, cfg, dt=dt, vertical_form=form))


# ===========================================================================
# (a) unchanged terms match; changed terms measured
# ===========================================================================

_TOL_SAME = 1e-12   # relative to the field max, fp64


@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
@pytest.mark.parametrize("ppm", [True, False])
@pytest.mark.parametrize("damp", [False, True])
def test_latlon_unchanged_terms_match(kind, ppm, damp):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    grid, coord = _ll_grid(), _coord(kind)
    # damp also turns the zero-mean surface-pressure correction on (fix_mass off)
    extra = dict(A_h=1.0e5, nu_del4=1.0e15, nu_div4_scale=1.0, sponge_coeff=1e-4,
                 fix_mass=False) if damp else {}
    cfg = CGridLatLonPrimitiveEquationConfig(use_ppm_transport=ppm, **extra)
    old, new = _ll_pair(_ll_state(grid), grid, coord, cfg)
    # momentum (T1 already SB81 on lat-lon; E4 = same upwind column operator),
    # surface pressure and tracers do not change form.
    for name, a, b in [("du", new[0], old[0]), ("dv", new[1], old[1]),
                       ("dps", new[3], old[3]), ("dq", new[4]["q"], old[4]["q"])]:
        assert _rel(a, b) < _TOL_SAME, (name, _rel(a, b))
    # dT changes form (T2 theta-form vertical, E1 omega split, T3 on the
    # centered path, T5 no clip): measured, not asserted.
    print(f"\nLATLON {kind} ppm={ppm} damp={damp}: dT rel change = {_rel(new[2], old[2]):.3e}")


@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
@pytest.mark.parametrize("damp", [False, True])
def test_mpas_unchanged_terms_match(kind, damp):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    mesh, coord = _mesh(), _coord(kind)
    extra = dict(nu_del2=1.0e5, nu_del4=1.0e15, nu_del4_ps=1.0e14, K_h=1.0e4,
                 nu_div4=1.0e15, nu_vert4_T=1e-5, apvm_scale=0.5) if damp else {}
    cfg = MPASPrimitiveEquationConfig(**extra)
    old, new = _mpas_pair(_mpas_state(mesh), mesh, coord, cfg)
    same = [("dps", new.dp_s_dt.data, old.dp_s_dt.data),
            ("dq", new.tracer_tendencies["q_v"].data, old.tracer_tendencies["q_v"].data)]
    if kind == "sigma":
        # sigma: only E5 (1/p guard) touches dT, and p_full > p_floor here.
        same += [("du", new.du_dt.data, old.du_dt.data),
                 ("dT", new.dT_dt.data, old.dT_dt.data)]
    for name, a, b in same:
        assert _rel(a, b) < _TOL_SAME, (name, _rel(a, b))
    if kind == "hybrid":   # T1 (SB81 PGF), T4 (same discrete ln p), E5
        print(f"\nMPAS hybrid damp={damp}: du rel change = "
              f"{_rel(new.du_dt.data, old.du_dt.data):.3e}, dT rel change = "
              f"{_rel(new.dT_dt.data, old.dT_dt.data):.3e}")


def test_mpas_sigma_bit_identity_report():
    """How far from bit-identical MPAS sigma is (summation order only)."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    mesh, coord = _mesh(), _coord("sigma")
    old, new = _mpas_pair(_mpas_state(mesh), mesh, coord, MPASPrimitiveEquationConfig())
    for name in ("du_dt", "dT_dt", "dp_s_dt"):
        a, b = getattr(new, name).data, getattr(old, name).data
        print(f"\nMPAS sigma {name}: bit-identical={bool(jnp.all(a == b))}, max rel {_rel(a, b):.2e}")
        assert _rel(a, b) < _TOL_SAME


def test_changed_terms_on_smooth_jw_state():
    """Changed-term size on a smooth (JW, sigma) state, not random noise."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_latlon,
        baroclinic_wave_init_mpas,
    )
    coord = _coord("sigma", 15)
    grid = _ll_grid()
    s = hydrostatic_to_cgrid(baroclinic_wave_init_latlon(grid, coord), grid)
    old, new = _ll_pair(s, grid, coord, CGridLatLonPrimitiveEquationConfig())
    mesh = _mesh()
    mo, mn = _mpas_pair(baroclinic_wave_init_mpas(mesh, coord), mesh, coord,
                        MPASPrimitiveEquationConfig())
    print(f"\nJW state dT rel change: latlon {_rel(new[2], old[2]):.3e} "
          f"(max|dT| {float(jnp.abs(old[2]).max()):.3e} K/s); MPAS sigma "
          f"{_rel(mn.dT_dt.data, mo.dT_dt.data):.3e}")


# ===========================================================================
# (b) #1029 rest over a mountain, hybrid coordinate
# ===========================================================================

def _rest_ll(coord):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import hydrostatic_to_cgrid
    from tests.test_cases.dcmip2012.rest_state_topography import (
        rest_state_topography_init_latlon,
    )
    grid = _ll_grid()
    return grid, hydrostatic_to_cgrid(rest_state_topography_init_latlon(grid, coord), grid)


def _rest_mpas(coord):
    from tests.test_cases.dcmip2012.rest_state_topography import (
        rest_state_topography_init_mpas,
    )
    mesh = _mesh()
    return mesh, rest_state_topography_init_mpas(mesh, coord)


_REST_TOL = 1e-10   # m/s^2: discrete cancellation, fp64 round-off only


def test_1029_rest_over_mountain_hybrid_both_grids():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    coord = _coord("hybrid", 20)
    grid, s = _rest_ll(coord)
    old, new = _ll_pair(s, grid, coord, CGridLatLonPrimitiveEquationConfig())
    ll_old = max(float(jnp.abs(old[0]).max()), float(jnp.abs(old[1]).max()))
    ll_new = max(float(jnp.abs(new[0]).max()), float(jnp.abs(new[1]).max()))
    mesh, sm = _rest_mpas(coord)
    mo, mn = _mpas_pair(sm, mesh, coord, MPASPrimitiveEquationConfig())
    mp_old = float(jnp.abs(mo.du_dt.data).max())
    mp_new = float(jnp.abs(mn.du_dt.data).max())
    print(f"\n#1029 max|dvel/dt| m/s^2: latlon old {ll_old:.2e} new {ll_new:.2e}; "
          f"MPAS old {mp_old:.2e} new {mp_new:.2e}")
    assert ll_new < _REST_TOL and mp_new < _REST_TOL
    assert mp_old > 1e3 * _REST_TOL      # the MPAS analytic-factor PGF was broken


# ===========================================================================
# (b) #930 dry adiabat: the vertical thermodynamic residual
# ===========================================================================

def _adiabat_T(coord, p_s):
    from legoesm import constants
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate, pressure_from_hybrid
    if isinstance(coord, HybridSigmaPressureCoordinate):
        p = pressure_from_hybrid(coord, p_s)
    else:
        p = coord.sigma_full * p_s[..., None]
    return 300.0 * (p / constants.p_ref) ** constants.kappa, p


def _amp_2dz(x):
    k = np.arange(x.shape[-1])
    return float(np.abs(np.mean(np.asarray(x) * (-1.0) ** k, axis=-1)).max())


@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
def test_930_dry_adiabat_both_grids(kind):
    """theta = const, uniform p_s, divergent wind: the exact dT/dt is
    kappa*T*omega_ps/p (theta conserved).  The theta-form leaves round-off; the
    old split form (lat-lon) leaves a 2dz residual."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    coord = _coord(kind, 24)
    res = {}
    grid = _ll_grid()
    s = _ll_state(grid, 24)._replace(p_s=jnp.full((grid.n_lat, grid.n_lon), 1.0e5),
                                     phis=jnp.zeros((grid.n_lat, grid.n_lon)), tracers={})
    T, p = _adiabat_T(coord, s.p_s)
    s = s._replace(T=T)
    old, new = _ll_pair(s, grid, coord, CGridLatLonPrimitiveEquationConfig())
    w = coord.B_full if kind == "hybrid" else coord.sigma_full
    for tag, d in (("old", old), ("new", new)):
        exact = constants.kappa * T * w * d[3][..., None] / (p + 1e-10)
        res[f"latlon_{tag}"] = (float(jnp.abs(d[2] - exact).max()), _amp_2dz(d[2] - exact))
    mesh = _mesh()
    sm = _mpas_state(mesh, 24, tracers=False)
    ps = jnp.full((mesh.nCells,), 1.0e5)
    T, p = _adiabat_T(coord, ps)
    sm = sm._replace(T=sm.T.replace(data=T), p_s=sm.p_s.replace(data=ps),
                     phis=sm.phis.replace(data=jnp.zeros_like(ps)))
    mo, mn = _mpas_pair(sm, mesh, coord, MPASPrimitiveEquationConfig())
    for tag, d in (("old", mo), ("new", mn)):
        exact = constants.kappa * T * w * d.dp_s_dt.data[..., None] / (p + 1e-10)
        r = d.dT_dt.data - exact
        res[f"mpas_{tag}"] = (float(jnp.abs(r).max()), _amp_2dz(r))
    print(f"\n#930 {kind} (max |dT - exact| K/s, 2dz amplitude): {res}")
    scale = float(jnp.abs(new[2]).max())
    assert res["latlon_new"][0] < 1e-9 * max(scale, 1e-12) + 1e-14
    assert res["mpas_new"][0] < 1e-9 * max(scale, 1e-12) + 1e-14
    assert res["latlon_old"][1] > 1e3 * max(res["latlon_new"][1], 1e-18)


# ===========================================================================
# (c, d) JW baroclinic wave, old vs shared core in the same model time loop
# ===========================================================================

_JW_DAYS = 6.0


def _energy_ll(s, coord, area):
    from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import kinetic_energy_cgrid
    from legoesm.core.conservation import dycore_column_energy
    ph = coord.pressure_at_half(s.p_s)
    dp = ph[..., 1:] - ph[..., :-1]
    e = dycore_column_energy(s.T, kinetic_energy_cgrid(s.u, s.v), dp, s.phis, s.p_s)
    return float(jnp.sum(e * area)), float(jnp.sum(s.p_s * area))


def _run_ll(generic, n_steps, dt, monkeypatch):
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pl
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import cgrid_latlon_pe_tendencies
    from legoesm.grids.latlon import create_latlon_grid
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon
    if generic:
        monkeypatch.setattr(pl, "cgrid_latlon_hydrostatic_tendencies",
                            partial(cgrid_latlon_pe_tendencies, vertical_form="sb81_theta"))
    grid, coord = create_latlon_grid(32, 64), _coord("sigma", 15)
    cfg = pl.CGridLatLonPrimitiveEquationConfig(use_polar_filter=True)
    model = pl.CGridLatLonPrimitiveEquationModel(grid, coord, cfg, dt=dt)
    s = pl.hydrostatic_to_cgrid(baroclinic_wave_init_latlon(grid, coord), grid)
    area = jnp.asarray(grid.area)
    e0 = _energy_ll(s, coord, area)
    vmax = []
    for i in range(n_steps):
        s = model.step(s, dt)
        if (i + 1) % (n_steps // 6) == 0:
            vmax.append(float(jnp.abs(s.v).max()))
    monkeypatch.undo()
    return s, e0, _energy_ll(s, coord, area), vmax


def _run_mpas(generic, n_steps, dt, monkeypatch):
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pm
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
    from legoesm.core.conservation import dycore_column_energy
    from legoesm.core.operators_voronoi import kinetic_energy_cell_3d
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    if generic:
        monkeypatch.setattr(pm, "mpas_hydrostatic_tendencies",
                            partial(mpas_pe_tendencies, vertical_form="sb81_theta"))
    mesh, coord = create_voronoi_mesh(4, lloyd_iterations=5), _coord("sigma", 15)
    model = pm.MPASPrimitiveEquationModel(mesh, coord, pm.MPASPrimitiveEquationConfig(nu_del4=1e15))
    s = baroclinic_wave_init_mpas(mesh, coord)
    area = jnp.asarray(mesh.areaCell)

    def en(st):
        ph = coord.pressure_at_half(st.p_s.data)
        e = dycore_column_energy(st.T.data, kinetic_energy_cell_3d(st.u.data, mesh),
                                 ph[..., 1:] - ph[..., :-1], st.phis.data, st.p_s.data)
        return float(jnp.sum(e * area)), float(jnp.sum(st.p_s.data * area))

    e0 = en(s)
    umax = []
    for i in range(n_steps):
        s = model.step(s, dt)
        if (i + 1) % (n_steps // 6) == 0:
            umax.append(float(jnp.abs(s.u.data).max()))
    monkeypatch.undo()
    return s, e0, en(s), umax


@pytest.mark.slow
def test_jw_baroclinic_latlon_old_vs_shared(monkeypatch):
    dt = 600.0
    n = int(_JW_DAYS * 86400 / dt)
    so, e0o, e1o, vo = _run_ll(False, n, dt, monkeypatch)
    sn, e0n, e1n, vn = _run_ll(True, n, dt, monkeypatch)
    assert all(np.isfinite(vn))
    du = float(jnp.sqrt(jnp.mean((sn.u - so.u) ** 2)))
    assert du > 0.0   # the monkeypatched core actually ran
    dps = float(jnp.abs(sn.p_s - so.p_s).max())
    print(f"\nJW latlon {_JW_DAYS} d: max|v| old {vo} new {vn}; rms du {du:.3e} m/s, "
          f"max dps {dps:.3e} Pa; energy drift old {(e1o[0]-e0o[0])/e0o[0]:.3e} "
          f"new {(e1n[0]-e0n[0])/e0n[0]:.3e}; mass drift old {(e1o[1]-e0o[1])/e0o[1]:.3e} "
          f"new {(e1n[1]-e0n[1])/e0n[1]:.3e}")
    assert abs(e1n[1] - e0n[1]) / e0n[1] < 1e-12   # fix_mass on in both


@pytest.mark.slow
def test_jw_baroclinic_mpas_old_vs_shared(monkeypatch):
    dt = 600.0
    n = int(_JW_DAYS * 86400 / dt)
    so, e0o, e1o, uo = _run_mpas(False, n, dt, monkeypatch)
    sn, e0n, e1n, un = _run_mpas(True, n, dt, monkeypatch)
    assert all(np.isfinite(un))
    du = float(jnp.sqrt(jnp.mean((sn.u.data - so.u.data) ** 2)))
    assert du > 0.0   # the monkeypatched core actually ran
    dps = float(jnp.abs(sn.p_s.data - so.p_s.data).max())
    print(f"\nJW MPAS {_JW_DAYS} d: max|u| old {uo} new {un}; rms du {du:.3e} m/s, "
          f"max dps {dps:.3e} Pa; energy drift old {(e1o[0]-e0o[0])/e0o[0]:.3e} "
          f"new {(e1n[0]-e0n[0])/e0n[0]:.3e}; mass drift old {(e1o[1]-e0o[1])/e0o[1]:.3e} "
          f"new {(e1n[1]-e0n[1])/e0n[1]:.3e}")
    assert abs(e1n[1] - e0n[1]) / e0n[1] < 1e-12


# ===========================================================================
# (d, f) one-tendency energy budget; the removed T clip
# ===========================================================================

def test_latlon_shared_core_has_no_T_clip():
    """Rule 0 / T5: the old clip only replaced the tendency's INPUT T (no sink,
    no source: the state is untouched; the step-level safety rail, unchanged,
    still clamps the state).  With cells below T_min the shared core must equal
    the old core with an inert floor and differ from the clipped old core in the
    terms whose form did not change."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    grid, coord = _ll_grid(), _coord("sigma")
    s = _ll_state(grid)
    s = s._replace(T=s.T.at[4:8, 5:9, 2].set(30.0))
    cfg = CGridLatLonPrimitiveEquationConfig()
    assert float(s.T.min()) < cfg.T_min
    clipped, new = _ll_pair(s, grid, coord, cfg)
    unclipped, _ = _ll_pair(s, grid, coord, cfg._replace(T_min=0.0))
    for k in (0, 1):   # du, dv: same form in both cores
        assert _rel(new[k], unclipped[k]) < _TOL_SAME
        assert _rel(new[k], clipped[k]) > 1e-6


def test_mass_and_energy_tendency_budgets():
    """Raw continuity closes (global p_s tendency ~ 0, no fixer); the global
    total-energy tendency of the old and shared cores on the same JW state."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import kinetic_energy_cgrid
    from legoesm.core.conservation import dycore_column_energy
    from legoesm.core.operators_voronoi import kinetic_energy_cell_3d
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_latlon,
        baroclinic_wave_init_mpas,
    )
    coord = _coord("sigma", 15)
    out = {}
    grid = _ll_grid()
    area = jnp.asarray(grid.area)
    s = hydrostatic_to_cgrid(baroclinic_wave_init_latlon(grid, coord), grid)
    cfg = CGridLatLonPrimitiveEquationConfig(zero_mean_ps_tendency=False)

    def E_ll(x):
        T, u, v, ps = x
        ph = coord.pressure_at_half(ps)
        return jnp.sum(area * dycore_column_energy(
            T, kinetic_energy_cgrid(u, v), ph[..., 1:] - ph[..., :-1], s.phis, ps))
    for tag, d in zip(("old", "new"), _ll_pair(s, grid, coord, cfg)):
        du, dv, dT, dps = d[:4]
        out[f"latlon_{tag}_mass"] = float(jnp.sum(area * dps) / jnp.sum(area * jnp.abs(dps)))
        out[f"latlon_{tag}_dE"] = float(jax.jvp(E_ll, ((s.T, s.u, s.v, s.p_s),),
                                                ((dT, du, dv, dps),))[1]) / float(E_ll((s.T, s.u, s.v, s.p_s)))
    mesh = _mesh()
    am = jnp.asarray(mesh.areaCell)
    sm = baroclinic_wave_init_mpas(mesh, coord)

    def E_mp(x):
        T, u, ps = x
        ph = coord.pressure_at_half(ps)
        return jnp.sum(am * dycore_column_energy(
            T, kinetic_energy_cell_3d(u, mesh), ph[..., 1:] - ph[..., :-1], sm.phis.data, ps))
    x0 = (sm.T.data, sm.u.data, sm.p_s.data)
    for tag, d in zip(("old", "new"), _mpas_pair(sm, mesh, coord, MPASPrimitiveEquationConfig())):
        dps = d.dp_s_dt.data
        out[f"mpas_{tag}_mass"] = float(jnp.sum(am * dps) / jnp.sum(am * jnp.abs(dps)))
        out[f"mpas_{tag}_dE"] = float(jax.jvp(E_mp, (x0,), ((d.dT_dt.data, d.du_dt.data, dps),))[1]) / float(E_mp(x0))
    print("\nbudgets (mass: global/sum|.|; dE: (dE/dt)/E [1/s]):",
          {k: f"{v:.3e}" for k, v in out.items()})
    for k in ("latlon_old_mass", "latlon_new_mass", "mpas_old_mass", "mpas_new_mass"):
        assert abs(out[k]) < 1e-12, (k, out[k])


# ===========================================================================
# (e) jit parity, check_grads order 2, gradient through 100 steps
# ===========================================================================

def _ll_fn(kind, ppm=False, form="sb81_theta"):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import cgrid_latlon_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig,
    )
    grid, coord = _ll_grid(), _coord(kind)
    s0 = _ll_state(grid)
    # PPM's Colella-Woodward limiter switches on the state (pre-existing), so
    # second derivatives and long finite differences cross its kinks.
    cfg = CGridLatLonPrimitiveEquationConfig(use_ppm_transport=ppm)

    def f(T, u, v, p_s, q):
        out = cgrid_latlon_pe_tendencies(
            s0._replace(T=T, u=u, v=v, p_s=p_s, tracers={"q": q}), grid, coord, cfg, dt=600.0,
            vertical_form=form)
        return out[:4] + (out[4]["q"],)
    return f, (s0.T, s0.u, s0.v, s0.p_s, s0.tracers["q"])


def _mpas_fn(kind, form="sb81_theta"):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    mesh, coord = _mesh(), _coord(kind)
    s0 = _mpas_state(mesh)
    cfg = MPASPrimitiveEquationConfig()

    def f(T, u, p_s, q):
        st = s0._replace(T=s0.T.replace(data=T), u=s0.u.replace(data=u),
                         p_s=s0.p_s.replace(data=p_s),
                         tracers={"q_v": s0.tracers["q_v"].replace(data=q)})
        t = mpas_pe_tendencies(st, mesh, coord, cfg, dt=600.0, vertical_form=form)
        return (t.du_dt.data, t.dT_dt.data, t.dp_s_dt.data,
                t.tracer_tendencies["q_v"].data)
    return f, (s0.T.data, s0.u.data, s0.p_s.data, s0.tracers["q_v"].data)


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
def test_default_vertical_form_is_as83(grid_kind):
    """No ``vertical_form`` = ``"as83"`` bit-for-bit; on sigma the default raises
    (no silent fallback to ``"sb81_theta"``)."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    if grid_kind == "latlon":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationConfig,
        )
        grid, cfg = _ll_grid(), CGridLatLonPrimitiveEquationConfig(use_ppm_transport=False)
        s0 = _ll_state(grid)

        def run(coord, **kw):
            return g.cgrid_latlon_pe_tendencies(s0, grid, coord, cfg, dt=600.0, **kw)[:3]
    else:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
        mesh, cfg = _mesh(), MPASPrimitiveEquationConfig()
        s0 = _mpas_state(mesh)

        def run(coord, **kw):
            t = g.mpas_pe_tendencies(s0, mesh, coord, cfg, dt=600.0, **kw)
            return t.du_dt.data, t.dT_dt.data, t.dp_s_dt.data
    assert g.DEFAULT_VERTICAL_FORM == "as83"
    for a, b in zip(jax.tree_util.tree_leaves(run(_coord("hybrid"))),
                    jax.tree_util.tree_leaves(run(_coord("hybrid"), vertical_form="as83"))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="hybrid coordinate"):
        run(_coord("sigma"))


_KIND_FORM = [("sigma", "sb81_theta"), ("hybrid", "sb81_theta"),
              ("hybrid", "as83"), ("hybrid", "as83_logmean")]


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("kind,form", _KIND_FORM)
def test_jit_parity_and_check_grads(grid_kind, kind, form):
    from jax.test_util import check_grads
    if grid_kind == "latlon":   # jit parity also on the PPM path
        fp, ap = _ll_fn(kind, ppm=True, form=form)
        for a, b in zip(fp(*ap), jax.jit(fp)(*ap)):
            assert _rel(b, a) < _TOL_SAME
    f, args = (_ll_fn if grid_kind == "latlon" else _mpas_fn)(kind, form=form)
    eager, jitted = f(*args), jax.jit(f)(*args)
    for a, b in zip(eager, jitted):
        assert _rel(b, a) < _TOL_SAME
    # random-weighted scalar of every tendency, differentiated w.r.t. every input
    w = [jnp.asarray(np.random.RandomState(i).randn(*o.shape)) for i, o in enumerate(eager)]

    scale = [float(jnp.abs(a).max()) for a in args]

    def scalar(*z):   # inputs normalised by their max, so eps is relative
        return sum(jnp.sum(wi * oi) for wi, oi in zip(w, f(*(zi * c for zi, c in zip(z, scale)))))
    # Small step: the upwind vertical advection and the tracer van Leer limiter
    # (both old cores) switch on the state, so a large step crosses their kinks.
    check_grads(scalar, tuple(a / c for a, c in zip(args, scale)), order=2,
                modes=("rev",), eps=1e-6)


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("kind,form", _KIND_FORM)
def test_gradient_through_100_steps(grid_kind, kind, form):
    """Forward-Euler 100 steps of the shared core: d(loss)/d(perturbation of T
    and tracer) is finite, non-zero and matches a central finite difference."""
    f, args = (_ll_fn if grid_kind == "latlon" else _mpas_fn)(kind, form=form)
    dt, n = 20.0, 100
    pert = jnp.asarray(np.random.RandomState(7).randn(*args[0].shape))
    pert_q = 1e-5 * jnp.asarray(np.random.RandomState(8).randn(*args[-1].shape))

    def loss(a):
        state = ((args[0] + a * pert,) + tuple(args[1:-1])
                 + (args[-1] + a * pert_q,))

        def body(st, _):
            tend = f(*st)
            if grid_kind == "latlon":
                du, dv, dT, dps, dq = tend
                incr = (dT, du, dv, dps, dq)
            else:
                du, dT, dps, dq = tend
                incr = (dT, du, dps, dq)
            return tuple(x + dt * d for x, d in zip(st, incr)), None
        st, _ = jax.lax.scan(body, state, None, length=n)
        return jnp.mean((st[0] / 250.0) ** 2) + jnp.mean((st[-1] / 1e-3) ** 2)

    g = float(jax.grad(loss)(0.0))
    h = 1e-4
    fd = float((loss(h) - loss(-h)) / (2 * h))
    print(f"\n100-step grad {grid_kind} {kind} {form}: AD {g:.6e} FD {fd:.6e}")
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) / abs(fd) < 1e-5


# ===========================================================================
# Energy instrument and semi-discrete attribution
# ===========================================================================
# E = sum_cells area * [ sum_k (c_pd T_k + K_k) dp_k / g + phis p_s / g ]
# (dycore_column_energy, the CAM ``te`` without water terms; conserved by the
# continuous adiabatic inviscid hydrostatic equations under a CONSTANT-pressure
# lid, hence the hybrid coordinate here; sigma's lid p_top = sigma_top p_s
# moves).  K_k is the lane's own kinetic energy (the one whose gradient enters
# the momentum equation).  dE/dt at one state = jvp of E along the tendency.

_E_NLEV = 20
_FORMS = ("sb81_theta", "as83", "as83_logmean")


def _energy_fn(grid_kind, ops, coord, area, phis):
    from legoesm.core.conservation import dycore_column_energy

    def E(T, vel, p_s):
        ph = coord.pressure_at_half(p_s)
        return jnp.sum(area * dycore_column_energy(
            T, ops.kinetic_energy(vel), ph[..., 1:] - ph[..., :-1], phis, p_s))
    return E


def _jw_case(grid_kind, form="sb81_theta", nlev=_E_NLEV):
    """(ops, T, vel, p_s, phis, area, coord, old_fn, (grid|mesh, cfg, state)) for
    the JW initial state on a constant-lid hybrid coordinate."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    coord = _coord("hybrid", nlev)
    if grid_kind == "latlon":
        import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pl
        from legoesm.grids.latlon import create_latlon_grid
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon
        grid = create_latlon_grid(32, 64)
        cfg = pl.CGridLatLonPrimitiveEquationConfig()
        s = pl.hydrostatic_to_cgrid(baroclinic_wave_init_latlon(grid, coord), grid)
        ops = g.latlon_pe_operators(grid, cfg, {}, 600.0, vertical_form=form)

        def old(T, vel, p_s):
            o = pl.cgrid_latlon_hydrostatic_tendencies(
                s._replace(T=T, u=vel[0], v=vel[1], p_s=p_s, tracers={}), grid, coord, cfg, dt=600.0)
            return o[2], (o[0], o[1]), o[3]
        return (ops, s.T, (s.u, s.v), s.p_s, s.phis, jnp.asarray(grid.area), coord, old,
                (grid, cfg, s))
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pm
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    mesh = create_voronoi_mesh(3, lloyd_iterations=5)
    cfg = pm.MPASPrimitiveEquationConfig()
    s = baroclinic_wave_init_mpas(mesh, coord)
    ops = g.mpas_pe_operators(mesh, cfg, None, 600.0, vertical_form=form)

    def old(T, vel, p_s):
        st = s._replace(T=s.T.replace(data=T), u=s.u.replace(data=vel[0]),
                        p_s=s.p_s.replace(data=p_s))
        o = pm.mpas_hydrostatic_tendencies(st, mesh, coord, cfg, dt=600.0)
        return o.dT_dt.data, (o.du_dt.data,), o.dp_s_dt.data
    return (ops, s.T.data, (s.u.data,), s.p_s.data, s.phis.data,
            jnp.asarray(mesh.areaCell), coord, old, (mesh, cfg, s))


def test_energy_instrument_known_answers():
    """Uniform heating c: dE/dt = c_pd c M exactly."""
    from legoesm import constants
    for gk in ("latlon", "mpas"):
        ops, T, vel, p_s, phis, area, coord, _, _ = _jw_case(gk)
        E = _energy_fn(gk, ops, coord, area, phis)
        zv = tuple(jnp.zeros_like(v) for v in vel)
        got = float(jax.jvp(E, (T, vel, p_s), (jnp.full_like(T, 1e-5), zv, jnp.zeros_like(p_s)))[1])
        ph = coord.pressure_at_half(p_s)
        M = float(jnp.sum(area * jnp.sum(ph[..., 1:] - ph[..., :-1], -1)) / constants.g)
        assert abs(got / (constants.c_pd * 1e-5 * M) - 1.0) < 1e-12


def _noisy_jw(grid_kind, form, seed=3):
    ops, T, vel, p_s, phis, area, coord, old, extra = _jw_case(grid_kind, form)
    r = np.random.RandomState(seed)
    T = T + jnp.asarray(r.randn(*T.shape))
    vel = tuple(v + jnp.asarray(r.randn(*v.shape)) for v in vel)
    if grid_kind == "latlon":   # periodic duplicate u face; polar v faces at rest (wall)
        vel = (vel[0].at[:, -1].set(vel[0][:, 0]), vel[1].at[0].set(0.0).at[-1].set(0.0))
    p_s = p_s + 100.0 * jnp.asarray(r.randn(*p_s.shape))
    return ops, T, vel, p_s, phis, area, coord, old


def _energy_groups(grid_kind, form):
    """dE/dt split into the KE group (must cancel alone) and the enthalpy +
    potential group at the JW state plus noise (identities hold for any state)."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    ops, T, vel, p_s, phis, area, coord, _ = _noisy_jw(grid_kind, form)
    dvel, dT, dps, _, ctx = g.primitive_equation_tendencies(T, vel, p_s, phis, coord, ops)
    t, gops = ctx["terms"], ops.grid_ops
    KE = ops.kinetic_energy(vel)
    dp = ctx["dp"]
    gr = lambda f: g._as_tuple(gops.gradient(f))

    def dK(piece):   # sum area sum_k dK_k dp_k / g along a velocity tendency
        return float(jax.jvp(lambda v: jnp.sum(area * jnp.sum(ops.kinetic_energy(v) * dp, -1)),
                             (vel,), (piece,))[1]) / constants.g
    cp = lambda x: float(jnp.sum(area * jnp.sum(constants.c_pd * x * dp, -1))) / constants.g
    dB = coord.dB.astype(p_s.dtype)
    out = {
        "W_gradPhi": dK(tuple(-x for x in gr(ctx["Phi"]))),
        "W_gradK": dK(tuple(-x for x in gr(KE))),
        "W_pgfL": dK(tuple(-c * dl for c, dl in zip(ctx["pgf_coef"], ctx["dL"]))),
        "W_coriolis": dK(t["coriolis"]), "W_vert_mom": dK(t["vert_mom"]),
        "H_horiz": cp(t["horiz_T"]), "H_vert": cp(t["vert_T"]),
        "H_conv_ps": cp(t["conv_ps"]), "H_conv_vgl": cp(t["conv_vgl"]),
        "M_K": float(jnp.sum(area * jnp.sum(KE * dB, -1) * dps)) / constants.g,
        "M_cpT": float(jnp.sum(area * jnp.sum(constants.c_pd * T * dB, -1) * dps)) / constants.g,
        "M_phis": float(jnp.sum(area * phis * dps)) / constants.g,
    }
    ke = out["W_gradK"] + out["W_coriolis"] + out["W_vert_mom"] + out["M_K"]
    th = sum(v for k, v in out.items() if k in (
        "W_gradPhi", "W_pgfL", "H_horiz", "H_vert", "H_conv_ps", "H_conv_vgl", "M_cpT", "M_phis"))
    E = _energy_fn(grid_kind, ops, coord, area, phis)
    total = float(jax.jvp(E, (T, vel, p_s), (dT, dvel, dps))[1])
    gross = sum(abs(v) for v in out.values())
    mass = float(jnp.sum(area * dps) / jnp.sum(area * jnp.abs(dps)))
    return out, ke, th, total, gross, mass


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("form", _FORMS)
def test_variant_energy_and_mass(grid_kind, form):
    """Semi-discrete energy residual (target round-off for the AS forms) and raw
    mass closure, noisy JW state, hybrid L20."""
    out, ke, th, total, gross, mass = _energy_groups(grid_kind, form)
    print(f"\nHARNESS energy {grid_kind} {form}: total/gross {total / gross:.3e}, "
          f"KE group {ke / gross:.3e}, thermo group {th / gross:.3e}, "
          f"Coriolis work {out['W_coriolis'] / gross:.3e}, mass {mass:.3e}")
    assert abs(ke + th - total) < 1e-10 * gross   # the recorded terms are complete
    assert abs(mass) < 1e-12
    if form != "sb81_theta":
        assert abs(total) < 1e-12 * gross


def _theta_exner(form, coord, p_s):
    """(theta-like reference, d pi/dps) for the #930 check: the scheme's own Exner."""
    from legoesm import constants
    from legoesm.grids import vertical as vt
    if form == "sb81_theta":
        p = vt.pressure_from_hybrid(coord, p_s)
        pi = (p / constants.p_ref) ** constants.kappa
        return pi, constants.kappa * pi * coord.B_full / (p + 1e-10)
    _, pi, dpi = vt.as83_layer_exner(coord, p_s)
    return pi, dpi


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("form", _FORMS)
def test_variant_930_and_1029(grid_kind, form):
    """#930: uniform theta (the scheme's own Exner), uniform p_s, divergent wind:
    dT = theta (d pi/d p_s) dp_s/dt exactly.  #1029: isothermal rest over the
    DCMIP mountain; the AS forms carry a recorded KNOWN DIFFERENCE."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    ops, T, vel, p_s, phis, area, coord, _ = _noisy_jw(grid_kind, form, seed=5)
    ps0 = jnp.full_like(p_s, 1.0e5)
    pi, dpi = _theta_exner(form, coord, ps0)
    _, dT, dps, _, _ = g.primitive_equation_tendencies(
        300.0 * pi, vel, ps0, jnp.zeros_like(ps0), coord, ops)
    res = dT - 300.0 * dpi * dps[..., None]
    r930 = float(jnp.abs(res).max()) / float(jnp.abs(dT).max())
    rc = _coord("hybrid", 20)
    if grid_kind == "latlon":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationConfig,
        )
        grid, s = _rest_ll(rc)
        o = g.latlon_pe_operators(grid, CGridLatLonPrimitiveEquationConfig(), {}, 600.0,
                                  vertical_form=form)
        dvel, *_ = g.primitive_equation_tendencies(s.T, (s.u, s.v), s.p_s, s.phis, rc, o)
    else:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
        mesh, sm = _rest_mpas(rc)
        o = g.mpas_pe_operators(mesh, MPASPrimitiveEquationConfig(), None, 600.0,
                                vertical_form=form)
        dvel, *_ = g.primitive_equation_tendencies(
            sm.T.data, (sm.u.data,), sm.p_s.data, sm.phis.data, rc, o)
    r1029 = max(float(jnp.abs(d).max()) for d in dvel)
    print(f"\nHARNESS rest {grid_kind} {form}: #930 residual/max|dT| {r930:.3e}, "
          f"#1029 max|dv/dt| {r1029:.3e} m/s^2")
    assert r930 < 1e-12
    if form == "sb81_theta":
        assert r1029 < _REST_TOL
    else:   # KNOWN DIFFERENCE, recorded in the module docstring; re-measure on change
        assert abs(r1029 / _AS_REST_GAP[(grid_kind, form)] - 1.0) < 1e-2


# measured max |dv/dt| [m/s^2], isothermal rest over the DCMIP mountain, hybrid L20
_AS_REST_GAP = {("latlon", "as83"): 1.284e-5, ("latlon", "as83_logmean"): 1.283e-5,
                ("mpas", "as83"): 9.093e-6, ("mpas", "as83_logmean"): 9.085e-6}


def test_logmean_face_weight():
    """Isothermal pairs: c_p theta_f (pi_2 - pi_1) = c_p T0 ln(pi_2/pi_1) on both
    sides of the Taylor switch; w = 1/2 at equal pressures; differs from the
    arithmetic mean (which leaves an O(x^2) error)."""
    from legoesm.grids.vertical import as83_logmean_face_weight
    T0, pi1 = 250.0, 0.8
    for x in (0.0, 3e-5, 9.9e-5, 1.01e-4, 3e-3, 0.2):
        pi2 = pi1 * (1.0 + x)
        w = float(as83_logmean_face_weight(jnp.asarray(pi1), jnp.asarray(pi2)))
        th = w * T0 / pi1 + (1.0 - w) * T0 / pi2
        if x == 0.0:
            assert w == 0.5
            continue
        xe = pi2 / pi1 - 1.0   # log1p: log(pi2/pi1) itself loses ~eps/x
        exact = T0 * np.log1p(xe) / (pi1 * xe)
        assert abs(th / exact - 1.0) < 1e-13, (x, th, exact)
        arith = 0.5 * T0 * (1.0 / pi1 + 1.0 / pi2)
        if x >= 3e-3:
            assert abs(arith / exact - 1.0) > 1e-7
    a = float(as83_logmean_face_weight(jnp.asarray(1.0), jnp.asarray(1.0 + 0.99999e-4)))
    b = float(as83_logmean_face_weight(jnp.asarray(1.0), jnp.asarray(1.0 + 1.00001e-4)))
    assert abs(a - b) < 1e-9   # the two branches meet continuously


def test_latlon_face_metrics_match_grid():
    """The operator-derived face metrics equal the grid's analytic ones."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import latlon_face_metrics
    from legoesm.grids.operator_adapters import latlon_sw_grid_operators
    grid = _ll_grid()
    l_u, d_u, l_v, d_v = latlon_face_metrics(grid, latlon_sw_grid_operators(grid))
    lat = np.asarray(grid.lat)
    R, dlon, dy = float(grid.radius), float(grid.dlon), np.asarray(grid.dy)
    np.testing.assert_allclose(l_u, 0.5 * dy, rtol=1e-12)
    np.testing.assert_allclose(d_u, R * dlon * np.cos(lat), rtol=1e-12)
    lat_v = 0.5 * (lat[:-1] + lat[1:])
    np.testing.assert_allclose(l_v[1:-1], R * dlon * np.cos(lat_v), rtol=1e-10)
    np.testing.assert_allclose(d_v[1:-1], 0.5 * (0.5 * dy[:-1] + 0.5 * dy[1:]), rtol=1e-12)
    assert float(l_v[0]) == float(l_v[-1]) == float(d_v[0]) == float(d_v[-1]) == 0.0


_E_DT = {"latlon": 60.0, "mpas": 600.0}   # lat-lon: no polar filter -> pole CFL


def _jw_energy_trajectory(grid_kind, form, dt, days):
    """Inviscid, unfiltered JW on the constant-lid hybrid coordinate in the
    model's own time loop: (E(t_end)/E0 - 1, integral of the semi-discrete
    dE/dt / E0 sampled every 6 h, final (T, vel, p_s)).  ``form="old"`` runs
    the old core."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_generic as g
    ops, T, vel, p_s, phis, area, coord, old, (obj, cfg, s) = _jw_case(
        grid_kind, "sb81_theta" if form == "old" else form)
    E = _energy_fn(grid_kind, ops, coord, area, phis)

    def new(T, vel, p_s):
        dvel, dT, dps, _, _ = g.primitive_equation_tendencies(T, vel, p_s, phis, coord, ops)
        return dT, dvel, dps
    tend = jax.jit(old if form == "old" else new)
    mp = pytest.MonkeyPatch()
    if grid_kind == "latlon":
        import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pl
        if form != "old":
            mp.setattr(pl, "cgrid_latlon_hydrostatic_tendencies",
                       partial(g.cgrid_latlon_pe_tendencies, vertical_form=form))
        model = pl.CGridLatLonPrimitiveEquationModel(obj, coord, cfg._replace(use_polar_filter=False), dt=dt)
        unpack = lambda st: (st.T, (st.u, st.v), st.p_s)
    else:
        import legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas as pm
        if form != "old":
            mp.setattr(pm, "mpas_hydrostatic_tendencies",
                       partial(g.mpas_pe_tendencies, vertical_form=form))
        model = pm.MPASPrimitiveEquationModel(obj, coord, cfg)
        unpack = lambda st: (st.T.data, (st.u.data,), st.p_s.data)

    def rate(st):
        x = unpack(st)
        return float(jax.jvp(E, x, tend(*x))[1])
    try:
        E0 = float(E(*unpack(s)))
        n = int(round(days * 86400 / dt))
        every = int(round(6 * 3600 / dt))
        rates, integ = [rate(s)], 0.0
        for i in range(n):
            s = model.step(s, dt)
            if (i + 1) % every == 0:
                rates.append(rate(s))
                integ += 0.5 * (rates[-1] + rates[-2]) * every * dt
        x1 = unpack(s)
        E1 = float(E(*x1))
    finally:
        mp.undo()
    assert np.isfinite(E1)
    return E1 / E0 - 1.0, integ / E0, x1


@pytest.mark.slow
@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("form", ("old",) + _FORMS)
@pytest.mark.parametrize("dt_factor", [1.0, 0.5])
def test_jw_energy_dt_halving(grid_kind, form, dt_factor):
    """6-day JW energy drift at dt and dt/2 (spatial = dt-independent part), and
    at dt the wave-state difference against the old core."""
    dt = _E_DT[grid_kind] * dt_factor
    drift, semi, x1 = _jw_energy_trajectory(grid_kind, form, dt, _JW_DAYS)
    msg = (f"\nEDRIFT {grid_kind} {form} dt={dt:g}: E drift {drift:.4e}, "
           f"integrated semi-discrete {semi:.4e}")
    if dt_factor == 1.0 and form != "old":
        _, _, xo = _jw_energy_trajectory(grid_kind, "old", dt, _JW_DAYS)
        du = float(jnp.sqrt(jnp.mean((x1[1][0] - xo[1][0]) ** 2)))
        dps = float(jnp.abs(x1[2] - xo[2]).max())
        msg += f"; vs old at day {_JW_DAYS:g}: rms du {du:.3e} m/s, max dps {dps:.3e} Pa"
    print(msg)
