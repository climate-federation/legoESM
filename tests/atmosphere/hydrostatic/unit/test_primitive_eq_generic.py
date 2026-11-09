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


def _ll_pair(state, grid, coord, cfg, dt=600.0):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import cgrid_latlon_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_latlon_hydrostatic_tendencies,
    )
    return (cgrid_latlon_hydrostatic_tendencies(state, grid, coord, cfg, dt=dt),
            cgrid_latlon_pe_tendencies(state, grid, coord, cfg, dt=dt))


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


def _mpas_pair(state, mesh, coord, cfg, dt=600.0):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    return (mpas_hydrostatic_tendencies(state, mesh, coord, cfg, dt=dt),
            mpas_pe_tendencies(state, mesh, coord, cfg, dt=dt))


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
        monkeypatch.setattr(pl, "cgrid_latlon_hydrostatic_tendencies", cgrid_latlon_pe_tendencies)
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
        monkeypatch.setattr(pm, "mpas_hydrostatic_tendencies", mpas_pe_tendencies)
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

def _ll_fn(kind, ppm=False):
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
            s0._replace(T=T, u=u, v=v, p_s=p_s, tracers={"q": q}), grid, coord, cfg, dt=600.0)
        return out[:4] + (out[4]["q"],)
    return f, (s0.T, s0.u, s0.v, s0.p_s, s0.tracers["q"])


def _mpas_fn(kind):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
    mesh, coord = _mesh(), _coord(kind)
    s0 = _mpas_state(mesh)
    cfg = MPASPrimitiveEquationConfig()

    def f(T, u, p_s, q):
        st = s0._replace(T=s0.T.replace(data=T), u=s0.u.replace(data=u),
                         p_s=s0.p_s.replace(data=p_s),
                         tracers={"q_v": s0.tracers["q_v"].replace(data=q)})
        t = mpas_pe_tendencies(st, mesh, coord, cfg, dt=600.0)
        return (t.du_dt.data, t.dT_dt.data, t.dp_s_dt.data,
                t.tracer_tendencies["q_v"].data)
    return f, (s0.T.data, s0.u.data, s0.p_s.data, s0.tracers["q_v"].data)


@pytest.mark.parametrize("grid_kind", ["latlon", "mpas"])
@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
def test_jit_parity_and_check_grads(grid_kind, kind):
    from jax.test_util import check_grads
    if grid_kind == "latlon":   # jit parity also on the PPM path
        fp, ap = _ll_fn(kind, ppm=True)
        for a, b in zip(fp(*ap), jax.jit(fp)(*ap)):
            assert _rel(b, a) < _TOL_SAME
    f, args = (_ll_fn if grid_kind == "latlon" else _mpas_fn)(kind)
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
@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
def test_gradient_through_100_steps(grid_kind, kind):
    """Forward-Euler 100 steps of the shared core: d(loss)/d(perturbation of T
    and tracer) is finite, non-zero and matches a central finite difference."""
    f, args = (_ll_fn if grid_kind == "latlon" else _mpas_fn)(kind)
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
    print(f"\n100-step grad {grid_kind} {kind}: AD {g:.6e} FD {fd:.6e}")
    assert np.isfinite(g) and g != 0.0
    assert abs(g - fd) / abs(fd) < 1e-5
