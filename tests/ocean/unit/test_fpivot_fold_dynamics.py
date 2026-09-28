"""F-pivot fold (eORCA1 layout, halo row stripped): the OPEN fold-line v-face.

On the F-pivot layout the stored top v row is NEMO's fold line: prognostic and
self-mapped, ``v(i) = -v(P_V i)``.  NEMO computes it with the interior formula
(reading the lbc'd ghost row) and then re-imposes the identity with
``lbc_lnk('V', -1)``.  If every operator reads fold-consistent ghosts, the
fold-line momentum tendency is ALREADY antisymmetric and the lbc is a no-op to
round-off; an operator that walls the fold or reads a one-column-off partner
breaks the antisymmetry.  These tests measure it without the lbc.

Grid: regular lat-lon band -60..80 N with the F-pivot descriptor
(``create_synthetic_tripole_fpivot``; every row uniform in longitude, so the
geometry itself is fold-symmetric), partial cells, random bathymetry, land on
the top row, card-like dynamics (SMC03 PGF, Hollingsworth KE, no-slip,
superbee, RK3 momentum, adaptive implicit vertical advection, NEMO quadratic
bottom drag).
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    compute_vertex_mask,
    curl_vertex_cgrid,
    fpivot_fold_line,
    interp_u_to_vface_4pt,
    upwind_cell_to_vface,
)
from legoesm.grids.tripole import create_synthetic_tripole_fpivot  # noqa: E402

N_LAT, N_LON, NLEV = 14, 24, 5


@pytest.fixture(autouse=True)
def _fp64_policy():
    """Round-off must be float64 round-off: the model stores float32 by
    default, which would bury a real asymmetry under 1e-7 noise."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.parallel.distributed import reset_distributed_topology
    # A backend armed by an earlier test in the same worker would route the
    # serial step through MPI reductions (seen under pytest -n).
    reset_distributed_topology()
    orig = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig)
CARD = dict(pgf_scheme="smc03", ke_gradient_scheme="hollingsworth",
            lateral_side_bc="no_slip", tracer_advection="superbee",
            momentum_time_integrator="rk3", adaptive_implicit_vertadv=True,
            bottom_drag_scheme="nemo_quadratic", A_h=2e4, C_smag_lap=0.0)


def _grid():
    lat = np.deg2rad(np.linspace(-60.0, 80.0, N_LAT))
    return create_synthetic_tripole_fpivot(N_LAT, N_LON, lat_1d=jnp.asarray(lat),
                                           dtype=jnp.float64)


def _setup(cfg_kw):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate)
    grid = _grid()
    rng = np.random.default_rng(0)
    land = np.ones((N_LAT, N_LON))
    land[0:2] = 0.0
    land[5, 3:7] = 0.0
    land[-1, 2] = 0.0                     # a land cell on the top row
    H = (4000.0 - 1500.0 * rng.random((N_LAT, N_LON))) * land
    zc = create_partial_cell_coordinate(
        create_ocean_z_star(n_levels=NLEV, H_max=4000.0), jnp.asarray(H))
    cfg = LatLonCGridOceanConfig.from_flat(**cfg_kw)
    st = rest_state_latlon_cgrid_ocean(grid, zc, H_max=4000.0,
                                       land_mask_override=jnp.asarray(land),
                                       H_bathy_override=jnp.asarray(H))
    u = 0.1 * rng.standard_normal(st.u.data.shape)
    u[:, -1] = u[:, 0]
    v = 0.1 * rng.standard_normal(st.v.data.shape)
    v[0] = 0.0
    v = fpivot_fold_line(jnp.asarray(v), grid.fold, point="V", sign=-1.0)
    T = np.asarray(st.T.data) + 0.3 * rng.standard_normal(st.T.data.shape)
    eta = 0.05 * rng.standard_normal(st.eta.data.shape)
    st = st._replace(
        u=st.u.replace(data=jnp.asarray(u) * st.u_mask.data[..., None]),
        v=st.v.replace(data=v * st.v_mask.data[..., None]),
        T=st.T.replace(data=jnp.asarray(T)),
        eta=st.eta.replace(data=jnp.asarray(eta) * st.land_mask.data))
    return grid, zc, cfg, st


def _antisym_defect(row, perm):
    a = np.asarray(row)
    right = np.asarray(perm) < np.arange(a.shape[0])
    return np.abs(a + a[np.asarray(perm)])[right].max(), np.abs(a).max()


def test_fold_line_face_and_vertex_masks_open_and_symmetric():
    grid, _, _, st = _setup(CARD)
    P = np.asarray(grid.fold.perm_v)
    m = np.asarray(st.land_mask.data)
    vm = np.asarray(st.v_mask.data)
    np.testing.assert_array_equal(vm[-1], m[-1] * m[-1][P])
    assert vm[-1].sum() > 0
    q = np.asarray(compute_vertex_mask(st.land_mask.data, grid))
    PF = np.asarray(grid.fold.perm_f)
    np.testing.assert_array_equal(q[-1, :-1], q[-1, :-1][PF])
    assert q[-1].sum() > 0


def test_fold_line_helpers_are_fold_consistent():
    """Face values across the fold: symmetric scalars, antisymmetric u-at-v,
    P_F-symmetric fold-line vorticity."""
    grid = _grid()
    rng = np.random.default_rng(3)
    P, PF = np.asarray(grid.fold.perm_v), np.asarray(grid.fold.perm_f)
    f = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 2)))
    flux = jnp.asarray(rng.standard_normal((N_LAT + 1, N_LON, 2)))
    flux = fpivot_fold_line(flux, grid.fold, point="V", sign=-1.0)
    fv = np.asarray(upwind_cell_to_vface(f, flux, grid))[-1]
    np.testing.assert_array_equal(fv, fv[P])
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        interp_to_v_points, tvd_to_v_points)
    for out in (interp_to_v_points(f, grid), tvd_to_v_points(f, flux, grid)):
        o = np.asarray(out)[-1]
        np.testing.assert_allclose(o, o[P], rtol=0, atol=1e-14)
    u = rng.standard_normal((N_LAT, N_LON + 1, 2))
    u[:, -1] = u[:, 0]
    uv = np.asarray(interp_u_to_vface_4pt(jnp.asarray(u), grid))[-1]
    np.testing.assert_allclose(uv, -uv[P], rtol=0, atol=1e-15)
    z = np.asarray(curl_vertex_cgrid(jnp.asarray(u), flux, grid))[-1, :-1]
    np.testing.assert_allclose(z, z[PF], rtol=1e-12, atol=0)
    # Independent assembly (NEMO rot at F(k-1, jpj-1), our vertex k): east
    # minus west fold-line v, plus top u minus the signed U ghost -u[-1][P_U].
    PU = np.asarray(grid.fold.perm_u)
    dxu, dyv = np.asarray(grid.dx_u), np.asarray(grid.dy_v)
    vv = np.asarray(flux)[-1]
    k = np.arange(N_LON)
    circ = (vv[k] * dyv[-1, k, None] - vv[k - 1] * dyv[-1, k - 1, None]
            + u[-1, k] * dxu[-1, k, None] + u[-1, PU] * dxu[-1, PU, None])
    ref = circ / np.asarray(grid.area_q)[-1, k, None]
    np.testing.assert_allclose(z, ref, rtol=1e-12, atol=0)


def test_fold_line_momentum_tendency_is_antisymmetric():
    """Every momentum term at the fold line is antisymmetric under P_V (no lbc)."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies)
    grid, zc, cfg, st = _setup(CARD)
    P = grid.fold.perm_v
    tend, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        st, grid, zc, cfg, diagnose_momentum=True, dt=300.0)
    d, m = _antisym_defect(tend.dv_dt.data[-1], P)
    assert m > 0.0
    assert d <= 1e-12 * m, (d, m)
    nonzero = 0
    for name in diag._fields:
        if not name.endswith("_v"):
            continue
        x = getattr(diag, name)
        x = getattr(x, "data", x)
        if x is None or np.ndim(x) < 2 or np.shape(x)[0] != N_LAT + 1:
            continue
        d, m = _antisym_defect(np.asarray(x)[-1], P)
        nonzero += m > 0
        assert d <= 1e-12 * max(m, 1e-300), (name, d, m)
    # PGF+KE, vorticity, lateral viscosity, bottom drag all act at the fold
    assert nonzero >= 4


@pytest.mark.parametrize("bt_cor", ["avg", "een", "implicit_cn"])
@pytest.mark.parametrize("n_steps", [100])
def test_fold_symmetry_survives_100_steps_without_lbc(n_steps, bt_cor,
                                                      monkeypatch):
    """The model's post-step fold lbc is DISABLED; the fold-line v must stay
    antisymmetric to round-off by itself for 100 steps of the full model."""
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mod
    # fix_eta_drift off: that fixer removes any global-mean eta drift and
    # would hide a fold leak from the volume check below.
    # "implicit_cn": the production card's barotropic solver (jacobi
    # preconditioner, face-f Coriolis).
    extra = (dict(barotropic_solver="implicit_cn",
                  barotropic_implicit_preconditioner="jacobi")
             if bt_cor == "implicit_cn" else dict(barotropic_coriolis=bt_cor))
    grid, zc, cfg, st = _setup(dict(CARD, fix_eta_drift=False, **extra))
    monkeypatch.setattr(mod, "fpivot_fold_line", lambda x, *a, **k: x)
    model = mod.LatLonCGridOceanModel(grid, zc, cfg)
    P = grid.fold.perm_v
    s = st
    worst = 0.0
    area = np.asarray(grid.area)
    lm = np.asarray(st.land_mask.data)

    def _vol(x):
        return float(np.sum(area * lm * np.asarray(x.eta.data)))
    vol0, scale = _vol(st), float(np.sum(area * lm))
    for _ in range(n_steps):
        s = model.step(s, 300.0)
        v = np.asarray(s.v.data)
        assert np.isfinite(v).all()
        d, m = _antisym_defect(v[-1], P)
        worst = max(worst, d / m)
    assert np.abs(np.asarray(s.v.data)[-1]).max() > 1e-3   # the fold is open
    assert worst <= 1e-12, worst
    # Volume closes through the open fold (no forcing): mean eta drift at
    # round-off, not a pair-wise leak.
    assert abs(_vol(s) - vol0) / scale <= 1e-13, (_vol(s) - vol0) / scale


@pytest.mark.parametrize("scheme", ["min", "nemo_avg"])
def test_fold_line_vertex_thickness_is_perm_f_symmetric(scheme):
    """The fold-line vertex k is bounded by the top cells k-1, k and their
    fold images; any thickness rule over those four is P_F-symmetric."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_vertex
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
    grid = _grid()
    rng = np.random.default_rng(7)
    h = rng.uniform(1.0, 50.0, (N_LAT, N_LON, 3))
    h[-1, 5, 2] = 0.0                                   # a dry cell on the top row
    PF = np.asarray(grid.fold.perm_f)
    hv, _, _ = een_e3f_h_vtx(jnp.asarray(h), None, None, grid, scheme,
                             dz_ref=jnp.asarray([10.0, 20.0, 30.0]))
    top = np.asarray(hv)[-1, :-1]
    # "min" is exact; "nemo_avg" sums the same four cells in a mirrored
    # order, so it agrees to round-off.
    np.testing.assert_allclose(top, top[PF], rtol=1e-14, atol=0)
    mv = np.asarray(min_cell_to_vertex(jnp.asarray(h), grid))[-1, :-1]
    np.testing.assert_array_equal(mv, mv[PF])


def test_noslip_fmask_fold_row_is_perm_f_symmetric():
    """NEMO rn_shlat fmask on the fold-line vertex row reads faces from both
    sides of the fold (NEMO lbc_lnk 'F', +1), so it is P_F-symmetric."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import nemo_fmask_shlat_3d
    grid = _grid()
    rng = np.random.default_rng(11)
    act = rng.random((N_LAT, N_LON, 3)) > 0.35
    act[:, 0] = act[:, -2]
    act[:, -1] = act[:, 1]
    fm = np.asarray(nemo_fmask_shlat_3d(jnp.asarray(act), grid, 2.0))[-1, :-1]
    PF = np.asarray(grid.fold.perm_f)
    assert (fm == 2.0).any()
    np.testing.assert_array_equal(fm, fm[PF])
