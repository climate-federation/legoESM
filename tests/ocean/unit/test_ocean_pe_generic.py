"""Shared ocean baroclinic core: live-oracle regression, budgets, AD and jit.

The oracle is each model's own tendency function at the same state and
config.  The per-grid momentum block (PV flux, advection, viscosity) is the
model's own code in both, so the oracle tests the unified terms: continuity
and w, -grad(KE + p'/rho_0), drag, explicit vertical mixing, tracer diffusion
and the sponge.  Tolerances (fp64, voted): oracle tendencies 1e-13 relative to the
field's max (exact when the reference field is zero); jit vs eager 1e-12 (XLA
fusion reorders sums; measured 1.1e-13 lat-lon, 3.7e-13 MPAS); budgets 1e-13
relative to the gross sum of the budget's terms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics import ocean_pe_generic as gen
from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as ll
from legoesm.ocean.dynamics import ocean_pe_mpas as mp
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.sponge import SpongeForcing
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

TEND_RTOL = 1e-13
JIT_RTOL = 1e-12
NO_LATERAL = dict(K_h=0.0, K_bih=0.0)
# all_on dT/dS shift, core vs model, max|new-old|/max|old| (measured on PBS): the
# core's lateral diffusion is thickness weighted, the models' is not.
LATERAL_SHIFT = {("latlon", "dT_dt"): 1.8371864066963238e-07,
                 ("latlon", "dS_dt"): 7.28375099276871e-08,
                 ("mpas", "dT_dt"): 5.700068429009425e-08,
                 ("mpas", "dS_dt"): 2.0672873015807817e-08}
BUDGET_RTOL = 1e-13
DT = 300.0
LL_ALL_ON = dict(K_h=1.0e3, K_bih=1.0e13, implicit_vertical_mixing=False, A_v=1.0e-2,
                 K_v=1.0e-3, bottom_drag_r=1.0e-3)
MPAS_ALL_ON = dict(A_h=1.0e3, K_h=1.0e3, K_bih=1.0e13, implicit_vertical_mixing=False,
                   A_v=1.0e-2, K_v=1.0e-3, bottom_drag_r=1.0e-3)


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def z_coord(_x64):
    return create_ocean_z_star(n_levels=5, H_max=4000.0, dz_surface=50.0, dz_deep=1500.0)


@pytest.fixture(scope="module")
def grid(_x64):
    return create_latlon_grid(n_lat=12, n_lon=24)


@pytest.fixture(scope="module")
def mesh(_x64):
    return create_voronoi_mesh(subdivision_level=2)


def _latlon_state(grid, z_coord):
    s = rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=4000.0, land_lat_threshold=60.0)
    rng = np.random.default_rng(1)
    um, vm, m = s.u_mask.data[..., None], s.v_mask.data[..., None], s.land_mask.data
    nl = s.T.data.shape[-1]
    u = 0.3 * rng.standard_normal(s.u.data.shape)
    u[:, -1] = u[:, 0]   # duplicated periodic face carries the same velocity
    np.testing.assert_array_equal(np.asarray(s.u_mask.data[:, 0]), np.asarray(s.u_mask.data[:, -1]))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u) * um),
        v=s.v.replace(data=jnp.asarray(0.2 * rng.standard_normal(s.v.data.shape)) * vm),
        T=s.T.replace(data=s.T.data + jnp.asarray(rng.standard_normal(s.T.data.shape))
                      * m[..., None]),
        S=s.S.replace(data=s.S.data + jnp.asarray(0.1 * rng.standard_normal(s.S.data.shape))
                      * m[..., None]),
        eta=s.eta.replace(data=jnp.asarray(0.5 * rng.standard_normal(m.shape)) * m),
    ), nl


def _mpas_state(mesh, z_coord):
    s = rest_state_mpas_ocean(mesh, z_coord, H_max=4000.0, land_lat_threshold=60.0)
    rng = np.random.default_rng(2)
    m = s.land_mask.data
    em = (m[mesh.cellsOnEdge[0]] * m[mesh.cellsOnEdge[1]])[:, None]
    return s._replace(
        u=s.u.replace(data=jnp.asarray(0.3 * rng.standard_normal(s.u.data.shape)) * em),
        T=s.T.replace(data=s.T.data + jnp.asarray(rng.standard_normal(s.T.data.shape))
                      * m[:, None]),
        S=s.S.replace(data=s.S.data + jnp.asarray(0.1 * rng.standard_normal(s.S.data.shape))
                      * m[:, None]),
        eta=s.eta.replace(data=jnp.asarray(0.5 * rng.standard_normal(m.shape)) * m),
    )


def _sponge(T, S, vel_refs, seed):
    rng = np.random.default_rng(seed)
    gamma = jnp.asarray(1e-5 * rng.random(T.shape[:-1]))
    refs = [jnp.asarray(0.1 * rng.standard_normal(v.shape)) for v in vel_refs]
    return SpongeForcing(gamma=gamma, T_ref=T + 1.0, S_ref=S - 0.2, u_ref=refs[0],
                         v_ref=refs[1] if len(refs) > 1 else None)


def _close(new, old, name, rtol=TEND_RTOL):
    new, old = np.asarray(new), np.asarray(old)
    scale = np.max(np.abs(old))
    err = np.max(np.abs(new - old))
    assert np.isfinite(err), name
    if scale == 0.0:
        assert err == 0.0, (name, err)
    else:
        assert err <= rtol * scale, (name, err / scale)


def _latlon_case(grid, z_coord, cname):
    state, _ = _latlon_state(grid, z_coord)
    if cname == "default":
        return state, LatLonCGridOceanConfig.from_flat(), None
    if cname == "c2":
        return state, LatLonCGridOceanConfig.from_flat(ke_gradient_scheme="c2"), None
    sponge = _sponge(state.T.data, state.S.data, (state.u.data, state.v.data), 3)
    kw = dict(LL_ALL_ON, **(NO_LATERAL if cname == "no_lateral" else {}))
    return state, LatLonCGridOceanConfig.from_flat(**kw), sponge


def _mpas_case(mesh, z_coord, cname):
    state = _mpas_state(mesh, z_coord)
    if cname == "default":
        return state, MPASOceanConfig(), None
    if cname in ("smc03", "ahh08"):   # partial-cell PGF schemes: inactive on z*
        return state, MPASOceanConfig(pgf_scheme=cname), None
    sponge = _sponge(state.T.data, state.S.data, (state.u.data,), 4)
    kw = dict(MPAS_ALL_ON, **(NO_LATERAL if cname == "no_lateral" else {}))
    return state, MPASOceanConfig(**kw), sponge


# ------------------------------------------------------------ live oracle

def _check_oracle(new, old, fields, cname, kind):
    # all_on: dT/dS intentionally differ (thickness-weighted lateral diffusion,
    # user decision); the shift is pinned.  no_lateral keeps the dT/dS oracle
    # for explicit vertical diffusion + sponge.
    if cname in ("all_on", "no_lateral"):   # the tracer terms are active
        for f in ("dT_dt", "dS_dt"):
            assert float(jnp.max(jnp.abs(getattr(old, f).data))) > 0.0, f
    for f in fields:
        if cname == "all_on" and f in ("dT_dt", "dS_dt"):
            a, b = np.asarray(getattr(new, f).data), np.asarray(getattr(old, f).data)
            shift = np.max(np.abs(a - b)) / np.max(np.abs(b))
            np.testing.assert_allclose(shift, LATERAL_SHIFT[kind, f], rtol=1e-6)
        else:
            _close(getattr(new, f).data, getattr(old, f).data, f)


@pytest.mark.parametrize("cname", ["default", "c2", "no_lateral", "all_on"])
def test_latlon_matches_model_tendency(grid, z_coord, cname):
    state, cfg, sponge = _latlon_case(grid, z_coord, cname)
    new = ll.latlon_ocean_pe_tendencies(state, grid, z_coord, cfg, sponge=sponge, dt=DT)
    old = ll.latlon_cgrid_ocean_baroclinic_tendencies(state, grid, z_coord, cfg,
                                                      sponge=sponge, dt=DT)
    _check_oracle(new, old, ("du_dt", "dv_dt", "dT_dt", "dS_dt", "deta_dt"), cname, "latlon")


@pytest.mark.parametrize("cname", ["default", "smc03", "ahh08", "no_lateral", "all_on"])
def test_mpas_matches_model_tendency(mesh, z_coord, cname):
    state, cfg, sponge = _mpas_case(mesh, z_coord, cname)
    new = mp.mpas_ocean_pe_tendencies(state, mesh, z_coord, cfg, sponge=sponge)
    old = mp.mpas_ocean_baroclinic_tendencies(state, mesh, z_coord, cfg, sponge=sponge)
    _check_oracle(new, old, ("du_dt", "dT_dt", "dS_dt", "deta_dt", "F_slow_u"), cname, "mpas")


# ---------------------------------------------------------------- budgets

def _ops(kind, grid, mesh, z_coord, cname):
    if kind == "latlon":
        state, cfg, sponge = _latlon_case(grid, z_coord, cname)
        ops = ll.latlon_ocean_pe_operators(state, grid, z_coord, cfg, DT)
        return state, ops, sponge, (state.u.data, state.v.data), grid.area
    state, cfg, sponge = _mpas_case(mesh, z_coord, cname)
    ops = mp.mpas_ocean_pe_operators(state, mesh, z_coord, cfg)
    return state, ops, sponge, (state.u.data,), mesh.areaCell


@pytest.mark.parametrize("kind", ["latlon", "mpas"])
def test_volume_budget_closes(kind, grid, mesh, z_coord):
    """sum(area deta/dt) = 0: the flux divergence telescopes through the masks."""
    state, ops, _, vel, area = _ops(kind, grid, mesh, z_coord, "all_on")
    J, h_k, _, _ = ops.geometry_density()
    h_face, _ = ops.face_thickness(h_k, vel)
    div_k, _ = gen.continuity(vel, h_face, ops.face_masks, ops.grid_ops, z_coord)
    deta = -jnp.sum(div_k, axis=-1) * ops.cell_mask
    gross = float(jnp.sum(jnp.abs(div_k * ops.cell_mask[..., None]) * area[..., None]))
    assert gross > 0.0
    assert abs(float(jnp.sum(deta * area))) <= BUDGET_RTOL * gross


@pytest.mark.parametrize("kind", ["latlon", "mpas"])
def test_tracer_budgets_close(kind, grid, mesh, z_coord):
    """Heat and salt: the vertical diffusion conserves sum_k(h dT) per column;
    the thickness-weighted lateral operator conserves sum(area h dT); the
    sponge adds exactly its explicit source."""
    state, ops, sponge, vel, area = _ops(kind, grid, mesh, z_coord, "all_on")
    J, h_k, _, _ = ops.geometry_density()
    c = ops.coeffs
    m3 = ops.cell_mask[..., None]
    for q in (state.T.data, state.S.data):
        vd = gen.vertical_diffusion(q, c.K_v, ops.dz_ref, ops.dz_half_ref, J[..., None])
        col = jnp.sum(vd * h_k, axis=-1)
        assert float(jnp.max(jnp.abs(col))) <= BUDGET_RTOL * float(
            jnp.max(jnp.sum(jnp.abs(vd * h_k), axis=-1)))
        h_face, _ = ops.face_thickness(h_k, vel)
        lap = gen.thickness_weighted_laplacian(q, ops.grid_ops, h_face, ops.face_masks,
                                               h_k, m3)
        w = lap * h_k * area[..., None]
        assert float(jnp.sum(jnp.abs(w))) > 0.0
        assert abs(float(jnp.sum(w))) <= BUDGET_RTOL * float(jnp.sum(jnp.abs(w)))
    t = gen.ocean_baroclinic_tendencies(vel, state.T.data, state.S.data, ops, sponge=sponge)
    t0 = gen.ocean_baroclinic_tendencies(vel, state.T.data, state.S.data, ops)
    src = sponge.gamma[..., None] * (sponge.T_ref - state.T.data) * m3
    resid = (t.dT - t0.dT - src) * area[..., None]
    assert float(jnp.max(jnp.abs(resid))) <= BUDGET_RTOL * float(jnp.max(jnp.abs(src * area[..., None])))


# Thickness-weighted lateral diffusion (Laplacian + biharmonic) closes heat and
# salt where h varies (z* with a random eta).  Before the fix the models' form
# leaked 1.6e-6 (lat-lon) / 5.3e-6 (MPAS) of the gross on these fixtures.
@pytest.mark.parametrize("kind", ["latlon", "mpas"])
def test_lateral_diffusion_conserves_heat_and_salt(kind, grid, mesh, z_coord):
    state, ops, _, vel, area = _ops(kind, grid, mesh, z_coord, "all_on")
    _, h_k, _, _ = ops.geometry_density()
    assert float(jnp.max(h_k) - jnp.min(jnp.where(ops.cell_mask[..., None] > 0, h_k, jnp.inf))) > 0
    lateral_only = ops._replace(coeffs=ops.coeffs._replace(implicit_vertical_mixing=True))
    t = gen.ocean_baroclinic_tendencies(vel, state.T.data, state.S.data, lateral_only)
    for d in (t.dT, t.dS):
        w = d * h_k * area[..., None]
        assert float(jnp.sum(jnp.abs(w))) > 0.0
        assert abs(float(jnp.sum(w))) <= BUDGET_RTOL * float(jnp.sum(jnp.abs(w)))


# --------------------------------------------------------------- AD / jit

@pytest.mark.parametrize("kind", ["latlon", "mpas"])
def test_gradients_and_jit_parity(kind, grid, mesh, z_coord):
    state, _, sponge, vel, _ = _ops(kind, grid, mesh, z_coord, "all_on")
    cfg = (_latlon_case if kind == "latlon" else _mpas_case)(
        grid if kind == "latlon" else mesh, z_coord, "all_on")[1]

    def tend(vel, T, S):
        # Operators rebuilt from the perturbed state: T and S reach density and p'.
        st = state._replace(T=state.T.replace(data=T), S=state.S.replace(data=S))
        o = (ll.latlon_ocean_pe_operators(st, grid, z_coord, cfg, DT) if kind == "latlon"
             else mp.mpas_ocean_pe_operators(st, mesh, z_coord, cfg))
        return gen.ocean_baroclinic_tendencies(vel, T, S, o, sponge=sponge)

    args = (vel, state.T.data, state.S.data)
    eager = tend(*args)
    # Every output (dvel, dT, dS, deta, MPAS F_slow) weighted to O(1), so none is
    # swamped in the scalar check_grads differentiates.
    leaves = jax.tree.leaves(eager)
    rng = np.random.default_rng(7)
    wts = [jnp.asarray(rng.standard_normal(x.shape)) / float(jnp.max(jnp.abs(x))) for x in leaves]

    def f(*a):
        return sum(jnp.sum(w * x) for w, x in zip(wts, jax.tree.leaves(tend(*a))))

    check_grads(f, args, order=1, modes=("rev",))
    jitted = jax.jit(tend)(*args)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(jitted)):
        _close(b, a, "jit", JIT_RTOL)


# --------------------------------------------------------------- refusals

def test_refuses_outside_pilot(grid, mesh, z_coord):
    state, _ = _latlon_state(grid, z_coord)
    for kw in (dict(momentum_advection="flux_form"), dict(coriolis_scheme="explicit_ab2"),
               dict(bottom_drag_r=1e-3, bottom_drag_bbl_thickness=10.0),
               dict(bottom_drag_scheme="nemo_quadratic"), dict(pgf_scheme="smc03"),
               dict(surface_stress_nemo_fold_vector_sign=True)):
        with pytest.raises(ValueError, match="not in the shared-core pilot"):
            ll.latlon_ocean_pe_operators(state, grid, z_coord,
                                         LatLonCGridOceanConfig.from_flat(**kw))
    ms = _mpas_state(mesh, z_coord)
    for kw in (dict(pgf_scheme="zero"), dict(pgf_scheme="typo"),
               dict(bottom_drag_scheme="nemo_quadratic")):
        with pytest.raises(ValueError, match="not in the pilot"):
            mp.mpas_ocean_pe_operators(ms, mesh, z_coord, MPASOceanConfig(**kw))
    sponge_mp = _sponge(ms.T.data, ms.S.data, (ms.u.data,), 5)
    full_rank = sponge_mp._replace(gamma=jnp.ones_like(ms.T.data) * 1e-5)
    with pytest.raises(ValueError, match="full-rank"):
        mp.mpas_ocean_pe_tendencies(ms, mesh, z_coord, MPASOceanConfig(), sponge=full_rank)
