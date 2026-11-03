"""turbulence=ysu routing: WRF surface layer (sfclayrev) -> WRF YSU.

Both kernels are oracle-tested on their own (test_sfclayrev_oracle.py,
test_ysu_oracle.py).  This file pins what joins them (R4 votes X/J in
oracle_audit/vote_1103_r4): the WRF module_pbl_driver mapping (ust, hfx, qfx,
br, psim=fm, psih=fh, wspd, znt, u10, v10), the hard sharpness limit, the
carried surface-layer state, the forced-flux path, the land/water and grid
spacing inputs, the refusals, and the lane plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics._shared import exner_function
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.combined import PhysicsConfig
from legoesm.atmosphere.physics.turbulence.config import (
    SurfaceLayerConfig,
    TurbulenceConfig,
    YSUConfig,
)
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
    turbulence_carry_field,
    turbulence_scheme_traits,
)
from legoesm.atmosphere.physics.turbulence.sfclayrev_wrf import (
    SfclayState,
    sfclayrev_run,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import surface_moisture_flux
from legoesm.atmosphere.physics.turbulence.ysu import (
    SFCLAY_FIELDS,
    YSUSurfaceLayer,
    bl_ysu_run,
    init_sfclay_carry,
    ysu_turbulence,
)

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parents[2] / "oracle_fixtures" / "ysu.npz"
DT = 600.0
CFG = YSUConfig(surface_type="water", dx_m=2.5e4)


@pytest.fixture(scope="module")
def cols():
    f = dict(np.load(FIXTURE))
    zi = jnp.asarray(f["in_zi"])
    d = {k: jnp.asarray(f[f"in_{k}"]) for k in ("u", "v", "T", "q", "p_full", "p_half")}
    d["z_half"] = zi
    d["z_full"] = 0.5 * (zi[:, :-1] + zi[:, 1:])
    # a skin 2 K warmer than the lowest air, its saturation humidity as q_sfc
    d["T_sfc"] = d["T"][:, -1] + 2.0
    d["q_sfc"] = d["q"][:, -1] + 2e-3
    d["rho"] = jnp.ones_like(d["T"])   # not read by the kernel
    return d


def _call(c, cfg=CFG, carry=None, **kw):
    ncol = c["T"].shape[0]
    carry = init_sfclay_carry(ncol, cfg) if carry is None else carry
    return ysu_turbulence(c["u"], c["v"], c["T"], c["q"], carry, c["p_full"], c["p_half"],
                          c["z_full"], c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"], DT,
                          cfg, **kw)


def _chain(c, cfg, carry, xland, dx, forced=None):
    """sfclayrev_run then bl_ysu_run, wired as WRF's module_pbl_driver does."""
    ncol = c["T"].shape[0]
    k = {n: carry[:, i] for i, n in enumerate(SFCLAY_FIELDS)}
    water = xland >= 2.0
    st = SfclayState(ust=k["ust"], mol=k["mol"],
                     znt=jnp.where(water, k["znt"], cfg.surface.z0),
                     hfx=k["hfx"] if forced is None else forced[0],
                     qfx=k["qfx"] if forced is None else forced[1], qsfc=c["q_sfc"])
    dz = c["z_half"][:, :-1] - c["z_half"][:, 1:]
    s = sfclayrev_run(c["u"][:, -1], c["v"][:, -1], c["T"][:, -1], c["q"][:, -1],
                      c["p_full"][:, -1], dz[:, -1], c["p_half"][:, -1], c["T_sfc"], xland,
                      jnp.ones(ncol), k["pblh"], dx, st, scm_force_flux=forced is not None)
    big = float(jnp.finfo(jnp.float64).max)
    o = bl_ysu_run(c["u"], c["v"], c["T"], c["q"], c["p_full"], c["p_half"],
                   exner_function(c["p_full"]), dz, s.ust, s.hfx, s.qfx,
                   YSUSurfaceLayer(br=s.br, psim=s.fm, psih=s.fh, wspd=s.wspd, znt=s.znt,
                                   u10=s.u10, v10=s.v10, xland=xland), DT,
                   pbl_sharpness=big, region_sharpness=big, ri_sharpness=big,
                   entrainment_cutoff_sharpness=big)
    return s, o


def test_route_is_the_two_oracle_kernels_chained(cols):
    ncol = cols["T"].shape[0]
    carry = init_sfclay_carry(ncol, CFG)
    out, new = _call(cols, carry=carry)
    s, o = _chain(cols, CFG, carry, jnp.full(ncol, 2.0), jnp.full(ncol, 2.5e4))
    for a, b in ((out.du_dt, o.du_dt), (out.dT_dt, o.dT_dt), (out.dq_v_dt, o.dq_v_dt),
                 (out.Kh, o.exch_h), (out.Km, o.exch_m), (out.h_pbl, o.hpbl),
                 (out.shflx, s.hfx), (out.lhflx, s.lh), (out.evap_sfc, s.qfx),
                 (out.ustar, s.ust)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    want = jnp.stack([s.ust, s.mol, s.znt, s.hfx, s.qfx, o.hpbl], axis=1)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(want))
    assert float(jnp.max(jnp.abs(out.dT_dt))) > 0.0 and float(jnp.max(o.hpbl)) > 0.0
    k = np.asarray(out.Kh_iface)
    np.testing.assert_array_equal(k[:, 0], 0.0)
    np.testing.assert_array_equal(k[:, -1], 0.0)
    np.testing.assert_array_equal(k[:, 1:-1], np.asarray(out.Kh)[:, :-1])


def test_hard_limit_is_not_the_smooth_mode(cols):
    """ysu_turbulence runs the oracle's hard switches, not the smooth widths."""
    ncol = cols["T"].shape[0]
    carry = init_sfclay_carry(ncol, CFG)
    out, _ = _call(cols, carry=carry)
    s, _ = _chain(cols, CFG, carry, jnp.full(ncol, 2.0), jnp.full(ncol, 2.5e4))
    dz = cols["z_half"][:, :-1] - cols["z_half"][:, 1:]
    soft = bl_ysu_run(cols["u"], cols["v"], cols["T"], cols["q"], cols["p_full"],
                      cols["p_half"], exner_function(cols["p_full"]), dz, s.ust, s.hfx,
                      s.qfx, YSUSurfaceLayer(br=s.br, psim=s.fm, psih=s.fh, wspd=s.wspd,
                                             znt=s.znt, u10=s.u10, v10=s.v10,
                                             xland=jnp.full(ncol, 2.0)), DT,
                      pbl_sharpness=20.0, region_sharpness=10.0, ri_sharpness=100.0,
                      entrainment_cutoff_sharpness=2.0)
    assert float(jnp.max(jnp.abs(soft.hpbl - out.h_pbl))) > 1.0


def test_carry_threads_state_between_steps(cols):
    ncol = cols["T"].shape[0]
    c0 = init_sfclay_carry(ncol, CFG)
    np.testing.assert_array_equal(np.asarray(c0[:, 0]), 1e-4)      # WRF cold-start ust
    np.testing.assert_array_equal(np.asarray(c0[:, 2]), CFG.surface.z0)
    np.testing.assert_array_equal(np.asarray(c0[:, [1, 3, 4, 5]]), 0.0)
    o1, c1 = _call(cols, carry=c0)
    o2, c2 = _call(cols, carry=c1)
    o2_cold, _ = _call(cols, carry=c0)
    # the second step depends on the first (ust averaging, mol, znt, pblh)
    assert float(jnp.max(jnp.abs(o2.ustar - o2_cold.ustar))) > 1e-6
    np.testing.assert_array_equal(np.asarray(o2_cold.ustar), np.asarray(o1.ustar))
    assert not np.array_equal(np.asarray(c1), np.asarray(c2))


def test_forced_flux_path_keeps_prescribed_fluxes(cols):
    ncol = cols["T"].shape[0]
    sh = jnp.linspace(-20.0, 300.0, ncol)
    lh = jnp.linspace(0.0, 400.0, ncol)
    cfg = CFG._replace(surface=SurfaceLayerConfig()._replace(
        prescribed_shflx_w_m2=sh, prescribed_lhflx_w_m2=lh))
    out, carry = _call(cols, cfg=cfg)
    qf = surface_moisture_flux(cfg.surface, lh, cols["T_sfc"])
    np.testing.assert_array_equal(np.asarray(out.shflx), np.asarray(sh))
    np.testing.assert_array_equal(np.asarray(out.lhflx), np.asarray(lh))
    np.testing.assert_array_equal(np.asarray(out.evap_sfc), np.asarray(qf))
    s, o = _chain(cols, cfg, init_sfclay_carry(ncol, cfg), jnp.full(ncol, 2.0),
                  jnp.full(ncol, 2.5e4), forced=(sh, qf))
    np.testing.assert_array_equal(np.asarray(out.dT_dt), np.asarray(o.dT_dt))
    ev = jnp.full(ncol, 1e-5)
    cfg_e = cfg._replace(surface=cfg.surface._replace(prescribed_evap_kg_m2_s=ev))
    np.testing.assert_array_equal(np.asarray(_call(cols, cfg=cfg_e)[0].evap_sfc),
                                  np.asarray(ev))


@pytest.mark.parametrize("over,match", [
    (dict(prescribed_tau_x_pa=0.1, prescribed_tau_y_pa=0.0), "prescribed surface stress"),
    (dict(prescribed_shflx_w_m2=10.0), "BOTH"),
    (dict(prescribed_lhflx_w_m2=10.0), "BOTH"),
])
def test_unsupported_prescriptions_refused(cols, over, match):
    cfg = CFG._replace(surface=SurfaceLayerConfig()._replace(**over))
    with pytest.raises(ValueError, match=match):
        _call(cols, cfg=cfg)


def test_land_water_and_grid_spacing_inputs(cols):
    ncol = cols["T"].shape[0]
    lf = jnp.where(jnp.arange(ncol) % 2 == 0, 0.5, 0.49)
    bare = YSUConfig()
    carry = init_sfclay_carry(ncol, bare)
    out, _ = _call(cols, cfg=bare, carry=carry, land_fraction=lf,
                   cell_area=jnp.full(ncol, 2.5e4 ** 2))
    xland = jnp.where(lf >= 0.5, 1.0, 2.0)
    s, _ = _chain(cols, bare, carry, xland, jnp.full(ncol, 2.5e4))
    np.testing.assert_array_equal(np.asarray(out.ustar), np.asarray(s.ust))
    assert set(np.asarray(xland).tolist()) == {1.0, 2.0}
    with pytest.raises(ValueError, match="surface_type"):
        _call(cols, cfg=bare._replace(dx_m=1e4))
    with pytest.raises(ValueError, match="dx_m"):
        _call(cols, cfg=bare._replace(surface_type="land"))
    with pytest.raises(ValueError, match="surface_type"):
        _call(cols, cfg=bare._replace(surface_type="ocean", dx_m=1e4))
    with pytest.raises(ValueError, match="inert"):
        _call(cols, land_fraction=lf)
    with pytest.raises(ValueError, match="inert"):
        _call(cols, cell_area=jnp.full(ncol, 1e8))


def test_missing_or_wrong_carry_refused(cols):
    ncol = cols["T"].shape[0]
    with pytest.raises(ValueError, match="sfclay"):
        _call(cols, carry=jnp.zeros((ncol, 5)))
    with pytest.raises(ValueError, match="sfclay"):
        ysu_turbulence(cols["u"], cols["v"], cols["T"], cols["q"], None, cols["p_full"],
                       cols["p_half"], cols["z_full"], cols["z_half"], cols["T_sfc"],
                       cols["q_sfc"], cols["rho"], DT, CFG)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_gradients_finite(cols, dtype):
    c = {k: v.astype(dtype) for k, v in cols.items()}
    ncol = c["T"].shape[0]
    carry = init_sfclay_carry(ncol, CFG, dtype)

    def loss(T, T_sfc, u):
        d = dict(c, T=T, T_sfc=T_sfc, u=u)
        o, new = _call(d, carry=carry)
        return (jnp.sum(o.dT_dt) * 1e4 + jnp.sum(o.du_dt) * 1e4 + jnp.sum(o.shflx)
                + jnp.sum(o.h_pbl) + jnp.sum(new))

    g = jax.grad(loss, argnums=(0, 1, 2))(c["T"], c["T_sfc"], c["u"])
    for gi in g:
        assert gi.dtype == dtype
        assert bool(jnp.all(jnp.isfinite(gi)))
    assert float(jnp.max(jnp.abs(g[1]))) > 0.0


def test_scheme_registers_its_carry():
    assert turbulence_carry_field("ysu", YSUConfig()) == "sfclay"
    assert turbulence_scheme_traits("ysu").carries_energy
    cfg = PhysicsConfig(turbulence=TurbulenceConfig(scheme="ysu"))
    ps = init_physics_state(7, 4, cfg)
    assert ps.sfclay.shape == (7, len(SFCLAY_FIELDS))
    np.testing.assert_array_equal(np.asarray(ps.sfclay),
                                  np.asarray(init_sfclay_carry(7, YSUConfig())))
    assert init_physics_state(7, 4, PhysicsConfig()).sfclay is None


def test_nonhydrostatic_lane_refuses():
    with pytest.raises(NotImplementedError, match="sfclay"):
        make_turbulence_physics(TurbulenceConfig(scheme="ysu"), "nonhydrostatic", dt=DT)


def _hydro_lane():
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    n, nlev = 4, 8
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    dims = ("face", "x", "y", "level")
    state = state._replace(
        u=Field(data=jnp.full((6, n, n, nlev), 8.0), name="u", dims=dims, units="m/s"),
        tracers={"q_v": Field(1e-3 * jnp.ones((6, n, n, nlev)), name="q_v", dims=dims,
                              units="kg/kg")})
    return state, grid, sigma


def test_hydrostatic_lane_threads_the_carry():
    state, grid, sigma = _hydro_lane()
    ncol, nlev = 6 * 4 * 4, sigma.n_levels
    turb = TurbulenceConfig(scheme="ysu", ysu=YSUConfig(surface_type="land"))
    fn = make_turbulence_physics(turb, "hydrostatic", dt=DT)
    ps = init_physics_state(ncol, nlev, PhysicsConfig(turbulence=turb))
    tend, upd = fn(state, grid, sigma, phys_state=ps)
    carry = upd["sfclay"]
    assert carry.shape == (ncol, len(SFCLAY_FIELDS))
    assert not np.array_equal(np.asarray(carry), np.asarray(ps.sfclay))
    assert bool(jnp.all(jnp.isfinite(tend.dT_dt.data)))
    with pytest.raises(ValueError, match="sfclay"):
        fn(state, grid, sigma)                     # no physics state: no reseed


def test_driver_pipeline_refuses_ysu():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="ysu")
    with pytest.raises(ValueError, match="sfclay"):
        build_physics_pipeline(create_latlon_grid(8, 16, dtype=jnp.float64),
                               create_sigma_coordinate(5), cfg)


def test_spectral_combined_refuses_sea_ice_forcing():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(8, dealiasing="quadratic")
    sigma = create_sigma_coordinate(4)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, p_s_init=1.0e5)
    turb = TurbulenceConfig(scheme="ysu")
    cfg = PhysicsConfig(turbulence=turb)
    fn = make_physics(cfg, model_type="spectral_pe", dt=DT)
    ps = init_physics_state(int(grid.n_lat) * int(grid.n_lon), 4, cfg)
    sic = jnp.zeros((int(grid.n_lat), int(grid.n_lon)))
    with pytest.raises(ValueError, match="sea ice"):
        fn(state, grid, sigma, phys_state=ps, forcing={"sic": sic})


def _mpas():
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(3)
    sigma = create_sigma_coordinate(6)
    state = held_suarez_init_mpas(mesh, sigma, seed=1)
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), state)
    return mesh, sigma, state


def test_mpas_lane_land_fraction_and_beta_refusal():
    from legoesm.atmosphere.physics.turbulence import integration as ti
    mesh, sigma, state = _mpas()
    ncol = state.T.data.shape[0]
    turb = TurbulenceConfig(scheme="ysu")
    ps = init_physics_state(ncol, sigma.n_levels, PhysicsConfig(turbulence=turb))
    f_land = jnp.where(jnp.arange(ncol) % 2 == 0, 1.0, 0.0)
    fn = ti.make_turbulence_physics(turb, "mpas", dt=DT, f_land=f_land)
    _, upd = fn(state, mesh, sigma, phys_state=ps)
    znt = np.asarray(upd["sfclay"][:, 2])
    land = np.asarray(f_land) >= 0.5
    np.testing.assert_array_equal(znt[land], YSUConfig().surface.z0)   # land keeps z0
    assert np.all(znt[~land] != YSUConfig().surface.z0)                 # Charnock over water
    fb = ti.make_turbulence_physics(turb, "mpas", dt=DT, f_land=f_land, land_beta=0.5)
    with pytest.raises(ValueError, match="surface-humidity"):
        fb(state, mesh, sigma, phys_state=ps)
