"""turbulence=ifs_vdf routing: the OpenIFS 49r1 VDF port (ifs_vdfmain, oracle-
tested in tests/atmosphere/test_ifs_vdf_oracle.py) as an MPAS-lane turbulence
option.  Pins the dispatch, the lane inputs (votes 9/11 in
oracle_audit/ifs_vdf/VOTES.md), the condensate tendencies reaching the
tracers, the refusals, and the filtered-orography loading."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.combined import PhysicsConfig
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.ifs_vdf import IFSVDFLaneConfig, ifs_vdf_turbulence
from legoesm.atmosphere.physics.turbulence.integration import (
    get_turbulence_fn,
    make_turbulence_physics,
    schemes_accepting_surface_flux,
)
from legoesm.driver.config import VALID_TURBULENCE

jax.config.update("jax_enable_x64", True)
DT = 600.0
TURB = TurbulenceConfig(scheme="ifs_vdf")


def test_dispatch_and_registration():
    name, fn, cfg = get_turbulence_fn(TURB)
    assert (name, fn, cfg) == ("ifs_vdf", ifs_vdf_turbulence, IFSVDFLaneConfig())
    assert "ifs_vdf" in VALID_TURBULENCE
    assert "ifs_vdf" in schemes_accepting_surface_flux()


def _mpas(sigflt=True):
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    sigma = create_sigma_coordinate(8)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0, perturbation_amplitude=0.0)
    state = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), state)
    ncol, nlev = state.T.data.shape
    lev = jnp.arange(nlev) / (nlev - 1)
    qc = jnp.broadcast_to(jnp.where(lev > 0.6, 1.0e-5, 0.0), (ncol, nlev))
    state = state._replace(tracers={"q_v": jnp.full((ncol, nlev), 0.006) * (0.2 + lev),
                                    "q_c": qc, "q_i": 0.1 * qc})
    if sigflt:
        mesh = mesh._replace(subgrid_topo_filt_stddev=jnp.linspace(0.0, 300.0, ncol))
    ps = init_physics_state(ncol, nlev, PhysicsConfig(turbulence=TURB))
    ps = ps._replace(rad_heating=jnp.full((ncol, nlev), -1.5e-5))
    f_land = jnp.where(jnp.arange(ncol) % 2 == 0, 1.0, 0.0)
    return mesh, sigma, state, ps, f_land


def test_mpas_lane_runs_and_routes_condensate():
    mesh, sigma, state, ps, f_land = _mpas()
    fn = make_turbulence_physics(TURB, "mpas", dt=DT, f_land=f_land)
    tend, _ = fn(state, mesh, sigma, phys_state=ps)
    for k in ("q_v", "q_c", "q_i"):
        d = tend.tracer_tendencies[k]
        d = np.asarray(d.data if hasattr(d, "data") else d)
        assert np.all(np.isfinite(d)), k
    assert np.any(np.asarray(tend.dT_dt.data) != 0.0)


@pytest.mark.parametrize("drop", ["sigflt", "q_i", "q_v", "rad_heating"])
def test_mpas_lane_refuses_missing_inputs(drop):
    mesh, sigma, state, ps, f_land = _mpas(sigflt=drop != "sigflt")
    if drop in ("q_i", "q_v"):
        state = state._replace(tracers={k: v for k, v in state.tracers.items() if k != drop})
    if drop == "rad_heating":
        ps = None
    fn = make_turbulence_physics(TURB, "mpas", dt=DT, f_land=f_land)
    with pytest.raises(ValueError, match=drop):
        fn(state, mesh, sigma, phys_state=ps)


def test_loader_never_substitutes_the_unfiltered_std(tmp_path):
    import xarray as xr
    from legoesm.grids.topography import load_subgrid_orography
    lat, lon = np.linspace(-89.5, 89.5, 180), np.arange(0.5, 360.0, 1.0)
    path = tmp_path / "sso.nc"
    xr.Dataset({"SSO_STDH": (("lat", "lon"), np.full((180, 360), 200.0))},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    mesh, *_ = _mpas(sigflt=False)
    with pytest.raises(KeyError, match="subgrid orography variable"):
        load_subgrid_orography(mesh, str(path), var_name="SSO_STDH_FILT", scale_check="off")
    xr.Dataset({"SSO_STDH": (("lat", "lon"), np.full((180, 360), 200.0)),
                "SSO_STDH_FILT": (("lat", "lon"), np.full((180, 360), 40.0))},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    out = load_subgrid_orography(mesh, str(path), var_name="SSO_STDH_FILT", scale_check="off")
    np.testing.assert_allclose(np.asarray(out), 40.0)


def test_driver_attaches_filtered_std_and_refuses_without_file(tmp_path):
    import xarray as xr
    from legoesm.driver.model_driver import ModelDriver
    lat, lon = np.linspace(-89.5, 89.5, 180), np.arange(0.5, 360.0, 1.0)
    path = tmp_path / "sso.nc"
    xr.Dataset({"SSO_STDH_FILT": (("lat", "lon"), np.full((180, 360), 40.0))},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    mesh, *_ = _mpas(sigflt=False)
    drv = SimpleNamespace(grid=mesh, _voronoi_layout=None)
    with pytest.raises(ValueError, match="subgrid_orography_path"):
        ModelDriver._attach_ifs_vdf_sigflt(drv, "", jnp.float64)
    ModelDriver._attach_ifs_vdf_sigflt(drv, str(path), jnp.float64)
    np.testing.assert_allclose(np.asarray(drv.grid.subgrid_topo_filt_stddev), 40.0)


def test_filtered_std_survives_partition_pad_and_reorder():
    """Every mesh rebuild that carries the GWD's SSO std carries the filtered one
    identically (local mesh, sharding pad, sharding reorder, local pad)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import _pad_local_mesh_to
    from legoesm.parallel.voronoi_partition import (
        _pad_voronoi_for_sharding, build_local_mesh, partition_voronoi_mesh,
        reorder_voronoi_for_sharding)
    mesh = create_voronoi_mesh(2, lloyd_iterations=0)
    f = jnp.linspace(1.0, 2.0, int(mesh.nCells))
    mesh = mesh._replace(subgrid_topo_stddev=f, subgrid_topo_filt_stddev=f)
    lm = build_local_mesh(mesh, partition_voronoi_mesh(mesh, 2, 0))
    outs = [lm, _pad_voronoi_for_sharding(mesh, 7), reorder_voronoi_for_sharding(mesh, 7),
            reorder_voronoi_for_sharding(mesh, 7, edge_order="block"),
            _pad_local_mesh_to(lm, int(lm.nCells) + 3, int(lm.nEdges) + 5, int(lm.nVertices) + 4)]
    for m in outs:
        assert m.subgrid_topo_filt_stddev is not None
        np.testing.assert_array_equal(np.asarray(m.subgrid_topo_filt_stddev),
                                      np.asarray(m.subgrid_topo_stddev))
