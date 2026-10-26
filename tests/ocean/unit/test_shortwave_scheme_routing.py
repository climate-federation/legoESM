"""--sw-penetration / --water-type / --chl-clim / --ocean-albedo reach the MPAS and
tripole lanes (FESOM2 parity batch 2)."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean.physics.shortwave_penetration import (
    SWEENEY_VISIBLE_FRACTION, JERLOV_PENETRATING_FRACTION)


@pytest.fixture
def _fp64_policy_restored():
    import argparse
    from legoesm.core.precision import get_policy, set_policy
    from scripts.run import run_omip as R
    prev = get_policy()
    R.apply_run_precision(argparse.Namespace(precision="fp64"))
    try:
        yield
    finally:
        set_policy(prev)


def test_cli_round_trip():
    from scripts.run.run_omip import parse_args
    a = parse_args(["--grid", "mpas", "--sw-penetration", "sweeney_2band",
                    "--chl-clim", "/x/chl.nc", "--ocean-albedo", "0.1", "--water-type", "IB"])
    assert (a.sw_penetration, a.chl_clim, a.ocean_albedo, a.water_type) == ("sweeney_2band", "/x/chl.nc", 0.1, "IB")
    d = parse_args(["--grid", "mpas"])
    assert (d.sw_penetration, d.chl_clim, d.ocean_albedo) == ("auto", None, 0.06)
    with pytest.raises(SystemExit):
        parse_args(["--grid", "mpas", "--sw-penetration", "bogus"])


def test_mpas_setup_routes_scheme_and_water_type():
    from scripts.run import run_omip as R
    _, _, config, _, _ = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
        physics_preset="none", water_type="IB", forcing_mode="jra55_do_tropical",
        sw_scheme="jerlov_2band")
    sf = config.physics.surface_forcing
    assert (sf.shortwave_scheme, sf.shortwave_water_type) == ("jerlov_2band", "IB")
    _, _, dflt, _, _ = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=3, H_max=4000.0,
        physics_preset="none", water_type="II", forcing_mode="jra55_do_tropical")
    assert dflt.physics.surface_forcing.shortwave_scheme == "auto"


def test_tripole_setup_routes_scheme(tmp_path):
    from scripts.run import run_omip as R
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh
    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    _, _, config, _, _ = R._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "III", forcing_mode="jra55_do_tropical",
        sw_scheme="sweeney_2band", tripole_mesh=str(mesh),
        tripole_fold_convention="(n_lon-i)%n_lon")
    sf = config.physics.surface_forcing
    assert (sf.shortwave_scheme, sf.shortwave_water_type) == ("sweeney_2band", "III")


def test_driver_passes_the_call_site_symbols():
    """The consumers are the two lanes' external-forcing blocks, not the
    disabled pipeline module; pin the symbols that execute."""
    from scripts.run import run_omip as R
    from legoesm.ocean.physics import mpas_physics
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as PE
    src = inspect.getsource(mpas_physics.make_mpas_ocean_physics)
    assert 'getattr(sf_config, "shortwave_scheme", "auto")' in src
    assert "apply_shortwave_penetration(" in src and "penetrating_fraction(" in src
    src = inspect.getsource(PE._bc_external_surface_forcing)
    assert 'shortwave_scheme == "sweeney_2band"' in src
    for fn in (R._build_jra55_block_fn, R._build_jra55_block_fn_interp):
        s = inspect.getsource(fn)
        assert "sw_down=sw_net if sw_net_to_forcing else atm.sw_down" in s
        assert "chl=_chl" in s
    s = inspect.getsource(R._setup_jra55_forcing_state) if hasattr(R, "_setup_jra55_forcing_state") else inspect.getsource(R)
    assert "ocean_albedo=float(getattr(args, \"ocean_albedo\", 0.06))" in s
    assert 'state["chl_monthly"] = jnp.asarray(_load_monthly_clim_target(' in s


def test_refs_for_block_carries_the_month_chl():
    import jax.numpy as jnp
    from scripts.run import run_omip as R
    chl = jnp.arange(12.0)[:, None] * jnp.ones((1, 5))
    st = {"chl_monthly": chl}
    r = R._refs_for_block(None, st, 40.0)          # Feb
    assert float(r["chl"][0]) == 1.0
    calls = []
    def shard(x):
        calls.append(1)
        return x
    r1 = R._refs_for_block(None, st, 40.0, shard_fn=shard)
    r2 = R._refs_for_block(None, st, 100.0, shard_fn=shard)   # Apr
    assert float(r1["chl"][0]) == 1.0 and float(r2["chl"][0]) == 3.0
    assert len(calls) == 12                          # sharded once, cached
    assert R._refs_for_block(None, {}, 40.0) is None


def test_mpas_physics_dispatch_column_total_is_q_net_for_every_scheme(_fp64_policy_restored):
    """Functional: on a small MPAS mesh, every scheme deposits exactly q_net in the
    column, the schemes differ from each other, and sweeney needs chl."""
    import jax.numpy as jnp
    from scripts.run import run_omip as R
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
    from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
    mesh, z_coord, config, model, _ = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=30, H_max=300.0,
        physics_preset="none", water_type="II", forcing_mode="jra55_do_tropical")
    state = R._init_rest_state("mpas", mesh, z_coord, H_max=300.0)
    n = state.T.data.shape[0]
    sw = jnp.full((n,), 200.0); q_net = jnp.full((n,), 50.0); chl = jnp.full((n,), 0.3)
    mask = state.land_mask.data
    out = {}
    for scheme in ("auto", "jerlov_2band", "sweeney_2band"):
        phys = config.physics._replace(surface_forcing=SurfaceForcingConfig(
            scheme="none", shortwave_scheme=scheme, shortwave_water_type="II"))
        fn = make_mpas_ocean_physics(phys)
        # chl only where the scheme needs it: with chl attached "auto" is RGB
        sf = OceanSurfaceForcing(sw_down=sw, q_net=q_net, tau_x=jnp.zeros(n),
                                 tau_y=jnp.zeros(n),
                                 chl=chl if scheme == "sweeney_2band" else None)
        tend = fn(state, mesh, z_coord, surface_forcing=sf)
        dT = np.asarray(getattr(tend.dT_dt, "data", tend.dT_dt))
        h = np.asarray(z_coord.dz_ref)[None, :] * np.ones((n, 1))
        col = (rho_0_ref * c_sw * h * dT).sum(axis=1)
        np.testing.assert_allclose(col[np.asarray(mask) > 0.5], 50.0, rtol=1e-9)
        out[scheme] = dT
    np.testing.assert_allclose(out["auto"], out["jerlov_2band"])   # same type II
    # an explicit sweeney_2band must not be replaced by RGB when chl is attached
    phys = config.physics._replace(surface_forcing=SurfaceForcingConfig(
        scheme="none", shortwave_scheme="auto"))
    rgb = make_mpas_ocean_physics(phys)(state, mesh, z_coord, surface_forcing=OceanSurfaceForcing(
        sw_down=sw, q_net=q_net, tau_x=jnp.zeros(n), tau_y=jnp.zeros(n), chl=chl))
    rgb = np.asarray(getattr(rgb.dT_dt, "data", rgb.dT_dt))
    assert np.abs(out["sweeney_2band"] - rgb).max() > 1e-2 * np.abs(rgb).max()
    assert np.abs(out["sweeney_2band"] - out["auto"]).max() > 1e-2 * np.abs(out["auto"]).max()
    # sweeney routes 0.54 of the shortwave into the column, jerlov 0.94: the
    # surface cell therefore holds MORE heat under sweeney
    assert out["sweeney_2band"][:, 0][np.asarray(mask) > 0.5].mean() > out["auto"][:, 0][np.asarray(mask) > 0.5].mean()
    assert SWEENEY_VISIBLE_FRACTION < JERLOV_PENETRATING_FRACTION
    phys = config.physics._replace(surface_forcing=SurfaceForcingConfig(
        scheme="none", shortwave_scheme="sweeney_2band"))
    fn = make_mpas_ocean_physics(phys)
    with pytest.raises(ValueError, match="requires OceanSurfaceForcing.chl"):
        fn(state, mesh, z_coord, surface_forcing=OceanSurfaceForcing(
            sw_down=sw, q_net=q_net, tau_x=jnp.zeros(n), tau_y=jnp.zeros(n)))


@pytest.mark.parametrize("scheme", ["sweeney_2band", "auto", "jerlov_2band"])
@pytest.mark.parametrize("partial_top", [False, True], ids=["deep-top", "partial-top"])
def test_mpas_sweeney_respects_partial_cells(_fp64_policy_restored, scheme, partial_top):
    """With a partial-cell coordinate (the ETOPO lane), no heat lands below the
    seabed and the column still integrates to q_net (codex P1 on batch 2).
    ``partial_top`` puts the seabed at half the top reference layer in some
    columns, so the TOP cell itself is partial: the non-solar surface deposit
    must use the live top thickness too, or the column does not close."""
    import jax.numpy as jnp
    from scripts.run import run_omip as R
    from legoesm.core.field import Field
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
    from legoesm.ocean.vertical import create_partial_cell_coordinate, compute_layer_thickness
    from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
    mesh, z_coord, config, model, _ = R._create_setup(
        grid_type="mpas", resolution="ico2", nlev=30, H_max=300.0,
        physics_preset="none", water_type="II", forcing_mode="jra55_do_tropical")
    state = R._init_rest_state("mpas", mesh, z_coord, H_max=300.0)
    n = state.T.data.shape[0]
    rng = np.random.default_rng(3)
    wet_np = np.asarray(state.land_mask.data) > 0.5
    H_np = np.where(wet_np, rng.uniform(15.0, 300.0, n), 0.0)
    if partial_top:
        H_np[np.flatnonzero(wet_np)[:10]] = 0.5 * float(np.asarray(z_coord.dz_ref)[0])
    H = jnp.asarray(H_np)
    pc = create_partial_cell_coordinate(z_coord, H)
    state = state._replace(H_bathy=Field(data=H, name="H_bathy", dims=state.H_bathy.dims))
    h_live = np.asarray(compute_layer_thickness(state.eta.data, H, pc))
    phys = config.physics._replace(surface_forcing=SurfaceForcingConfig(
        scheme="none", shortwave_scheme=scheme))
    fn = make_mpas_ocean_physics(phys)
    # chl only for sweeney: with chl attached the "auto" lane runs RGB, and the
    # Jerlov kernel is what these arms must exercise.
    chl = jnp.full((n,), 0.3) if scheme == "sweeney_2band" else None
    sf = OceanSurfaceForcing(sw_down=jnp.full((n,), 200.0), q_net=jnp.full((n,), 50.0),
                             tau_x=jnp.zeros(n), tau_y=jnp.zeros(n), chl=chl)
    tend = fn(state, mesh, pc, surface_forcing=sf)
    # isolate the surface-heat part from the recipe's vertical mixing (which
    # is not level-masked on MPAS): subtract the same physics with zero heat
    base = fn(state, mesh, pc, surface_forcing=sf._replace(sw_down=jnp.zeros(n), q_net=jnp.zeros(n)))
    dT = (np.asarray(getattr(tend.dT_dt, "data", tend.dT_dt))
          - np.asarray(getattr(base.dT_dt, "data", base.dT_dt)))
    wet = np.asarray(state.land_mask.data) > 0.5
    assert np.all(dT[h_live <= 0.0] == 0.0)
    col = (rho_0_ref * c_sw * h_live * dT).sum(axis=1)
    np.testing.assert_allclose(col[wet], 50.0, rtol=1e-9)
    assert (h_live[wet] <= 0.0).any()          # the fixture really has dry levels
    if partial_top:                            # partial TOP cells really exist
        assert (h_live[wet, 0] < 0.6 * np.asarray(z_coord.dz_ref)[0]).sum() == 10


@pytest.mark.parametrize("scheme", ["sweeney_2band", "auto", "jerlov_2band"])
@pytest.mark.parametrize("partial_top", [False, True], ids=["deep-top", "partial-top"])
def test_tripole_sweeney_column_total_is_q_net_on_partial_cells(
        tmp_path, _fp64_policy_restored, scheme, partial_top):
    """Functional tripole twin of the MPAS test (Claude review, batch 2): on a
    synthetic tripole mesh with a partial-cell coordinate the Sweeney branch
    deposits exactly q_net per wet column and nothing below the seabed."""
    import jax.numpy as jnp
    from scripts.run import run_omip as R
    from legoesm.core.field import Field
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.ocean.vertical import create_partial_cell_coordinate, compute_layer_thickness
    from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh
    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)
    common = dict(tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon",
                  forcing_mode="jra55_do_tropical")
    grid, z, config, model, _ = R._create_setup(
        "tripole", "eorca1", 12, 600.0, "none", "II", sw_scheme=scheme, **common)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    state = R._init_rest_state("tripole", grid, z, H_max=600.0)
    wet2d = np.asarray(state.land_mask.data) > 0.5
    rng = np.random.default_rng(5)
    H_np = np.where(wet2d, rng.uniform(40.0, 600.0, wet2d.shape), 0.0)
    if partial_top:
        wi, wj = np.nonzero(wet2d)
        H_np[wi[:6], wj[:6]] = 0.5 * float(np.asarray(z.dz_ref)[0])
    H = jnp.asarray(H_np)
    pc = create_partial_cell_coordinate(z, H)
    model = LatLonCGridOceanModel(grid, pc, config)
    state = state._replace(H_bathy=Field(data=H, name="H_bathy", dims=state.H_bathy.dims))
    h_k = np.asarray(compute_layer_thickness(state.eta.data, H, pc))
    shape2d = state.T.data.shape[:2]
    chl = jnp.full(shape2d, 0.3) if scheme == "sweeney_2band" else None
    sf = OceanSurfaceForcing(sw_down=jnp.full(shape2d, 200.0), q_net=jnp.full(shape2d, 50.0),
                             tau_x=jnp.zeros(shape2d), tau_y=jnp.zeros(shape2d),
                             chl=chl)
    t_on = model.tendencies(state, surface_forcing=sf)
    t_zero = model.tendencies(state, surface_forcing=sf._replace(
        sw_down=jnp.zeros(shape2d), q_net=jnp.zeros(shape2d)))
    dT = np.asarray(t_on.dT_dt.data) - np.asarray(t_zero.dT_dt.data)
    assert np.all(dT[h_k <= 0.0] == 0.0)
    col = (rho_0_ref * c_sw * h_k * dT).sum(axis=-1)
    np.testing.assert_allclose(col[wet2d], 50.0, rtol=1e-9)
    assert (h_k[wet2d] <= 0.0).any()
    if partial_top:                            # partial TOP cells really exist
        assert (h_k[wet2d][:, 0] < 0.6 * np.asarray(z.dz_ref)[0]).sum() == 6
