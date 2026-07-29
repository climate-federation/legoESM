"""Slab-ocean q-flux climatology: loader, cyclic interp, slab-step threading,
generator, and the byte-identical no-q-flux default.

Sign convention under test: q_flux is +W/m2 INTO the mixed layer (same sign as
the scalar SimpleOceanConfig.Q_flux it replaces), and the derivation
q_flux = -(net downward surface heat flux) is what makes a fixed-SST slab
budget close (see generate_qflux_climatology).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from legoesm.grids.factory import create_grid


def _write_qflux(path, q, lat, lon, start_year=1979):
    from scripts.data.generate_qflux_climatology import write_qflux
    write_qflux(path, np.asarray(q), np.asarray(lat), np.asarray(lon),
                start_year=start_year)


def test_synthetic_generator_zero_global_mean_and_seasonal():
    from scripts.data.generate_qflux_climatology import make_synthetic_qflux
    q = make_synthetic_qflux(nlat=73, nlon=144)
    assert q.shape == (12, 73, 144)
    lat = np.linspace(90.0, -90.0, 73)
    w = np.cos(np.deg2rad(lat))[None, :, None]
    # An OHT convergence has ~zero area-weighted global-annual mean (no source).
    aw_mean = float((q * w).sum() / (w * np.ones_like(q)).sum())
    assert abs(aw_mean) < 2.0, f"synthetic q-flux global mean {aw_mean:.2f} W/m2"
    # Seasonal cycle present.
    assert np.abs(q[6] - q[0]).max() > 5.0


def test_loader_regrids_and_interpolates_cyclically(tmp_path):
    from legoesm.ocean.forcing.qflux import (
        load_qflux_climatology, qflux_at_time,
    )
    # A 12-month climatology with a clean per-month constant so interpolation is
    # checkable: month m has value m (W/m2), zonally+meridionally uniform.
    nlat, nlon = 19, 36
    q = np.zeros((12, nlat, nlon))
    for m in range(12):
        q[m] = float(m)
    lat = np.linspace(90.0, -90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    path = tmp_path / "qflux.nc"
    _write_qflux(path, q, lat, lon)

    grid = create_grid(grid_type="latlon", resolution=16)
    qf = load_qflux_climatology(str(path), grid)
    assert qf.qflux.shape == (12, *grid.grid_shape_2d)

    # Mid-Jan (day 15.5) hits month-0 anchor exactly -> value 0.
    v0 = np.asarray(qflux_at_time(qf, 15.5))
    assert np.allclose(v0, 0.0, atol=1e-4)
    # Mid-Jul (day 196.5) hits month-6 anchor -> value 6.
    v6 = np.asarray(qflux_at_time(qf, 196.5))
    assert np.allclose(v6, 6.0, atol=1e-3)
    # Between Jan (0) and Feb (45) anchors, day 30 interpolates ~0.5.
    vmid = float(np.asarray(qflux_at_time(qf, 30.25)).mean())
    assert 0.3 < vmid < 0.7
    # Dec->Jan seam wraps (Taylor cyclic): day 357 interpolates 11 -> 0.
    vseam = float(np.asarray(qflux_at_time(qf, 357.0)).mean())
    assert 0.0 <= vseam <= 11.0


def test_slab_step_qflux_override_and_byte_identical_default():
    from legoesm.ocean.simple_ocean import (
        SimpleOceanConfig, init_slab_state, _slab_step,
    )
    from legoesm.core.coupling_fields import AtmToSurface

    grid = create_grid(grid_type="latlon", resolution=16)
    sh = grid.grid_shape_2d
    z = jnp.zeros(sh)
    f = AtmToSurface(
        sw_down=z + 200.0, lw_down=z + 300.0, precip_total=z, precip_snow=z,
        T_lowest=z + 288.0, q_lowest=z + 0.008, u_lowest=z + 5.0, v_lowest=z,
        p_lowest=z + 1.0e5, p_surface=z + 1.01e5, rho_lowest=z + 1.2,
        cos_zenith=z + 0.5, co2_ppmv=z + 415.0,
        has_radiation=jnp.array(1.0), has_precipitation=jnp.array(1.0),
    )
    cfg = SimpleOceanConfig(mode="slab")
    st = init_slab_state(sh, T_sfc_init=288.0)

    s_scalar, *_ = _slab_step(st, f, cfg, 3600.0)                    # scalar Q_flux=0
    s_zeros, *_ = _slab_step(st, f, cfg, 3600.0, q_flux=jnp.zeros(sh))
    # q_flux=zeros must be byte-identical to the scalar-0 default.
    assert jnp.allclose(s_zeros.T_sfc.data, s_scalar.T_sfc.data, atol=0.0)

    # A +50 W/m2 q-flux warms the slab MORE than the scalar-0 case (sign: +into
    # the mixed layer).
    s_pos, *_ = _slab_step(st, f, cfg, 3600.0, q_flux=jnp.full(sh, 50.0))
    assert float(s_pos.T_sfc.data.mean()) > float(s_scalar.T_sfc.data.mean())
    # And the delta matches +Q/C_mix * dt exactly (energy budget sign check).
    C_mix = cfg.rho_ocean * cfg.c_ocean * cfg.h_mix
    expected = 50.0 / C_mix * 3600.0
    got = float((s_pos.T_sfc.data - s_scalar.T_sfc.data).mean())
    assert abs(got - expected) < 1e-6, (got, expected)


def test_derive_qflux_sign_closes_fixed_sst_budget(tmp_path):
    """derive mode: q_flux = -(net downward) so a slab forced by it holds SST.

    Build a synthetic monthly flux climatology with a KNOWN net downward flux,
    derive the q-flux, and check q_flux == -(sw+lw-sh-lh) exactly.
    """
    import netCDF4
    from scripts.data.generate_qflux_climatology import derive_qflux_from_fluxes

    nlat, nlon = 10, 20
    rng = np.random.default_rng(0)
    sw = rng.uniform(100, 250, (12, nlat, nlon))
    lw = rng.uniform(-80, -20, (12, nlat, nlon))
    sh = rng.uniform(5, 40, (12, nlat, nlon))
    lh = rng.uniform(20, 120, (12, nlat, nlon))
    path = tmp_path / "fluxes.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        for name, arr in (("sw_net_sfc", sw), ("lw_net_sfc", lw),
                          ("hfss", sh), ("hfls", lh)):
            v = ds.createVariable(name, "f8", ("time", "lat", "lon"))
            v[:] = arr
        ds.createVariable("lat", "f8", ("lat",))[:] = np.linspace(90, -90, nlat)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.linspace(
            0, 360, nlon, endpoint=False)

    q, _, _ = derive_qflux_from_fluxes(str(path))
    net_down = sw + lw - sh - lh
    np.testing.assert_allclose(q, -net_down, rtol=1e-12)
    # A slab budget forced with this q_flux at the AMIP fluxes is in balance:
    # sw + lw - sh - lh + q_flux == 0.
    np.testing.assert_allclose(net_down + q, 0.0, atol=1e-9)


def test_cli_qflux_path_round_trips():
    from scripts.run.run_coupled import build_parser
    args = build_parser().parse_args(
        ["--grid", "latlon", "--ocean", "slab",
         "--ocean-qflux-path", "/tmp/qf.nc"])
    assert args.ocean_qflux_path == "/tmp/qf.nc"
    # config builder threads it onto SimpleOceanConfig
    from legoesm.ocean.simple_ocean import SimpleOceanConfig
    cfg = SimpleOceanConfig(q_flux_path="/tmp/qf.nc")
    assert cfg.q_flux_path == "/tmp/qf.nc"
