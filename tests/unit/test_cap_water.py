"""cap_water.py: per-column log-p interpolation and the masked area mean."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias" / "cap_water.py")
_spec = importlib.util.spec_from_file_location("cap_water", _TOOL)
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)


def test_columns_to_plev_is_log_linear_and_never_extrapolates_below_the_surface():
    p_full = np.array([[1e4, 5e4, 9e4], [1e4, 5e4, 1e5]])
    f = np.log(p_full)                                   # exactly linear in log-p
    out = cw.columns_to_plev(p_full, f, np.array([3e4, 9.5e4]))
    assert out[0, 0] == pytest.approx(np.log(3e4))
    assert np.isnan(out[0, 1])                           # 950 hPa below column 0's surface (900)
    assert out[1, 1] == pytest.approx(np.log(9.5e4))


def test_run_month_uses_the_real_calendar():
    assert cw.run_month({"start_day": 0.0}, 0) == 1           # day 0 = Jan 1
    assert cw.run_month({"start_day": 0.0}, 30) == 1          # Jan 31, not "month 2"
    assert cw.run_month({"start_day": 0.0}, 31) == 2
    assert cw.run_month({"start_day": 0.0}, 45) == 2
    assert cw.run_month({"start_day": 0.0}, 59) == 3          # noleap: no Feb 29
    assert cw.run_month({"start_day": None}, 364) == 12
    assert cw.run_month({"start_day": 40.0}, 25) == 3


def test_paired_area_mean_drops_the_same_cells_on_both_sides():
    area = np.array([1.0, 1.0, 2.0]); mask = np.array([True, True, True])
    model = np.array([1.0, np.nan, 3.0]); ref = np.array([10.0, 100.0, 30.0])
    m, r, kept = cw.paired_area_mean(model, ref, area, mask)
    assert m == pytest.approx((1 + 6) / 3) and r == pytest.approx((10 + 60) / 3)   # cell 1 dropped from BOTH
    assert kept == pytest.approx(3 / 4)
    ref[2] = np.nan                                       # a missing reference value drops that cell too
    m, r, kept = cw.paired_area_mean(model, ref, area, mask)
    assert m == 1.0 and r == 10.0 and kept == pytest.approx(1 / 4)
    with pytest.raises(SystemExit):
        cw.paired_area_mean(np.full(3, np.nan), ref, area, mask)


def test_columns_to_plev_rejects_targets_above_the_top():
    p_full = np.array([[1e4, 5e4, 9e4]])
    out = cw.columns_to_plev(p_full, np.log(p_full), np.array([5e3, 2e4]))
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(np.log(2e4))


def test_area_mean_weights_and_skips_nan_and_refuses_empty():
    area = np.array([1.0, 3.0, 5.0]); mask = np.array([True, True, False])
    assert cw.area_mean(np.array([2.0, 6.0, 100.0]), area, mask) == pytest.approx((2 + 18) / 4)
    assert cw.area_mean(np.array([np.nan, 6.0, 100.0]), area, mask) == pytest.approx(6.0)
    x2 = np.array([[2.0, 1.0], [6.0, 1.0], [100.0, 1.0]])
    assert cw.area_mean(x2, area, mask).tolist() == pytest.approx([5.0, 1.0])
    with pytest.raises(SystemExit):
        cw.area_mean(np.array([np.nan, np.nan, 1.0]), area, mask)


def test_on_cells_is_periodic_in_longitude():
    import xarray as xr
    lon = np.arange(0.0, 360.0, 90.0)                     # 0, 90, 180, 270
    d = xr.DataArray(np.arange(4.0)[None, :].repeat(2, 0), dims=("lat", "lon"),
                     coords={"lat": [70.0, 80.0], "lon": lon})
    v = cw.on_cells(d, np.array([70.0, 70.0]), np.array([359.0, 271.0]))
    assert v[0] == 0.0                                      # 359 is nearest to the 0-degree column, not 270
    assert v[1] == 3.0



def test_load_state_layer_thickness_closes_the_column(tmp_path, monkeypatch):
    """dp sums to p_s - p_top, so the column-water storage term spans the whole column."""
    import json
    import numpy as np
    (tmp_path / "r").mkdir()
    vg = np.array([[200.0 / 1e5 * 1.0, 0.0, 0.0, 0.0], [0.0, 0.3, 0.7, 1.0]])   # p_top = 200 Pa * p_ref/1e5
    np.savez(tmp_path / "r" / "checkpoint_day_0015.npz", T=np.full((4, 3), 250.0),
             p_s=np.array([1e5, 9e4, 1e5, 9.5e4]), trc_q_v=np.full((4, 3), 1e-3),
             meta_vgrid=vg, day=np.array(15.0))
    json.dump({}, open(tmp_path / "r" / "experiment_config.json", "w"))
    lat = np.array([80.0, 76.0, 70.0, 60.0]); lon = np.zeros(4); area = np.ones(4)
    monkeypatch.setattr(cw.rb, "ROOT", str(tmp_path))
    monkeypatch.setattr(cw.cl, "mesh_coords", lambda exp: (lat, lon, area))
    monkeypatch.setattr(cw.cl, "cell_order", lambda z, n: np.arange(n))
    st = cw.load_state("r", 15)[0]
    from legoesm import constants
    p_top = vg[0][0] * constants.p_ref
    assert np.allclose(st["dp"].sum(axis=1), st["p_s"] - p_top)
    assert np.all(st["dp"] > 0)


def test_cap_moisture_transport_boundary_sum_matches_the_divergence_operator():
    """One unit of edge flux across a single boundary edge of the northern cap: gross in/out
    follow the edge orientation, the net equals -sum(area*div) to round-off."""
    import numpy as np
    from legoesm import constants
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", resolution=2)
    lat = np.asarray(mesh.latCell); cap = lat >= 0.0
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    boundary = np.where(cap[c0] != cap[c1])[0]
    e_in = next(e for e in boundary if cap[c1[e]])        # flow c0 -> c1 enters the cap
    e_out = next(e for e in boundary if cap[c0[e]])       # flow c0 -> c1 leaves the cap
    n_cells, n_edges, nlev = lat.size, c0.size, 3
    q = np.full((n_cells, nlev), 2.0e-3); dp = np.full((n_cells, nlev), 1000.0)
    u = np.zeros((n_edges, nlev)); u[e_in, :] = 1.0; u[e_out, :] = 0.5
    gin, gout, net, net_div = cw.cap_moisture_transport(q, u, dp, mesh, cap)
    unit = 2.0e-3 * 1000.0 / constants.g * nlev
    assert gin == pytest.approx(unit * float(mesh.dvEdge[e_in]), rel=1e-6)
    assert gout == pytest.approx(-0.5 * unit * float(mesh.dvEdge[e_out]), rel=1e-6)
    assert net == pytest.approx(gin + gout, rel=1e-6) and net_div == pytest.approx(net, rel=1e-6)
    # an interior edge moves nothing across the boundary
    e_int = next(e for e in range(n_edges) if cap[c0[e]] and cap[c1[e]])
    u2 = np.zeros_like(u); u2[e_int, :] = 3.0
    assert cw.cap_moisture_transport(q, u2, dp, mesh, cap)[2] == pytest.approx(0.0, abs=1e-9)


def test_cap_moisture_transport_splits_in_and_out_per_level():
    """A boundary column with inflow below and outflow aloft counts in BOTH gross terms."""
    import numpy as np
    from legoesm import constants
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", resolution=2)
    lat = np.asarray(mesh.latCell); cap = lat >= 0.0
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    e = next(e for e in np.where(cap[c0] != cap[c1])[0] if cap[c1[e]])
    q = np.full((lat.size, 2), 1.0e-3); dp = np.full((lat.size, 2), 1000.0)
    u = np.zeros((c0.size, 2)); u[e] = [-1.0, 2.0]                 # aloft out, below in
    gin, gout, net, _ = cw.cap_moisture_transport(q, u, dp, mesh, cap)
    unit = 1.0e-3 * 1000.0 / constants.g * float(mesh.dvEdge[e])
    assert gin == pytest.approx(2.0 * unit, rel=1e-6) and gout == pytest.approx(-unit, rel=1e-6)
    assert net == pytest.approx(unit, rel=1e-6)
    with pytest.raises(SystemExit, match="non-finite"):
        cw.cap_moisture_transport(np.where(q > 0, np.nan, q), u, dp, mesh, cap)


def test_inflow_humidity_weights_only_the_entering_air():
    """Reference humidity twice the model's on the inflow edge: ratio 0.5 and the extra import
    equals the inflow's own transport; an outflow-only wind gives no weight; NaN reference
    levels are excluded from the weights and reported in the cover fraction."""
    import numpy as np
    from legoesm import constants
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", resolution=2)
    lat = np.asarray(mesh.latCell); cap = lat >= 0.0
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    e = next(e for e in np.where(cap[c0] != cap[c1])[0] if cap[c1[e]])
    q = np.full((lat.size, 2), 1.0e-3); dp = np.full((lat.size, 2), 1000.0)
    u = np.zeros((c0.size, 2)); u[e] = [1.0, 1.0]
    qm, qr, extra, netchg, cover, out_cover = cw.inflow_humidity(q, 2.0 * q, u, dp, mesh, cap)
    assert qm == pytest.approx(1.0e-3) and qr == pytest.approx(2.0e-3) and cover == pytest.approx(1.0)
    assert extra == pytest.approx(cw.cap_moisture_transport(q, u, dp, mesh, cap)[0], rel=1e-6)
    assert netchg == pytest.approx(extra, rel=1e-6)                  # inflow only: net change = extra
    ref = 2.0 * q; ref[:, 0] = np.nan                                # top level undefined
    qm, qr, extra, netchg, cover, out_cover = cw.inflow_humidity(q, ref, u, dp, mesh, cap)
    assert cover == pytest.approx(0.5) and qr == pytest.approx(2.0e-3)
    # add an outflow edge with a DRIER reference: less export, so the signed net change EXCEEDS the inflow extra
    e_out = next(k for k in np.where(cap[c0] != cap[c1])[0] if cap[c0[k]])
    u2 = u.copy(); u2[e_out] = [1.0, 1.0]
    ref2 = 2.0 * q; ref2[c0[e_out]] = 0.0; ref2[c1[e_out]] = 0.0
    _, _, extra2, netchg2, _, out_cover2 = cw.inflow_humidity(q, ref2, u2, dp, mesh, cap)
    assert netchg2 > extra2 and out_cover2 == pytest.approx(1.0)
    ref3 = ref2.copy(); ref3[c0[e_out]] = np.nan
    assert cw.inflow_humidity(q, ref3, u2, dp, mesh, cap)[5] == pytest.approx(0.0)   # outflow reference undefined
    with pytest.raises(SystemExit, match="no inflow"):
        cw.inflow_humidity(q, np.full_like(q, np.nan), u, dp, mesh, cap)


def test_era5_q_on_levels_masks_below_surface_brackets():
    """A model level bracketed by a below-surface ERA5 level is NaN, not extrapolated."""
    import numpy as np
    plev = np.array([50000.0, 70000.0, 85000.0, 100000.0])
    cols = np.array([[1.0, 2.0, 3.0, 4.0]]) * 1e-3
    ps = np.array([90000.0])                                          # 1000 hPa level is below ground
    p_full = np.array([[60000.0, 80000.0, 88000.0, 95000.0]])
    out = cw.era5_q_on_levels(plev, cols, ps, p_full)
    assert np.isfinite(out[0, :2]).all() and np.isnan(out[0, 2]) and np.isnan(out[0, 3])
    assert out[0, 0] == pytest.approx(np.interp(np.log(60000.0), np.log(plev[:3]), cols[0, :3]))


def test_cap_transport_bins_recover_the_totals():
    import numpy as np
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", resolution=2)
    lat = np.asarray(mesh.latCell); cap = lat >= 0.0
    rng = np.random.default_rng(0)
    n_cells, n_edges, nlev = lat.size, np.asarray(mesh.dvEdge).size, 4
    q = rng.uniform(1e-4, 5e-3, (n_cells, nlev)); dp = rng.uniform(500.0, 2000.0, (n_cells, nlev))
    p_full = np.cumsum(dp, axis=1) + 50000.0 * rng.uniform(0.5, 1.5, (n_cells, 1))
    u = rng.normal(0.0, 5.0, (n_edges, nlev))
    q_ref = 1.5 * q; q_ref[:, 0] = np.nan
    b = cw.cap_transport_bins(q, u, dp, p_full, mesh, cap, q_ref)
    gin, gout, net, _ = cw.cap_moisture_transport(q, u, dp, mesh, cap)
    tot = b.sum(axis=(0, 1))
    assert tot[:3] == pytest.approx([gin, gout, net], rel=1e-6)   # one path sums in float32
    assert tot[3] == pytest.approx(cw.inflow_humidity(q, q_ref, u, dp, mesh, cap)[3], rel=1e-6)
    assert b.shape == (3, 6, 5) and (b[:, :, 0] >= 0).all() and (b[:, :, 1] <= 0).all()
    assert np.abs(b[:, :, 4]).sum() > 0 and not np.allclose(b[:, :, 4], b[:, :, 3])   # inflow-only column is its own number


def test_era5_boundary_transport_matches_the_analytic_uniform_flow():
    """Uniform 1 m/s northward flow of unit mixing ratio through a 1000 hPa column
    crosses the circle at the analytic rate; the surface cut-off, the in/out split
    per element and the layer/sector binning are checked against hand values."""
    import numpy as np
    from legoesm import constants
    plev = np.array([100.0, 500.0, 700.0, 850.0, 925.0, 1000.0]) * 100.0
    nlon = 360
    lon = np.arange(nlon) * (360.0 / nlon)
    v = np.ones((plev.size, nlon)); q = np.ones_like(v); ps = np.full(nlon, 1000e2)
    lat_b = 72.5
    b = cw.era5_boundary_transport(v, q, ps, lon, plev, lat_b)
    # the column reaches from 0 (top slab floored) to ps = 1000 hPa (the lowest slab extends to the surface)
    expect = 1000e2 / constants.g * 2.0 * np.pi * constants.R_earth * np.cos(np.radians(lat_b))
    assert b.sum(axis=(0, 1))[2] == pytest.approx(expect, rel=1e-12)
    assert (b[:, :, 1] == 0).all() and b.sum(axis=(0, 1))[0] == pytest.approx(expect, rel=1e-12)
    # layers by level pressure: 100/500 above 600, 700 in 600-800, 850/925/1000 below 800.
    # Literal slab thicknesses [hPa] for plev 100/500/700/850/925/1000 with ps = 1000: top slab 0..300 (mid-point 300,
    # floored at 0), then 300..600, 600..775, 775..887.5, 887.5..962.5, 962.5..1000 (lowest slab cut at ps).
    SLAB_HPA = (300.0, 300.0, 175.0, 112.5, 75.0, 37.5)
    assert sum(SLAB_HPA) == 1000.0
    slab = lambda k: SLAB_HPA[k] * 100.0
    per = 2.0 * np.pi * constants.R_earth * np.cos(np.radians(lat_b)) / constants.g
    assert b[0].sum(axis=0)[2] == pytest.approx(per * (slab(0) + slab(1)), rel=1e-12)
    assert b[1].sum(axis=0)[2] == pytest.approx(per * slab(2), rel=1e-12)
    assert b[2].sum(axis=0)[2] == pytest.approx(per * (slab(3) + slab(4) + slab(5)), rel=1e-12)
    # six equal sectors
    assert b.sum(axis=0)[:, 2] == pytest.approx(np.full(6, expect / 6.0), rel=1e-12)
    # a southward half: split per element, not after the sum
    v2 = v.copy(); v2[:, : nlon // 2] = -1.0
    b2 = cw.era5_boundary_transport(v2, q, ps, lon, plev, lat_b)
    assert b2.sum(axis=(0, 1))[2] == pytest.approx(0.0, abs=1e-6 * expect)
    assert b2.sum(axis=(0, 1))[0] == pytest.approx(expect / 2.0, rel=1e-12)
    assert b2.sum(axis=(0, 1))[1] == pytest.approx(-expect / 2.0, rel=1e-12)
    # a surface at 800 hPa removes everything below it
    b3 = cw.era5_boundary_transport(v, q, np.full(nlon, 800e2), lon, plev, lat_b)
    assert b3.sum(axis=(0, 1))[2] == pytest.approx(per * 800e2, rel=1e-12)
    # a surface ABOVE the lowest level (1040 hPa > 1000): the column still reaches ps, not 1012.5
    b4 = cw.era5_boundary_transport(v, q, np.full(nlon, 1040e2), lon, plev, lat_b)
    assert b4.sum(axis=(0, 1))[2] == pytest.approx(per * 1040e2, rel=1e-12)
    # opposing winds at different levels of ONE longitude both count (in aloft, out below)
    v3 = v.copy(); v3[3:, :] = -1.0
    b5 = cw.era5_boundary_transport(v3, q, ps, lon, plev, lat_b)
    assert b5.sum(axis=(0, 1))[0] == pytest.approx(per * (slab(0) + slab(1) + slab(2)), rel=1e-12)
    assert b5.sum(axis=(0, 1))[1] == pytest.approx(-per * (slab(3) + slab(4) + slab(5)), rel=1e-12)
    with pytest.raises(SystemExit):
        cw.era5_boundary_transport(v, q, ps, lon, plev[::-1], lat_b)
    with pytest.raises(SystemExit):
        cw.era5_boundary_transport(v, q, ps, lon, plev / 100.0, lat_b)      # hPa, not Pa
    with pytest.raises(SystemExit):
        cw.era5_boundary_transport(np.where(v > 2, v, np.nan), q, ps, lon, plev, lat_b)


def test_era5_transport_end_to_end_on_a_synthetic_two_day_file(tmp_path, capsys):
    """cdo-style files (float 'YYYYMMDD.f' time on the pressure-level files, decoded
    time on the surface file): all three samples of an inclusive --dates window are
    used, the window mean is the PLAIN mean (a trapezoid would weight the middle
    sample twice), --hour keeps one sample, and a uniform northward flow reproduces
    the analytic transport in the printed ALL column."""
    import numpy as np, xarray as xr
    from legoesm import constants
    plev = np.array([100.0, 500.0, 700.0, 850.0, 925.0, 1000.0]) * 100.0
    lat = np.array([74.0, 73.0, 72.0, 71.0]); lon = np.arange(0.0, 360.0, 10.0)
    tf = np.array([19790107.47916667, 19790108.47916667, 19790109.47916667])
    td = np.array(["1979-01-07T11:30", "1979-01-08T11:30", "1979-01-09T11:30"], dtype="datetime64[ns]")
    v = np.zeros((3, plev.size, lat.size, lon.size)); v[0] = 1.0; v[1] = 7.0; v[2] = 1.0   # plain mean 3, trapezoid 4
    xr.Dataset({"var132": (("time", "plev", "lat", "lon"), v)}, coords={"time": tf, "plev": plev, "lat": lat, "lon": lon}).to_netcdf(tmp_path / "pl_132.nc")
    xr.Dataset({"var133": (("time", "plev", "lat", "lon"), np.full_like(v, 1e-3))}, coords={"time": tf, "plev": plev, "lat": lat, "lon": lon}).to_netcdf(tmp_path / "pl_133.nc")
    xr.Dataset({"var134": (("time", "lat", "lon"), np.full((3, lat.size, lon.size), 1000e2))}, coords={"time": td, "lat": lat, "lon": lon}).to_netcdf(tmp_path / "sf_134.nc")
    import json
    args = type("A", (), dict(era5_dir=str(tmp_path), dates=("1979-01-07", "1979-01-09"), lat=72.5, stems="v=pl_132,q=pl_133,ps=sf_134",
                              json=str(tmp_path / "bins.json"), hour=None))
    assert cw.era5_transport(args) == 0
    out = capsys.readouterr().out
    assert "3 samples" in out
    A = 2.0 * np.pi * constants.R_earth ** 2 * (1.0 - np.sin(np.radians(72.5)))
    per_v = 1e-3 * 1000e2 / constants.g * 2.0 * np.pi * constants.R_earth * np.cos(np.radians(72.5)) * 86400.0 / A
    line = [l for l in out.splitlines() if "net import" in l][0]
    assert float(line.split("samples")[1].split()[1]) == pytest.approx(3.0 * per_v, abs=6e-4)   # plain mean; trapezoid would give 4
    j = json.load(open(tmp_path / "bins.json"))
    assert j["n"] == 3 and len(j["samples_moisture_mm_day"]) == 3 and len(j["times"]) == 3
    assert np.asarray(j["samples_moisture_mm_day"])[1][:, :, 2].sum() == pytest.approx(7.0 * per_v, rel=1e-6)   # net column, middle sample
    args.hour = 11
    assert cw.era5_transport(args) == 0 and "3 samples" in capsys.readouterr().out
    args.hour = 5
    with pytest.raises(SystemExit):
        cw.era5_transport(args)
    with pytest.raises(SystemExit):
        cw.era5_transport(type("B", (), dict(vars(args), dates=("1979-01-07",))))
    # a proper hourly subset: 6-hourly files, --hour 12 keeps 3 of 12 samples (v = 7 at 12 UTC, 1 elsewhere) and the
    # provenance step is the FILTERED cadence (86400 s), not the file's
    th = np.array([f"1979-01-0{d}T{h:02d}:00" for d in (7, 8, 9) for h in (0, 6, 12, 18)], dtype="datetime64[ns]")
    vh = np.ones((12, plev.size, lat.size, lon.size)); vh[2::4] = 7.0
    for name, var, arr in (("pl_132", "var132", vh), ("pl_133", "var133", np.full_like(vh, 1e-3))):
        xr.Dataset({var: (("time", "plev", "lat", "lon"), arr)}, coords={"time": th, "plev": plev, "lat": lat, "lon": lon}).to_netcdf(tmp_path / f"{name}.nc")
    xr.Dataset({"var134": (("time", "lat", "lon"), np.full((12, lat.size, lon.size), 1000e2))}, coords={"time": th, "lat": lat, "lon": lon}).to_netcdf(tmp_path / "sf_134.nc")
    args.hour = 12; args.dates = ("1979-01-07", "1979-01-09")
    assert cw.era5_transport(args) == 0
    out = capsys.readouterr().out
    assert "3 samples" in out and "step 86400 seconds" in out
    line = [l for l in out.splitlines() if "net import" in l][0]
    assert float(line.split("samples")[1].split()[1]) == pytest.approx(7.0 * per_v, abs=6e-4)


def test_bin_sums_hand_values_and_sector_index_wrap():
    import numpy as np
    assert list(cw.sector_index([0.0, 29.9, 30.0, 359.9, -30.0, 210.0, 269.9, 270.0])) == [0, 0, 1, 0, 0, 4, 4, 5]
    F = np.array([[1.0, -2.0], [3.0, 4.0]]); chg = np.array([[0.5, 0.5], [0.5, 0.5]])
    layer = np.array([[0, 0], [2, 2]]); sector = np.array([[1, 1], [1, 3]])
    out = cw.bin_sums(F, chg, layer, sector, np.array([[True, True], [True, False]]))
    assert out[0, 1].tolist() == [1.0, -2.0, -1.0, 1.0, 0.5]   # inflow-only chg counts the +1 element alone
    assert out[2, 1].tolist() == [3.0, 0.0, 3.0, 0.5, 0.5]
    assert out[2, 3].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]      # inactive element left out
    # entering AIR with zero moisture (F = 0, w > 0): the inflow-only substitution must still count it
    dry = cw.bin_sums(np.array([[0.0]]), np.array([[2.0]]), np.array([[0]]), np.array([[0]]), np.array([[True]]),
                      air_in=np.array([[True]]))
    assert dry[0, 0].tolist() == [0.0, 0.0, 0.0, 2.0, 2.0]
    assert cw.bin_sums(np.array([[0.0]]), np.array([[2.0]]), np.array([[0]]), np.array([[0]]), np.array([[True]]))[0, 0, 4] == 0.0


def test_inflow_air_state_weights_only_entering_air_and_returns_rh():
    import numpy as np
    w = np.array([[2.0, -1.0], [1.0, 0.0]]); q = np.array([[1e-3, 9.0], [3e-3, 9.0]])
    T = np.array([[250.0, 999.0], [260.0, 999.0]]); q_sat = np.array([[2e-3, 1.0], [6e-3, 1.0]])
    layer = np.zeros((2, 2), int); sector = np.zeros((2, 2), int)
    S = cw.inflow_air_state(w, q, T, q_sat, layer, sector, np.ones((2, 2), bool))
    M, qm, Tm, rh = S[0, 0]
    assert M == 3.0 and qm == pytest.approx((2 * 1e-3 + 1 * 3e-3) / 3) and Tm == pytest.approx((2 * 250 + 260) / 3)
    assert rh == pytest.approx((2 * 1e-3 + 3e-3) / (2 * 2e-3 + 6e-3))      # both entering elements at RH 0.5
    assert np.isnan(S[1, 0, 0]) and np.isnan(S[0, 1, 0])                    # bins with no entering air
    # the ERA5 path: uniform northward flow, q = q_sat/2 everywhere -> RH 0.5, T as given
    plev = np.array([500.0, 700.0, 850.0, 1000.0]) * 100.0; lon = np.arange(0.0, 360.0, 30.0)
    from legoesm.thermo import saturation_mixing_ratio
    Tf = np.full((plev.size, lon.size), 255.0)
    rs = np.asarray(saturation_mixing_ratio(Tf, plev[:, None] * np.ones_like(Tf))); qs = rs / (1 + rs)
    A = cw.era5_boundary_transport(np.ones_like(Tf), 0.5 * qs, np.full(lon.size, 1000e2), lon, plev, 72.5, T=Tf)
    assert np.nanmax(np.abs(A[..., 3] - 0.5)) < 1e-12 and np.nanmax(np.abs(A[..., 2] - 255.0)) < 1e-12
