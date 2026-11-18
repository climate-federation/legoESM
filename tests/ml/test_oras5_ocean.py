"""Direct tests of ``legoesm.training.oras5_ocean`` and ``scripts/run/train_ocean_wb_physics.py``
helpers (no data, no model): layouts, masks, face averaging, loss zero/scale, level guard."""
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training import oras5_ocean as O

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.run import train_ocean_wb_physics as T  # noqa: E402


def _day():
    a = np.zeros((4 * O.NLEV + 4, O.NLAT, O.NLON), np.float32)
    a[:O.NLEV] = 10.0                                     # T
    a[O.NLEV:2 * O.NLEV] = 35.0                           # S
    a[2 * O.NLEV:3 * O.NLEV] = np.arange(O.NLON)[None, None, :]   # u = column index
    a[3 * O.NLEV:4 * O.NLEV] = np.arange(O.NLAT)[None, :, None]   # v = row index
    a[4 * O.NLEV] = 0.3
    a[:, 100, 50] = np.nan                                # an ORAS5 gap in a wet cell
    return a


def test_day_to_fields_layout_faces_and_gap():
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(_day(), wet3, um, vm)
    assert f["T"].shape == (180, 360, 75) and f["u"].shape == (180, 361, 75)
    assert f["v"].shape == (181, 360, 75) and f["eta"].shape == (180, 360)
    assert np.isclose(f["T"][100, 50, 3], 10.0) and not v["T"][100, 50, 3]   # filled, masked
    assert np.isclose(f["u"][10, 5, 0], 4.5) and np.isclose(f["u"][10, 0, 0], 179.5)
    assert np.isclose(f["u"][10, 360, 0], 179.5) and np.isclose(f["v"][10, 7, 0], 9.5)
    assert not v["v"][0].any() and not v["v"][-1].any()      # polar faces carry no target


def test_loss_zero_at_truth_and_std_scaled():
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(_day(), wet3, um, vm)
    w = O.weights(O.base_weights(np.arange(-89.5, 90.0), np.ones(O.NLEV)), v)
    std = {k: 2.0 for k in O.FIELDS}
    t = {k: jnp.asarray(f[k]) for k in O.FIELDS}
    assert float(O.l1_loss(t, t, w, std)) == 0.0
    p = dict(t, T=t["T"] + 1.0)                     # +1 K everywhere -> 1/std/5
    assert np.isclose(float(O.l1_loss(p, t, w, std)), 0.5 / 5)
    p = dict(t, eta=t["eta"] + 1.0)                 # uniform eta offset is removed
    assert np.isclose(float(O.l1_loss(p, t, w, std)), 0.0)


def test_check_levels(tmp_path):
    d = np.linspace(0.5, 5900.0, O.NLEV)
    (tmp_path / "meta.json").write_text(json.dumps({"depths_m": d.tolist()}))
    O.check_levels(tmp_path, -(d + 0.6))
    with pytest.raises(ValueError, match="differ"):
        O.check_levels(tmp_path, d + 1.5)


def test_sample_dates_in_range_and_reproducible():
    a = T.sample_dates(1993, 2017, 2, 4, 0)
    assert a == T.sample_dates(1993, 2017, 2, 4, 0) and len(set(a)) == 4
    assert all(np.datetime64("1993-01-01") <= d <= np.datetime64("2017-12-29") for d in a)


def test_forcing_must_be_named():
    with pytest.raises(SystemExit):
        T.parse_args(["--store", "s", "--out", "o"])
    assert T.parse_args(["--store", "s", "--out", "o", "--forcing", "none"]).days == 2


def test_loss_weights_area_and_thickness_unequally():
    """A localised error is scaled by its cell's area x thickness share, not counted uniformly."""
    lat = np.arange(-89.5, 90.0)
    h = np.ones((O.NLAT, O.NLON, O.NLEV))
    h[..., 1] = 3.0                                          # level 1 three times thicker
    base = O.base_weights(lat, h)
    v = {k: np.ones(base[k].shape, bool) for k in O.FIELDS}
    w = O.weights(base, v)
    t = {k: jnp.zeros(base[k].shape) for k in O.FIELDS}
    std = {k: 1.0 for k in O.FIELDS}
    p = dict(t, T=t["T"].at[90, 0, 1].set(1.0))              # one equatorial cell, thick level
    wt = np.cos(np.deg2rad(lat))[:, None, None] * h
    expect = np.cos(np.deg2rad(0.5)) * 3.0 / wt.sum() / 5.0
    np.testing.assert_allclose(float(O.l1_loss(p, t, w, std)), expect, rtol=1e-12)


def test_face_thickness_is_the_thinner_cell():
    lat = np.arange(-89.5, 90.0)
    h = np.full((O.NLAT, O.NLON, 2), 10.0)
    h[50, 7, 1] = 4.0                                        # partial bottom cell
    h[60, 0, 1] = 0.0                                        # below the seabed at the seam column
    b = O.base_weights(lat, h)
    cu = np.cos(np.deg2rad(lat[50]))
    np.testing.assert_allclose(b["u"][50, 7, 1], cu * 4.0)   # west face of the partial cell
    np.testing.assert_allclose(b["u"][50, 8, 1], cu * 4.0)   # east face
    np.testing.assert_allclose(b["u"][50, 9, 1], cu * 10.0)
    assert b["u"][60, 0, 1] == 0.0 and b["u"][60, 1, 1] == 0.0   # seam face and its neighbour
    assert b["v"][50, 7, 1] < b["v"][49, 7, 1] * 0.5 and b["v"][51, 7, 1] < b["v"][52, 7, 1] * 0.5
    assert b["u"][:, -1].sum() == 0.0                        # seam face counted once


def test_field_std_pooled_and_eta_mean_removed():
    w = {k: np.ones(3) for k in O.FIELDS}
    f1 = {k: np.array([0.0, 0.0, 0.0]) for k in O.FIELDS}
    f2 = {k: np.array([2.0, 2.0, 2.0]) for k in O.FIELDS}
    s = O.field_std([f1, f2], [w, w])
    assert np.isclose(s["T"], 1.0)                           # pooled over both days
    assert s["eta"] == 0.0                                   # each day's eta mean removed first



def test_gate_refuses_one_nan_sample():
    assert T.gate_failure(1.0, {"A_h": 0.1, "K_v": -2.0}) is None
    assert "K_v" in T.gate_failure(1.0, {"A_h": 0.1, "K_v": float("nan")})
    assert "nan" in T.gate_failure(float("nan"), {"A_h": 0.1})


def test_d20_mae_ignores_dry_nan_columns():
    m = np.array([[[[10.0, np.nan]]]]); o = np.array([[[[12.0, np.nan]]]])
    keep = np.array([[[[True, False]]]]); cosw = np.ones((1, 2))
    np.testing.assert_allclose(T.d20_mae(m, o, keep, cosw), [2.0])


def test_level_without_oras5_data_copies_level_above_and_is_masked():
    a = _day()
    a[O.NLEV - 1] = np.nan                                   # deepest T level: no data anywhere
    a[O.NLEV - 2] = 4.0
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = np.ones((O.NLAT, O.NLON + 1)), np.ones((O.NLAT + 1, O.NLON))
    f, v = O.day_to_fields(a, wet3, um, vm)
    assert np.isfinite(f["T"]).all() and np.allclose(f["T"][..., -1], f["T"][..., -2])
    assert not v["T"][..., -1].any()


def test_place_ts_matches_driver_placement(tmp_path):
    """The trainer's IC T, S equal what the OMIP driver builds from the same day written as a
    climatology file, and an ORAS5 gap takes same-depth water, not the 4-neighbour mean."""
    xr = pytest.importorskip("xarray")
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import init_ocean_from_woa
    from legoesm.ocean.vertical import create_ocean_z_star

    depths = np.linspace(0.5, 990.0, O.NLEV)
    a = _day()
    a[:O.NLEV] += (np.arange(O.NLEV) * 0.1)[:, None, None]          # T varies with depth
    a[O.NLEV:2 * O.NLEV, :, 200:] += 1.0                             # S front in longitude
    a[:O.NLEV, 100, 49] += 1.0                                       # the gap's four neighbours differ
    a[:O.NLEV, 100, 51] += 3.0
    a[:O.NLEV, 101, 50] += 7.0
    meta = {"lat_centres": np.arange(-89.5, 90.0, 1.0).tolist(),
            "lon_centres": np.arange(0.0, 360.0, 1.0).tolist(), "depths_m": depths.tolist()}
    grid = create_latlon_grid(O.NLAT, O.NLON)
    z = create_ocean_z_star(n_levels=8, H_max=1000.0)
    land = np.ones((O.NLAT, O.NLON))
    cd = np.broadcast_to(np.abs(np.asarray(z.z_full_ref)), (O.NLAT, O.NLON, 8)).copy()
    cd[:, ::3, -1] -= 40.0                                           # cut bottom cells
    T, S = O.place_ts(a, meta, grid, z, cd, land)

    ds = xr.Dataset({"thetao": (("time", "lev", "lat", "lon"), np.asarray(a[:O.NLEV], np.float64)[None]),
                     "so": (("time", "lev", "lat", "lon"), np.asarray(a[O.NLEV:2 * O.NLEV], np.float64)[None])},
                    coords={"time": [0.0], "lev": ("lev", depths, {"units": "meters", "positive": "down"}),
                            "lat": meta["lat_centres"], "lon": meta["lon_centres"]})
    path = tmp_path / "day.nc"
    ds.to_netcdf(path)
    Td, Sd = init_ocean_from_woa(grid, z, path, path, T_var="thetao", S_var="so", cell_center_depths=cd)
    np.testing.assert_array_equal(T, np.asarray(Td))
    np.testing.assert_array_equal(S, np.asarray(Sd))
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(S))
    donors = [T[100, 49], T[100, 51], T[99, 50], T[101, 50]]
    assert any(np.allclose(T[100, 50], d, atol=1e-4) for d in donors)   # one same-depth donor (float32 grid lat) ...
    assert not np.allclose(T[100, 50], np.mean(donors, axis=0))      # ... not the 4-neighbour mean


def test_depth_cap_is_oras5_contiguous_bottom_and_refuses_changing_mask():
    nlev = O.NLEV
    day = np.zeros((4 * nlev + 4, 2, 3), np.float32)
    day[:nlev, 0, 0] = np.nan                    # land column
    day[3:nlev, 0, 1] = np.nan                   # three wet levels
    day[2, 1, 2] = np.nan                        # gap at level 2, valid below: ignored
    dz = np.arange(1.0, nlev + 1.0)
    cap, n_gap = O.depth_cap([day, day.copy()], dz)
    assert cap[0, 0] == 0.0 and cap[0, 1] == 1.0 + 2.0 + 3.0 and cap[1, 2] == 1.0 + 2.0
    assert cap[1, 0] == dz.sum() and n_gap == 1
    other = day.copy(); other[5, 1, 1] = np.nan
    with pytest.raises(ValueError, match="differs"):
        O.depth_cap([day, other], dz)


def test_apply_depth_cap_shallows_and_lands():
    from scripts.run.run_omip import apply_depth_cap
    H = np.array([[4000.0, 3000.0, 50.0]]); land = np.array([[1.0, 1.0, 0.0]])
    cap = np.array([[500.0, 5.0, 4000.0]])
    Hn, ocean, n_land, n_sh = apply_depth_cap(H, land, cap, 10.0)
    assert Hn[0, 0] == 500.0 and ocean.tolist() == [[True, False, False]]
    assert (n_land, n_sh) == (1, 1)
    with pytest.raises(ValueError, match="shape"):
        apply_depth_cap(H, land, cap[:, :2], 10.0)


def test_face_walls_from_native_mask():
    lat = np.arange(-80.0, 90.01, 0.25)
    lon = np.arange(0.0, 360.0, 0.25)
    wet = np.ones((4, lat.size, lon.size), bool)
    on_edge = np.isclose(lon, 120.5)                  # meridian between cells 120 and 121
    rows = (lat > 5.0) & (lat < 6.0)                  # model row j=95 (5.5N)
    wet[2:, np.ix_(rows, on_edge)[0], np.ix_(rows, on_edge)[1]] = False
    wet[:, lat < -78.0] = False                       # land rows
    wet[3, np.isclose(lat, 10.0), 50] = False         # one v-edge point closed at level 3
    fu, fv = O.face_walls(wet, lat, lon)
    assert fu.shape == (O.NLAT, O.NLON + 1) and fv.shape == (O.NLAT + 1, O.NLON)
    assert fu[95, 121] == 2 and fu[94, 121] == 4 and fu[95, 120] == 4
    assert np.array_equal(fu[:, 0], fu[:, -1]) and fu[5, 10] == 0
    assert fv[100, 50] == 4                            # other native points on that edge stay wet
    assert fv[0].max() == 0 and fv[-1].max() == 0


def test_sample_face_masks_close_levels_below_walls():
    """ORAS5 velocity below a wall is neither initial condition nor loss target."""
    from types import SimpleNamespace as NS
    fu = np.full((O.NLAT, O.NLON + 1), O.NLEV)
    fv = np.full((O.NLAT + 1, O.NLON), O.NLEV)
    fu[10, 5], fv[20, 7] = 3, 0
    st = NS(u_mask=NS(data=jnp.ones((O.NLAT, O.NLON + 1))),
            v_mask=NS(data=jnp.ones((O.NLAT + 1, O.NLON))))
    wet3 = np.ones((O.NLAT, O.NLON, O.NLEV))
    um, vm = T.sample_face_masks(
        st, NS(face_nlev_u=jnp.asarray(fu), face_nlev_v=jnp.asarray(fv)), O.NLEV)
    f, v = O.day_to_fields(_day(), wet3, um, vm)
    w = O.weights(O.base_weights(np.arange(-89.5, 90.0), np.ones(O.NLEV)), v)
    assert np.isclose(f["u"][10, 5, 2], 4.5) and v["u"][10, 5, 2] and float(w["u"][10, 5, 2]) > 0
    assert (f["u"][10, 5, 3:] == 0).all() and not v["u"][10, 5, 3:].any()
    assert (np.asarray(w["u"])[10, 5, 3:] == 0).all()
    assert (f["v"][20, 7] == 0).all() and not v["v"][20, 7].any()
    assert (np.asarray(w["v"])[20, 7] == 0).all() and float(w["v"][20, 8, 0]) > 0
    # without walls: exactly the 2-D masks' fields
    um0, vm0 = T.sample_face_masks(st, NS(), O.NLEV)
    f0, v0 = O.day_to_fields(_day(), wet3, um0, vm0)
    f2, v2 = O.day_to_fields(_day(), wet3, np.ones(fu.shape), np.ones(fv.shape))
    assert all(np.array_equal(f0[k], f2[k]) and np.array_equal(v0[k], v2[k]) for k in O.FIELDS)
