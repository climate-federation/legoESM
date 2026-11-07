"""Direct tests for scripts/validate/ocean_fidelity/compare_hires_lane_surface.py."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts/validate/ocean_fidelity/compare_hires_lane_surface.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_hires_lane_surface", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _points(rng, n=4000):
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    sst = 28.0 * np.cos(np.radians(lat)) ** 2
    return lon, lat, sst


def _write_fesom(tmp_path, lon, lat, sst, day, area=None):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / f"day_1958_{day:03d}"
    g = zarr.open_group(str(store), mode="w")
    for k, v in (("lon", lon), ("lat", lat)):
        g.create_array(k, data=v.astype(np.float32))
    for k, v in (("sst", sst), ("sss", np.full_like(sst, 35.0)), ("ssh", np.zeros_like(sst))):
        g.create_array(k, data=v[None].astype(np.float32))
    g.attrs["calendar_date"] = f"1958-doy{day:03d}"
    mesh = tmp_path / "mesh"
    mesh.mkdir(exist_ok=True)
    a = np.ones(lon.size) if area is None else area
    np.save(mesh / "area.npy", np.stack([a, a * 0.5, a * 0.1], axis=1))
    return store, mesh


def _write_lane(path, lon, lat, sst, day=30.0, sst_override=None):
    n = lon.size
    wet = np.ones(n, dtype=bool)
    wet[: n // 10] = False                    # land cells carry a plausible non-zero T
    T = np.stack([sst if sst_override is None else sst_override, sst - 5.0], axis=1)
    np.savez(path, T=T, S=np.full_like(T, 35.0), land_mask=wet.astype(float), lat_T=lat, lon_T=lon,
             eta=np.zeros(n), H_bathy=np.where(wet, 4000.0, 0.0), z_center_ref=np.array([5.0, 15.0]),
             _time_s=np.float64(day * 86400.0), _step=np.int64(21600))
    return path


def test_wet_is_land_mask_gt_half_and_nonfinite_wet_is_fatal(mod, tmp_path):
    rng = np.random.default_rng(0)
    lon, lat, sst = _points(rng)
    k = lon.size // 10
    a = mod.load_lane_snapshot(_write_lane(tmp_path / "a.npz", lon, lat, sst))
    assert a["sst"].size == lon.size - k
    np.testing.assert_array_equal(a["sst"], sst[k:])
    assert a["day"] == 30.0
    bad = sst.copy(); bad[k + 3] = np.nan        # one wet NaN
    with pytest.raises(ValueError, match="non-finite wet values"):
        mod.load_lane_snapshot(_write_lane(tmp_path / "b.npz", lon, lat, sst, sst_override=bad))
    ok = sst.copy(); ok[0] = np.nan              # NaN on land is fine
    mod.load_lane_snapshot(_write_lane(tmp_path / "c.npz", lon, lat, sst, sst_override=ok))


def test_score_recovers_a_uniform_offset_on_the_common_mask(mod, tmp_path):
    rng = np.random.default_rng(1)
    lon, lat, sst = _points(rng, n=20000)
    k = lon.size // 10                       # the lane's land cells: same point set on both sides
    store, mesh = _write_fesom(tmp_path, lon[k:], lat[k:], sst[k:], 30)
    ref = mod.load_fesom_daily(store, mesh)
    assert ref["day"] == 30
    lon_e, lat_e, ref_g = mod.raster(ref, 5.0)
    lat_c = 0.5 * (lat_e[1:] + lat_e[:-1])
    lane = mod.load_lane_snapshot(_write_lane(tmp_path / "snap.npz", lon, lat, sst + 0.5))
    _, _, lane_g = mod.raster(lane, 5.0)
    # a lane that is missing half the raster must shrink the scored domain for every lane
    holey = {kk: v.copy() for kk, v in lane_g.items()}
    holey["sst"][:, ::2] = np.nan
    mask = mod.common_mask(ref_g, [lane_g, holey])
    assert mask.sum() < np.isfinite(ref_g["sst"]).sum()
    assert not mask[:, ::2].any()
    st = mod.score(ref_g, lane_g, lat_c, mask)
    assert st["sst"]["bias"] == pytest.approx(0.5, abs=1e-5)
    assert st["sst"]["rms"] == pytest.approx(0.5, abs=1e-5)
    assert st["sss"]["bias"] == pytest.approx(0.0, abs=1e-5)
    assert st["sst"]["n_cells"] == int(mask.sum())
    assert set(st["sst"]["band_bias"]) == {"antarctic_S_of_45S", "SH_midlat_45S_23S", "tropics_23S_23N",
                                           "NH_midlat_23N_45N", "arctic_N_of_45N"}
    for v in st["sst"]["band_bias"].values():
        assert v == pytest.approx(0.5, abs=1e-5)
    with pytest.raises(ValueError, match="no raster cell"):
        mod.score(ref_g, lane_g, lat_c, np.zeros_like(mask))


def test_main_checks_the_day_and_weights_the_fesom_series(mod, tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    lon, lat, sst = _points(rng, n=3000)
    k = lon.size // 10
    area = rng.uniform(1.0, 3.0, lon.size - k)
    for day in (1, 2):
        store, mesh = _write_fesom(tmp_path, lon[k:], lat[k:], sst[k:] + 0.1 * day, day, area=area)
    snap = _write_lane(tmp_path / "snap.npz", lon, lat, sst, day=2.0)
    csv = tmp_path / "mean_timeseries.csv"
    csv.write_text("time_days,step,mean_SST,mean_SSS,mean_eta,max_speed,P_bt,j_maxu,i_maxu,chi\n"
                   "0.0,0,18.0,34.7,0,0,0,-1,-1,0\n2.0,10,17.0,34.8,0,0,0,-1,-1,0\n")
    out = tmp_path / "out"
    argv = ["x", "--fesom-daily", str(store), "--mesh-dir", str(mesh), "--lane", f"mpas={snap}",
            "--series", f"mpas={csv}", "--fesom-daily-root", str(tmp_path), "--series-days", "5",
            "--res-deg", "10", "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    rep = json.loads((out / "hires_lane_surface.json").read_text())
    assert rep["lanes"]["mpas"]["stats"]["sst"]["bias"] == pytest.approx(-0.2, abs=1e-5)
    assert "daily mean" in rep["sampling"] and "instantaneous" in rep["sampling"]
    assert rep["series_end"]["fesom"]["day"] == 2.0
    expected = (sst[k:].astype(np.float32).astype(np.float64) + np.float32(0.2)) @ area / area.sum()
    assert rep["series_end"]["fesom"]["sst"] == pytest.approx(expected, abs=1e-4)
    assert rep["series_end"]["fesom"]["sst"] != pytest.approx(np.mean(sst[k:]) + 0.2, abs=1e-3)
    assert rep["series_end"]["mpas"]["sst"] == 17.0
    assert (out / "hires_lane_surface.png").exists()
    # a lane whose end day is not the FESOM store's day is refused
    day3 = _write_lane(tmp_path / "d3.npz", lon, lat, sst, day=3.0)
    monkeypatch.setattr(sys, "argv", argv[:6] + [f"mpas={day3}"] + argv[7:])
    with pytest.raises(ValueError, match="ends at day 3"):
        mod.main()
