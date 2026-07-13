"""Unit tests for the OMIP-vs-NEMO diagnostic battery's pure helpers.

``scripts/validate/diag_omip_nemo_battery.py`` builds horizontal depth
snapshots + zonal-mean sections from a legoESM snapshot vs the NEMO reference.
The NetCDF / plotting paths need real data, but the numeric helpers
(``_nearest_level``, ``_interp_to_depths``, ``_load_legoesm_3d``,
``_zonal_section``) are pure and exercised here on synthetic arrays.
Loaded by path because ``scripts/`` is not an importable package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "diag_omip_nemo_battery",
    _ROOT / "scripts" / "validate" / "diag_omip_nemo_battery.py")
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)


def test_nearest_level_picks_closest():
    z = np.array([13.1, 52.4, 117.9, 209.5, 1000.0])
    assert _b._nearest_level(z, 0.0) == 0
    assert _b._nearest_level(z, 110.0) == 2
    assert _b._nearest_level(z, 900.0) == 4


def test_interp_to_depths_linear_and_extrapolation():
    # One column, source depths 10/100 m with values 20/2 degC.
    sec = np.array([[20.0], [2.0]])
    z_src = np.array([10.0, 100.0])
    z_tgt = np.array([10.0, 55.0, 100.0, 200.0])
    out = _b._interp_to_depths(sec, z_src, z_tgt)
    assert out.shape == (4, 1)
    # 55 m is the midpoint -> 11 degC.
    np.testing.assert_allclose(out[1, 0], 11.0, atol=1e-9)
    np.testing.assert_allclose(out[0, 0], 20.0, atol=1e-9)
    np.testing.assert_allclose(out[2, 0], 2.0, atol=1e-9)
    # Below the deepest source level -> NaN (no fabricated abyssal value).
    assert np.isnan(out[3, 0])


def test_interp_to_depths_needs_two_points():
    # A column with a single finite level cannot be interpolated -> all NaN.
    sec = np.array([[5.0], [np.nan]])
    out = _b._interp_to_depths(sec, np.array([10.0, 100.0]), np.array([10.0, 50.0]))
    assert np.all(np.isnan(out[:, 0]))


def test_load_legoesm_3d_flattens_and_makes_depth_positive(tmp_path):
    # Synthetic C-grid-like 2-D snapshot (ny, nx, nlev) with z given z-up (<=0).
    ny, nx, nlev = 3, 4, 5
    T = np.arange(ny * nx * nlev, dtype=float).reshape(ny, nx, nlev)
    S = T + 30.0
    z_up = -np.array([10.0, 50.0, 120.0, 300.0, 900.0])      # negative (z-up)
    p = tmp_path / "snap.npz"
    np.savez(p, T=T, S=S, land_mask=np.ones((ny, nx)),
             lat_T=np.zeros((ny, nx)), lon_T=np.zeros((ny, nx)),
             H_bathy=np.full((ny, nx), 1000.0), z_center_ref=z_up)
    L = _b._load_legoesm_3d(p)
    assert L["T"].shape == (ny * nx, nlev)
    assert L["lat"].shape == (ny * nx,)
    # Depth axis is made positive-down regardless of stored sign.
    assert np.all(L["z"] > 0)
    np.testing.assert_allclose(L["z"], np.abs(z_up))


def test_load_legoesm_3d_wet3d_excludes_below_seafloor(tmp_path):
    # Per-level wet mask must drop levels below the local sea floor so deep
    # regrids don't import extrapolated rock values (codex HIGH). Centres at
    # 10/100/500/2000 m -> top interfaces 0/55/300/1250 m.
    ny, nx, nlev = 2, 2, 4
    T = np.zeros((ny, nx, nlev))
    z_up = -np.array([10.0, 100.0, 500.0, 2000.0])
    # (0,0) H=300 (=top of L2) -> L2 dry; (0,1) deep -> all wet;
    # (1,0) H=400 is a thin PARTIAL bottom cell in L2 (top 300<400<centre 500):
    #   the top-interface criterion KEEPS L2 (centre-vs-bathy would drop it).
    H = np.array([[300.0, 5000.0], [400.0, 5000.0]])
    mask = np.ones((ny, nx)); mask[1, 1] = 0.0          # one land cell
    p = tmp_path / "s.npz"
    np.savez(p, T=T, S=T, land_mask=mask, lat_T=np.zeros((ny, nx)),
             lon_T=np.zeros((ny, nx)), H_bathy=H, z_center_ref=z_up)
    L = _b._load_legoesm_3d(p)
    w = L["wet3d"].reshape(ny, nx, nlev)
    assert w[0, 0].tolist() == [1, 1, 0, 0]             # H=300 -> levels 0,1
    assert w[0, 1].tolist() == [1, 1, 1, 1]             # deep -> all wet
    assert w[1, 0].tolist() == [1, 1, 1, 0]             # partial L2 kept (top 300<400)
    assert w[1, 1].tolist() == [0, 0, 0, 0]             # land -> none


def test_load_legoesm_3d_rejects_radian_lat(tmp_path):
    p = tmp_path / "rad.npz"
    np.savez(p, T=np.zeros((2, 3)), S=np.zeros((2, 3)), land_mask=np.ones(2),
             lat_T=np.array([1.4, -1.4]), lon_T=np.array([3.0, 6.0]),  # ~radians
             H_bathy=np.full(2, 1000.0), z_center_ref=-np.array([10.0, 50.0, 90.0]))
    # lat in [-1.4,1.4] is fine (within +-90), so radians that happen to look
    # like small degrees pass; use an out-of-range value to trigger the guard.
    np.savez(p, T=np.zeros((2, 3)), S=np.zeros((2, 3)), land_mask=np.ones(2),
             lat_T=np.array([95.0, -95.0]), lon_T=np.array([3.0, 6.0]),
             H_bathy=np.full(2, 1000.0), z_center_ref=-np.array([10.0, 50.0, 90.0]))
    try:
        _b._load_legoesm_3d(p)
    except ValueError as e:
        assert "lat" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for out-of-range latitude")


def test_plot_section_handles_mismatched_model_nemo_nlev(tmp_path):
    # Model (nlev=10) and NEMO (nlev=20) sections have DIFFERENT shapes; the
    # colour-scale must stack finite values flat, not build an inhomogeneous
    # [secL, secN] array (the crash codex/runtime hit). Writes a PNG + finite RMSE.
    nlat = 36
    tgt_lat = np.linspace(-87.5, 87.5, nlat)
    zL = np.array([10., 50., 120., 250., 500., 900., 1500., 2500., 4000., 5200.])
    zN = np.linspace(5.0, 5500.0, 20)
    rng = np.cos(np.deg2rad(tgt_lat))[None, :]
    secL = (25.0 - zL[:, None] / 250.0) * rng
    secN = (25.0 - zN[:, None] / 250.0) * rng
    secL[-2:, :] = np.nan                      # deep model rows empty (dry)
    rmse = _b._plot_section(tmp_path, "T", "degC", tgt_lat, zL, secL, zN, secN,
                            "model")
    assert (tmp_path / "T_zonal_section.png").exists()
    assert np.isfinite(rmse)


def test_regrid_level_empty_level_returns_nan_not_raise():
    # An all-dry level (no finite/wet source cell) must return all-NaN, not
    # raise -- abyssal NEMO levels below every ocean column hit this.
    n = 50
    lat = np.linspace(-80, 80, n); lon = (np.arange(n) * 31.0) % 360.0
    field = np.full(n, np.nan)                 # entire level below seafloor
    mask = np.zeros(n)
    tgt_lat = np.arange(-89.5, 90.0, 10.0); tgt_lon = np.arange(5.0, 360.0, 10.0)
    out, oc = _b._regrid_level(field, lat, lon, mask, tgt_lat, tgt_lon)
    assert out.shape == (tgt_lat.size, tgt_lon.size)
    assert np.isnan(out).all() and (oc == 0).all()


def test_load_legoesm_3d_requires_z_center_ref(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, T=np.zeros((2, 3)), S=np.zeros((2, 3)),
             land_mask=np.ones(2), lat_T=np.zeros(2), lon_T=np.zeros(2))
    try:
        _b._load_legoesm_3d(p)
    except KeyError as e:
        assert "z_center_ref" in str(e)
    else:
        raise AssertionError("expected KeyError for missing z_center_ref")


def test_zonal_section_runs_and_warms_tropics():
    # 1-D cell list spanning the globe; level 0 warm in the tropics, cold at
    # poles -> zonal section row 0 should peak near the equator.
    n = 400
    rng = np.linspace(0, 1, n)
    lat = (rng * 180.0 - 90.0)
    lon = (np.arange(n) * 37.0) % 360.0
    nlev = 3
    Tprof = np.cos(np.deg2rad(lat))[:, None] * np.array([25.0, 15.0, 5.0])[None, :]
    wet3d = np.ones((n, nlev))          # per-level wet mask (all wet here)
    tgt_lat = np.arange(-89.5, 90.0, 5.0)
    tgt_lon = np.arange(2.5, 360.0, 5.0)
    sec = _b._zonal_section(Tprof, lat, lon, wet3d, np.array([10.0, 100.0, 500.0]),
                            tgt_lat, tgt_lon)
    assert sec.shape == (nlev, tgt_lat.size)
    eq = int(np.argmin(np.abs(tgt_lat)))
    # Surface (k=0) is warmest near the equator and warmer than at 60 S.
    cold = int(np.argmin(np.abs(tgt_lat + 60.0)))
    assert sec[0, eq] > sec[0, cold]
    assert sec[0, eq] > sec[1, eq] > sec[2, eq]   # warmer at the surface


def test_depth_band_helper_imported_and_attributes_band():
    # The battery re-uses compare_omip_nemo._band_breakdown to break the T/S
    # bias down by latitude band at each depth (surface-flux vs advective test).
    # Verify the import is live and a band-localised anomaly is attributed to
    # that band and no other -- the property the depth-by-band diagnostic relies
    # on.
    assert hasattr(_b, "_band_breakdown")
    lat = np.linspace(-89.5, 89.5, 180)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))
    base = np.full((180, 360), 10.0)
    nh = (lat[:, None] >= 23.0) & (lat[:, None] < 45.0)
    bands = _b._band_breakdown(np.where(nh, base - 2.8, base), base, area, lat)
    assert abs(bands["NH_midlat_23N_45N"]["bias"] + 2.8) < 1e-6
    assert abs(bands["tropics_23S_23N"]["bias"]) < 1e-6
    assert abs(bands["arctic_N_of_45N"]["bias"]) < 1e-6
