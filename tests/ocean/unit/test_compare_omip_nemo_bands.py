"""Analytic tests for the latitude-band breakdown in
``scripts/validate/compare_omip_nemo.py``.

``_band_breakdown`` is the helper the OMIP/NEMO comparison uses to attribute a
global SST/SSS/MLD bias to a latitude band -- e.g. to check whether the mixed
layer is too deep *exactly* in the NH-midlat band where SST is coldest (the
entrainment link).  The test plants a band-localised anomaly and asserts it is
attributed to that band and to no other.

Loaded by path because ``scripts/`` is not an importable package.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "compare_omip_nemo",
    _ROOT / "scripts" / "validate" / "compare_omip_nemo.py")
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)


def _grid(n_lat=180, n_lon=360):
    tgt_lat = np.linspace(-89.5, 89.5, n_lat)
    area = (np.cos(np.deg2rad(tgt_lat))[:, None]
            * np.ones((1, n_lon)))          # ocean everywhere, cos-lat weight
    return tgt_lat, area


def test_band_anomaly_attributed_to_that_band_only():
    tgt_lat, area = _grid()
    base = np.full((tgt_lat.size, area.shape[1]), 50.0)   # nemo MLD 50 m
    lego = base.copy()
    # +60 m deep anomaly ONLY in the NH-midlat band [23, 45).
    nh = (tgt_lat[:, None] >= 23.0) & (tgt_lat[:, None] < 45.0)
    lego = np.where(nh, base + 60.0, base)

    bands = _c._band_breakdown(lego, base, area, tgt_lat)

    assert abs(bands["NH_midlat_23N_45N"]["bias"] - 60.0) < 1e-6
    for name in ("antarctic_S_of_45S", "SH_midlat_45S_23S",
                 "tropics_23S_23N", "arctic_N_of_45N"):
        assert abs(bands[name]["bias"]) < 1e-6, name


def test_empty_band_returns_none():
    # Area zero outside the tropics -> the polar/midlat bands have no wet cells.
    tgt_lat, area = _grid()
    trop_only = area * ((tgt_lat[:, None] >= -23.0) & (tgt_lat[:, None] < 23.0))
    f = np.ones((tgt_lat.size, area.shape[1]))
    bands = _c._band_breakdown(f, f, trop_only, tgt_lat)
    assert bands["arctic_N_of_45N"] is None
    assert bands["antarctic_S_of_45S"] is None
    assert bands["tropics_23S_23N"] is not None


def test_band_breakdown_is_area_weighted():
    # A uniform offset -> every populated band reports exactly that bias,
    # independent of the cos-lat weighting.
    tgt_lat, area = _grid()
    base = np.full((tgt_lat.size, area.shape[1]), 12.0)
    bands = _c._band_breakdown(base + 2.5, base, area, tgt_lat)
    for name, bs in bands.items():
        if bs is not None:
            assert abs(bs["bias"] - 2.5) < 1e-6, name


def test_sst_verdict_capped_by_enso_box():
    """Global rmse 0.94 with nino3 +2.78 must NOT be excellent (2026-08-18)."""
    boxes = {"nino3_5S5N_150W90W": {"bias": 2.78, "rmse": 2.93, "corr": 0.17},
             "nino4_5S5N_160E150W": {"bias": 1.38, "rmse": 1.43, "corr": 0.90}}
    v, capped = _c.capped_sst_verdict(0.94, boxes)
    assert v == "poor" and capped[0] == "nino3_5S5N_150W90W"
    # moderate box bias -> good, not excellent
    v, capped = _c.capped_sst_verdict(0.94, {"nino3_5S5N_150W90W":
                                             {"bias": -1.5, "rmse": 1.6,
                                              "corr": 0.8}})
    assert v == "good"
    # clean boxes leave the global tier alone
    v, capped = _c.capped_sst_verdict(0.94, {"nino3_5S5N_150W90W":
                                             {"bias": 0.3, "rmse": 0.5,
                                              "corr": 0.95}})
    assert v == "excellent" and capped is None
    # empty/None boxes tolerated
    assert _c.capped_sst_verdict(0.94, {})[0] == "excellent"
    assert _c.capped_sst_verdict(0.94, {"b": None})[0] == "excellent"


def _write_snapshot(tmp_path, with_means):
    T = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    kw = dict(T=T, S=T + 30.0, lat_T=np.zeros((2, 3)), lon_T=np.zeros((2, 3)),
              land_mask=np.ones((2, 3)))
    if with_means:
        kw.update(T_mean=T + 1.0, S_mean=T + 31.0, T_mean_hw=T + 2.0,
                  S_mean_hw=T + 32.0, mld_mean=np.full((2, 3), 17.0),
                  flux_mean_window_s=np.asarray(5 * 86400.0))
    p = tmp_path / "snapshot_day0005.npz"
    np.savez(p, **kw)
    return p, T


def test_loader_use_mean_reads_the_window_means_not_the_instantaneous_state(tmp_path):
    import pytest
    """NEMO's 5-day files are window means; the loader must hand the scorers
    the plain means at the surface, the thickness-weighted means for the
    columns and the window-mean MLD -- and the instantaneous state otherwise."""
    p, T = _write_snapshot(tmp_path, with_means=True)
    L = _c._load_legoesm(p, use_mean=True)
    np.testing.assert_array_equal(L["sst"], T[..., 0] + 1.0)
    np.testing.assert_array_equal(L["sss"], T[..., 0] + 31.0)
    np.testing.assert_array_equal(L["T3d"], T + 2.0)
    np.testing.assert_array_equal(L["S3d"], T + 32.0)
    np.testing.assert_array_equal(L["mld_mean"], 17.0)
    assert L["window_days"] == 5.0
    assert _c.require_window(L, True) == "window_mean(5d)"
    assert _c.require_window(L, False) == "instantaneous"
    L["window_days"] = 1.0
    with pytest.raises(SystemExit, match="not window-matched"):
        _c.require_window(L, True)
    L0 = _c._load_legoesm(p)
    np.testing.assert_array_equal(L0["sst"], T[..., 0])
    np.testing.assert_array_equal(L0["T3d"], T)
    assert L0["mld_mean"] is None


def test_loader_use_mean_without_mld_mean_loads_with_mld_none(tmp_path):
    """--state-accumulate without --mld-accumulate: T/S means are matched, the
    MLD is None so the scorers SKIP it loudly (never MLD of the mean state)."""
    p, T = _write_snapshot(tmp_path, with_means=True)
    d = dict(np.load(p)); d.pop("mld_mean"); np.savez(p, **d)
    L = _c._load_legoesm(p, use_mean=True)
    np.testing.assert_array_equal(L["sst"], T[..., 0] + 1.0)
    assert L["mld_mean"] is None


def test_loader_use_mean_refuses_a_snapshot_without_means(tmp_path):
    """No silent fallback to the instantaneous state (the phase confound)."""
    import pytest
    p, _ = _write_snapshot(tmp_path, with_means=False)
    with pytest.raises(SystemExit, match="T_mean"):
        _c._load_legoesm(p, use_mean=True)
