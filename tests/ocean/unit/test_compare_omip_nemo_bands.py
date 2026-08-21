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
