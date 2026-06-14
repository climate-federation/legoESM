"""Unit test for the NEMO sea-ice-concentration loader (--ice-albedo surrogate).

``load_nemo_siconc`` lives in the run driver (scripts/run/run_omip_core2.py); it
IDW-regrids NEMO ORCA1 annual `siconc` onto the model grid for the SW-albedo fix.
Validated against the real ORCA1 RUN_REF icemod file (skipped if absent), since
the loader's whole job is the curvilinear->target regrid + clip + land masking.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


def _runner_module():
    if not hasattr(_runner_module, "_mod"):
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / "scripts" / "run" / "run_omip_core2.py"
        if str(repo_root / "scripts") not in sys.path:
            sys.path.insert(0, str(repo_root / "scripts"))
        spec = importlib.util.spec_from_file_location("_run_omip_core2_for_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _runner_module._mod = mod
    return _runner_module._mod


def _latlon_target(nlat=90, nlon=180):
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat2d = np.broadcast_to(lat[:, None], (nlat, nlon)).copy()
    lon2d = np.broadcast_to(lon[None, :], (nlat, nlon)).copy()
    return lat2d, lon2d


def test_load_nemo_siconc_real_file():
    rom = _runner_module()
    sic_path = rom._SICONC_NC
    if not os.path.exists(sic_path):
        pytest.skip(f"NEMO icemod file absent: {sic_path}")
    lat2d, lon2d = _latlon_target()
    # grid is unused in the loader body; grid_type only labels the log line.
    sic = rom.load_nemo_siconc(None, "latlon", lat2d, lon2d)

    assert sic.shape == lat2d.shape
    assert np.isfinite(sic).all()
    assert sic.min() >= 0.0 and sic.max() <= 1.0
    # Physical: Antarctic + Arctic carry ice; the deep tropics do not.
    antarctic = lat2d < -60.0
    arctic = lat2d > 70.0
    tropics = np.abs(lat2d) < 20.0
    assert sic[antarctic].max() > 0.5, "no Antarctic sea ice found"
    assert sic[arctic].max() > 0.5, "no Arctic sea ice found"
    assert sic[tropics].max() < 0.15, "spurious tropical sea ice"


def test_load_nemo_siconc_land_mask_zeroes():
    """A land_mask (0 over land) zeroes siconc on land cells."""
    rom = _runner_module()
    sic_path = rom._SICONC_NC
    if not os.path.exists(sic_path):
        pytest.skip(f"NEMO icemod file absent: {sic_path}")
    lat2d, lon2d = _latlon_target()
    land_mask = np.ones_like(lat2d)
    land_mask[lat2d > 0.0] = 0.0   # call the whole NH "land"
    sic = rom.load_nemo_siconc(None, "latlon", lat2d, lon2d, land_mask=land_mask)
    assert np.all(sic[land_mask <= 0.5] == 0.0)


# ---------------------------------------------------------------------------
# Seasonal (12-month) siconc surrogate (--ice-albedo-seasonal) — codex HIGH fix
# for the NH cold bias.  The pure combine/ramp/indexing helpers are testable
# without the NEMO files; the full loader is checked against the real files.
# ---------------------------------------------------------------------------

def test_ice_presence_from_tos_monotone_and_bounds():
    rom = _runner_module()
    tos = np.array([-2.0, -1.8, -1.0, 0.0, 2.0, 10.0])
    p = rom._ice_presence_from_tos(tos, ice_edge_C=-1.0, ramp_C=1.0)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) < 0.0), "presence must decrease as SST warms"
    assert p[0] > 0.7, "near-freezing SST -> high ice presence"
    assert p[-1] < 0.05, "warm SST -> ~no ice"


def test_seasonal_siconc_mean_preserving():
    rom = _runner_module()
    # 4 cells: perennial (presence~1 all mo), seasonal-clip (iced only mo2 -> winter
    # conc saturates), mean-conserving (iced 6 mo, low annual -> no clip), ice-free.
    # annual = the annual-MEAN concentration.
    annual = np.array([0.9, 0.4, 0.1, 0.0])
    presence = np.zeros((12, 4))
    presence[:, 0] = 0.95                        # perennial
    presence[2, 1] = 1.0                         # seasonal: iced only month 2
    presence[0:6, 2] = 1.0                       # iced Jan-Jun (6 months)
    presence[:, 3] = 0.0                         # ice-free
    out = rom._seasonal_siconc_from_presence(annual, presence)
    assert out.shape == (12, 4)
    assert np.all((out >= 0.0) & (out <= 1.0))
    # Perennial cell ~unchanged at its annual value every month.
    assert np.allclose(out[:, 0], 0.9, atol=1e-6)
    # Seasonal cell: peak-ice month BOOSTED above the annual mean (mean-preserving),
    # capped at 1; warm months ~0.
    assert out[2, 1] >= annual[1] and out[2, 1] <= 1.0
    assert out[8, 1] == 0.0
    assert out[8, 1] < out[2, 1]                 # summer albedo relaxed
    # Mean-conserving cell (no clip): 12-month MEAN == annual; ice months elevated.
    assert abs(out[:, 2].mean() - annual[2]) < 1e-6
    assert abs(out[0, 2] - 0.2) < 1e-6           # 0.1 * (1/0.5)
    assert out[8, 2] == 0.0
    # Ice-free cell stays zero.
    assert np.allclose(out[:, 3], 0.0)


def test_siconc_at_step_month_indexing():
    rom = _runner_module()
    dt = 3600.0
    monthly = np.arange(12)[:, None, None] * np.ones((12, 2, 2))  # value == month
    # Jan (step 0) and a day in mid-Feb (~day 40) map to months 0 and 1.
    s_jan = rom._siconc_at_step(monthly, 0, dt, monthly=True)
    step_feb = int(40 * rom._SEC_PER_DAY / dt)
    s_feb = rom._siconc_at_step(monthly, step_feb, dt, monthly=True)
    assert np.all(s_jan == 0.0) and np.all(s_feb == 1.0)
    # Annual (monthly=False) returns the array unchanged; None passes through.
    annual = np.ones((2, 2)) * 0.5
    assert rom._siconc_at_step(annual, step_feb, dt, monthly=False) is annual
    assert rom._siconc_at_step(None, 0, dt, monthly=True) is None


def test_load_nemo_siconc_monthly_real_files():
    rom = _runner_module()
    if not (os.path.exists(rom._SICONC_NC) and os.path.exists(rom._TOS_MONTHLY_NC)):
        pytest.skip("NEMO icemod / monthly grid_T files absent")
    lat2d, lon2d = _latlon_target()
    sic = rom.load_nemo_siconc_monthly(None, "latlon", lat2d, lon2d)
    assert sic.shape == (12,) + lat2d.shape
    assert np.isfinite(sic).all()
    assert sic.min() >= 0.0 and sic.max() <= 1.0
    # NH seasonality: more ice in March (mo 2) than September (mo 8) north of 60N
    # — the whole point of the seasonal fix (summer albedo relaxed).
    nh = lat2d > 60.0
    assert sic[2][nh].mean() > sic[8][nh].mean(), "NH ice should retreat by Sept"
    # SH phase is opposite: more ice in Sept than March south of 60S.
    sh = lat2d < -60.0
    assert sic[8][sh].mean() > sic[2][sh].mean(), "SH ice should retreat by March"
