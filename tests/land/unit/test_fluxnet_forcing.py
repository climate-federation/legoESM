"""Smoke tests for :mod:`legoesm.land.boundary_data.fluxnet_forcing`.

Loads the slim ONEFlux FULLSET CSVs for US-MMS (HR) and FI-Hyy (HH)
and checks shape/timestep/temperature/valid-mask/cos-zenith invariants.

The slim CSVs live under ``data/fluxnet/<site>/*_slim.csv``.  When
the checkout does not include them (minimal CI images), each test
skips cleanly — same pattern as :mod:`tests.land.unit.test_chats_obs`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.land.boundary_data.fluxnet_forcing import (
    SITES,
    load_fluxnet_forcing,
)

# Repo root: tests/land/unit/<file>.py -> parents[3] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLUXNET_ROOT = _REPO_ROOT / "data" / "fluxnet"

_US_MMS_CSV = (
    _FLUXNET_ROOT
    / "US-MMS"
    / "AMF_US-MMS_FLUXNET_FLUXMET_HR_1999-2023_v1.3_r1_slim.csv"
)
_FI_HYY_CSV = (
    _FLUXNET_ROOT
    / "FI-Hyy"
    / "ICOS_FI-Hyy_FLUXNET_FLUXMET_HH_1997-2024_v1.3_r1_slim.csv"
)
_US_TON_CSV = (
    _FLUXNET_ROOT
    / "US-Ton"
    / "AMF_US-Ton_FLUXNET_FLUXMET_HH_2001-2025_v1.3_r1_slim.csv"
)


# ---------------------------------------------------------------------------
# US-MMS (hourly): shape, dt, T range, valid mask, cos_zen day/night
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _US_MMS_CSV.exists(),
    reason=f"US-MMS slim FLUXNET CSV not present at {_US_MMS_CSV}",
)
def test_load_fluxnet_forcing_us_mms_hourly() -> None:
    forcing = load_fluxnet_forcing(_US_MMS_CSV, "US-MMS")

    # 1) Row count and dt: hourly file, 19752 rows, dt = 3600 s.
    assert len(forcing.time) == 19752
    assert forcing.dt_s == 3600.0
    assert forcing.site is SITES["US-MMS"]
    assert forcing.site.pft_clm == 7  # broadleaf_deciduous_temperate_tree

    # 3) Mean temperature in a physically sensible range for temperate
    # deciduous forest.  This catches C-vs-K unit bugs (K adds ~273 K).
    mean_T = float(np.nanmean(forcing.T_bot_K))
    assert 275.0 <= mean_T <= 300.0, f"US-MMS mean T_bot_K = {mean_T:.2f} K"

    # 4) At least half of the steps have full observations (QC == 0 and
    # every required forcing field finite).
    assert forcing.valid.sum() > 0.5 * len(forcing.time)

    # 5) cos_zen daytime/night sanity.
    #  * Night (local hours 00-04) should be exactly zero.
    #  * Local midday (hours 11-13) should include values > 0.5.
    local_hour = forcing.time.hour.to_numpy()
    night = (local_hour >= 0) & (local_hour < 4)
    midday = (local_hour >= 11) & (local_hour <= 13)
    assert night.any() and midday.any()
    assert np.all(forcing.cos_zen[night] == 0.0), \
        "expected all-zero cos_zen before local dawn"
    assert forcing.cos_zen[midday].max() > 0.5, \
        f"expected midday cos_zen > 0.5, got max={forcing.cos_zen[midday].max():.3f}"

    # Sanity on unit conversions: precip in kg/m2/s should be a
    # non-negative finite series where present (no C-vs-K style bugs).
    finite_precip = forcing.precip_kg_m2_s[np.isfinite(forcing.precip_kg_m2_s)]
    assert (finite_precip >= 0.0).all()
    # Surface pressure ~ 95-105 kPa -> 9.5e4 to 1.05e5 Pa at low elevation.
    finite_p = forcing.p_surface_Pa[np.isfinite(forcing.p_surface_Pa)]
    assert (finite_p > 8.0e4).all() and (finite_p < 1.1e5).all()


# ---------------------------------------------------------------------------
# FI-Hyy (half-hourly): confirms dt=1800 branch works
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _FI_HYY_CSV.exists(),
    reason=f"FI-Hyy slim FLUXNET CSV not present at {_FI_HYY_CSV}",
)
def test_load_fluxnet_forcing_fi_hyy_halfhourly() -> None:
    forcing = load_fluxnet_forcing(_FI_HYY_CSV, "FI-Hyy")

    # 2) Half-hourly file: 26352 rows, dt = 1800 s.
    assert len(forcing.time) == 26352
    assert forcing.dt_s == 1800.0
    assert forcing.site.pft_clm == 2  # needleleaf_evergreen_boreal_tree

    # Boreal ENF: mean T probably around 275 K (a bit below MMS).
    mean_T = float(np.nanmean(forcing.T_bot_K))
    assert 260.0 <= mean_T <= 290.0, f"FI-Hyy mean T_bot_K = {mean_T:.2f} K"


# ---------------------------------------------------------------------------
# US-Ton (half-hourly): sanity check the third pilot site loads
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _US_TON_CSV.exists(),
    reason=f"US-Ton slim FLUXNET CSV not present at {_US_TON_CSV}",
)
def test_load_fluxnet_forcing_us_ton_halfhourly() -> None:
    forcing = load_fluxnet_forcing(_US_TON_CSV, "US-Ton")

    assert len(forcing.time) == 26400
    assert forcing.dt_s == 1800.0
    assert forcing.site.pft_clm == 7  # savanna oak (broadleaf deciduous)


# ---------------------------------------------------------------------------
# build_atm2sfc_at: 1-column AtmToSurface for step i
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _US_MMS_CSV.exists(),
    reason=f"US-MMS slim FLUXNET CSV not present at {_US_MMS_CSV}",
)
def test_build_atm2sfc_at_us_mms_single_step() -> None:
    from legoesm.land.boundary_data.fluxnet_forcing import build_atm2sfc_at

    forcing = load_fluxnet_forcing(_US_MMS_CSV, "US-MMS")
    # Find a step where every input is finite so the builder produces
    # a fully-defined AtmToSurface.
    finite = (
        np.isfinite(forcing.T_bot_K)
        & np.isfinite(forcing.sw_down)
        & np.isfinite(forcing.lw_down)
        & np.isfinite(forcing.ws)
        & np.isfinite(forcing.p_surface_Pa)
        & np.isfinite(forcing.precip_kg_m2_s)
        & np.isfinite(forcing.co2_ppmv)
        & np.isfinite(forcing.q_bot)
    )
    idx = int(np.flatnonzero(finite)[0])

    atm = build_atm2sfc_at(forcing, idx)
    # Every field should be shape (1,).
    for name in (
        "sw_down", "lw_down", "precip_total", "precip_snow",
        "T_lowest", "q_lowest", "u_lowest", "v_lowest",
        "p_lowest", "p_surface", "rho_lowest", "cos_zenith",
        "co2_ppmv", "has_radiation", "has_precipitation",
    ):
        assert getattr(atm, name).shape == (1,), name

    # Sanity: rho_lowest positive, T_lowest matches forcing.T_bot_K.
    assert float(atm.rho_lowest[0]) > 0.5
    assert float(atm.T_lowest[0]) == pytest.approx(forcing.T_bot_K[idx])
