"""Direct tests for the SCM-vs-CRM near-surface humidity probe.

The probe's whole job is to attribute an evaporation deficit to one of two
multiplicands of the bulk formula, so the tests here pin exactly the properties
that an attribution rests on:

* the saturation value is the model's own Tetens curve at a KNOWN answer, so a
  swapped curve (AERK, Goff, an ice form) cannot pass;
* the two conventions are not silently interchanged;
* the decomposition identity the probe prints as its self-check is genuinely an
  identity, and the self-check is shown to FAIL on a perturbed row rather than
  being assumed non-vacuous.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.validate.check_scm_rce_surface_humidity import (
    SurfaceHumidityRow,
    crm_evap_from_hfls,
    crm_sensible_from_hfss,
    saturation_at_sst,
    surface_humidity_row,
)

#: Expected saturation mixing ratio at 300 K and 1014.80 hPa on the model's own
#: Tetens curve.  Worked by hand from ``legoesm.thermo`` (the curve is not
#: restated here -- see that module) and pinned so a swapped saturation
#: formulation cannot pass this file.
_P_SFC_PA = 101480.0
_R_SAT_300K = 0.022445


def test_saturation_matches_hand_computed_tetens():
    r_sat, q_sat = saturation_at_sst(sst_K=300.0, p_sfc_Pa=_P_SFC_PA)
    assert r_sat == pytest.approx(_R_SAT_300K, rel=2.0e-3)
    # The specific-humidity form is strictly smaller, by the ~2 % that makes
    # mixing the two conventions a real bias at this temperature.
    assert q_sat == pytest.approx(r_sat / (1.0 + r_sat), rel=1.0e-12)
    assert 0.015 < q_sat < r_sat


def test_saturated_air_reads_unit_relative_humidity():
    """A column AT saturation must read RH = 1 and a driver of exactly the
    surface-minus-air saturation difference -- the control that the two
    saturation calls use the same curve."""
    T_air = 297.0
    r_air, _ = saturation_at_sst(sst_K=T_air, p_sfc_Pa=_P_SFC_PA)
    row = surface_humidity_row(
        "saturated", T_air_K=T_air, r_air=r_air, p_air_Pa=_P_SFC_PA,
        evap_mm_day=1.0, evap_source="test",
        sst_K=300.0, p_sfc_Pa=_P_SFC_PA,
    )
    assert row.relative_humidity == pytest.approx(1.0, rel=1.0e-12)
    assert row.driver_kg_kg == pytest.approx(_R_SAT_300K - r_air, rel=2.0e-3)
    assert row.delta_T_K == pytest.approx(3.0)


def test_driver_and_transfer_reconstruct_the_evaporation():
    """``transfer * driver == E`` for every row, which is what makes the ratio
    decomposition an identity rather than an approximation."""
    row = surface_humidity_row(
        "col", T_air_K=297.5, r_air=0.0180, p_air_Pa=100000.0,
        evap_mm_day=1.37, evap_source="test",
        sst_K=300.0, p_sfc_Pa=_P_SFC_PA,
    )
    assert row.transfer_mm_day_per_kg_kg * row.driver_kg_kg == pytest.approx(
        1.37, rel=1.0e-12)


def test_ratio_decomposition_is_exact_and_the_check_can_fail():
    """E_ratio == driver_ratio * transfer_ratio, and a PERTURBED row breaks it.

    The second half is the non-vacuity gate: an identity that cannot fail is
    not a self-check.
    """
    kw = dict(p_air_Pa=100000.0, evap_source="test",
              sst_K=300.0, p_sfc_Pa=_P_SFC_PA)
    crm = surface_humidity_row("crm", T_air_K=296.9, r_air=0.0175,
                               evap_mm_day=2.40, **kw)
    scm = surface_humidity_row("scm", T_air_K=299.0, r_air=0.0195,
                               evap_mm_day=1.30, **kw)
    lhs = scm.evap_mm_day / crm.evap_mm_day
    rhs = ((scm.driver_kg_kg / crm.driver_kg_kg)
           * (scm.transfer_mm_day_per_kg_kg / crm.transfer_mm_day_per_kg_kg))
    assert abs(lhs - rhs) < 1.0e-12

    broken = SurfaceHumidityRow(**{**vars(scm),
                                   "transfer_mm_day_per_kg_kg":
                                       scm.transfer_mm_day_per_kg_kg * 1.05})
    rhs_broken = ((broken.driver_kg_kg / crm.driver_kg_kg)
                  * (broken.transfer_mm_day_per_kg_kg
                     / crm.transfer_mm_day_per_kg_kg))
    assert abs(lhs - rhs_broken) > 1.0e-9


def test_zero_driver_returns_nan_not_an_exception():
    """Air already at the surface saturation value is a degenerate column; the
    probe must surface it as NaN, never as a divide-by-zero crash or -- worse --
    a plausible finite number (CLAUDE.md: a valid-looking sentinel passes every
    downstream guard)."""
    r_sat, _ = saturation_at_sst(sst_K=300.0, p_sfc_Pa=_P_SFC_PA)
    row = surface_humidity_row(
        "degenerate", T_air_K=300.0, r_air=r_sat, p_air_Pa=_P_SFC_PA,
        evap_mm_day=0.5, evap_source="test",
        sst_K=300.0, p_sfc_Pa=_P_SFC_PA,
    )
    assert row.driver_kg_kg == 0.0
    assert math.isnan(row.transfer_mm_day_per_kg_kg)


def test_missing_archive_flux_files_return_none(tmp_path: Path):
    """An absent measurement must read as absent so the caller can fall back to
    the equilibrium identity AND say that it did."""
    assert crm_evap_from_hfls(tmp_path) is None
    assert crm_sensible_from_hfss(tmp_path) is None
