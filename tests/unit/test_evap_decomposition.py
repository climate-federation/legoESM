"""Direct tests for scripts/validate/evap_decomposition.py (#847 diagnostic).

Exercises the pure-compute core on a synthetic two-regime ocean where every
term of E = rho * L_v * (C_E*U) * dq is known analytically, so the
decomposition must recover the planted transfer velocity and gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from scripts.validate.evap_decomposition import (
    _Q_SAT_SALINE_FACTOR,
    decompose_evap,
)


def _synthetic_ocean(ce_u: float, rh_air: float):
    """Build a uniform all-ocean state with a planted C_E*U and BL RH.

    Returns (fields dict, expected dq [kg/kg], expected hfls [W/m^2]).
    """
    nlat, nlon = 8, 16
    lat = np.linspace(-60.0, 60.0, nlat)
    shape = (nlat, nlon)

    tos = np.full(shape, 300.0)          # K
    tas = np.full(shape, 298.0)          # K
    ps = np.full(shape, 101325.0)        # Pa

    q_sat_sst = _Q_SAT_SALINE_FACTOR * np.asarray(
        saturation_mixing_ratio(tos, ps))
    hus_low = rh_air * q_sat_sst         # BL humidity as a fraction of q_sat
    dq = q_sat_sst - hus_low

    virt = 1.0 + (1.0 / constants.epsilon - 1.0) * hus_low
    rho = ps / (constants.R_d * tas * virt)
    hfls = rho * constants.L_v * ce_u * dq

    fields = dict(
        hfls=hfls, hus_low=hus_low, tos=tos, tas=tas, ps=ps,
        sftlf_pct=np.zeros(shape), lat=lat,
    )
    return fields, float(dq.mean()), float(hfls.mean())


def test_recovers_planted_transfer_velocity_and_gradient():
    ce_u = 8.0e-3                        # COARE-typical C_E*U [m/s]
    fields, dq_true, hfls_true = _synthetic_ocean(ce_u, rh_air=0.75)
    r = decompose_evap(**fields)
    assert r["ce_u_bulk_m_s"] == pytest.approx(ce_u, rel=1e-6)
    assert r["dq_ocean_mean_g_kg"] == pytest.approx(dq_true * 1e3, rel=1e-6)
    assert r["hfls_ocean_mean_W_m2"] == pytest.approx(hfls_true, rel=1e-6)
    assert r["n_ocean_cells"] == fields["hfls"].size


def test_humid_bl_shows_up_as_dq_collapse_not_transfer():
    """The #847 hypothesis pattern: same transfer velocity, near-saturated BL
    => hfls drops via dq while C_E*U stays put.  The decomposition must
    attribute the deficit to dq."""
    ce_u = 8.0e-3
    dry, dq_dry, hfls_dry = _synthetic_ocean(ce_u, rh_air=0.75)
    wet, dq_wet, hfls_wet = _synthetic_ocean(ce_u, rh_air=0.92)
    r_dry = decompose_evap(**dry)
    r_wet = decompose_evap(**wet)
    # flux deficit driven by dq (about (1-0.92)/(1-0.75) ~ 3x)
    assert hfls_wet < 0.4 * hfls_dry
    assert r_wet["dq_ocean_mean_g_kg"] < 0.4 * r_dry["dq_ocean_mean_g_kg"]
    # ...while the recovered transfer velocity is unchanged
    assert r_wet["ce_u_bulk_m_s"] == pytest.approx(
        r_dry["ce_u_bulk_m_s"], rel=1e-6)


def test_land_and_nonfinite_cells_excluded():
    fields, _, _ = _synthetic_ocean(8.0e-3, rh_air=0.75)
    sftlf = fields["sftlf_pct"].copy()
    sftlf[:, 0] = 100.0                  # a land column
    fields["sftlf_pct"] = sftlf
    hfls = fields["hfls"].copy()
    hfls[0, 1] = np.nan                  # a missing cell
    fields["hfls"] = hfls
    r = decompose_evap(**fields)
    assert r["n_ocean_cells"] == hfls.size - hfls.shape[0] - 1


def test_all_land_raises():
    fields, _, _ = _synthetic_ocean(8.0e-3, rh_air=0.75)
    fields["sftlf_pct"] = np.full_like(fields["sftlf_pct"], 100.0)
    with pytest.raises(ValueError, match="no valid ocean cells"):
        decompose_evap(**fields)
