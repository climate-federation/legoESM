"""Canary for the shipped Sundqvist defaults.

Condensation (Zhao-Carr gscond) and precipitation (precpd) are pinned against
the running ccpp oracles in tests/atmosphere/microphysics/unit/
(test_sundqvist_gscond_oracle.py, test_sundqvist_zhao_carr_oracle.py).
"""

from __future__ import annotations

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig


def test_departure_default_constants():
    """Shipped ``SundqvistConfig`` defaults (canary): a silent change trips this.

    qc_crit 3e-4, c1 300 and c2 0.5 are the oracle's hard-coded values and
    evap_coeff 2e-5 the GFS operational evpco (tier-2 decisions R12-R14); the
    Gaussian peak/width fields have no oracle counterpart (non-default raises).
    """
    c = SundqvistConfig()
    assert c.rh_crit == 0.8
    assert c.auto_rate == 1.5e-4       # c_0 [1/s], GFS prautco
    assert c.qc_crit == 3e-4           # q_c,crit = oracle m_r [kg/kg]
    assert c.coalescence_enh_coeff == 300.0   # c1 (SBK89 F1)
    assert c.bergeron_enh_coeff == 0.5        # c2, oracle hard-coded
    assert c.bergeron_T_peak_K == constants.T_freeze - 15.0
    assert c.bergeron_T_width_K == 7.0
    assert c.evap_coeff == 2e-5             # GFS operational evpco
    assert (c.psautco_fine, c.psautco_coarse) == (8.0e-4, 5.0e-4)  # GFS namelist
    assert c.wminco_liquid == 1.0e-5        # fv3atm GFS_typedefs.F90:2854
    assert c.wminco_ice == 1.0e-5
