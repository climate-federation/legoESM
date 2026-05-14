"""AIMIP test-case registration sentinel.

Phase 1 of the AIMIP plan added an ``"aimip"`` template to
``legoesm.forcing.experiments.EXPERIMENT_TEMPLATES`` so future
training/validation scripts can resolve era-appropriate GHG
concentrations from the IC year automatically.  This test pins:

- the template metadata (years 2015-2020, transient GHG forcing)
- the GHG interpolation actually returns realistic 2015-2020 values
  (CO2 ~400-413 ppmv, CH4 ~1840-1890 ppbv, N2O ~328-333 ppbv)

A regression that drops the AIMIP template or breaks the GHG table
wiring trips this test with a clear pointer to
``src/legoesm/forcing/experiments.py``.
"""

from __future__ import annotations

import pytest

from legoesm.forcing.experiments import (
    EXPERIMENT_TEMPLATES,
    ghg_at_year,
)


def test_aimip_template_registered():
    """AIMIP template exists in EXPERIMENT_TEMPLATES with the canonical
    2015-2020 span + transient GHG forcing."""
    assert "aimip" in EXPERIMENT_TEMPLATES, (
        "AIMIP template missing from EXPERIMENT_TEMPLATES.  See "
        "src/legoesm/forcing/experiments.py."
    )
    tmpl = EXPERIMENT_TEMPLATES["aimip"]
    assert tmpl.name == "aimip"
    assert tmpl.start_year == 2015
    assert tmpl.end_year == 2020
    assert tmpl.forcing_type == "transient", (
        f"AIMIP forcing_type should be 'transient' (CMIP6 historical "
        f"trajectory), got {tmpl.forcing_type!r}."
    )
    assert tmpl.parent_experiment == "historical"
    # Base GHG = 2015 starting values per template definition.
    assert tmpl.base_co2_ppmv == pytest.approx(401.0, abs=1.0)
    assert tmpl.base_ch4_ppbv == pytest.approx(1877.0, abs=10.0)
    assert tmpl.base_n2o_ppbv == pytest.approx(328.9, abs=1.0)


@pytest.mark.parametrize(
    "year,exp_co2,exp_ch4,exp_n2o",
    [
        (2015, 400.0, 1843.0, 328.0),
        (2017, 405.0, 1860.0, 330.0),
        (2020, 412.0, 1886.0, 333.0),
    ],
)
def test_aimip_ghg_interpolation_realistic(year, exp_co2, exp_ch4, exp_n2o):
    """GHG interpolation for AIMIP returns the expected CMIP6
    historical-trajectory values for 2015, 2017, 2020 (within ~3%)."""
    co2, ch4, n2o = ghg_at_year("aimip", year)
    assert co2 == pytest.approx(exp_co2, rel=0.03), (
        f"AIMIP {year} CO2 = {co2:.2f} ppmv differs from expected "
        f"{exp_co2:.0f} ppmv by more than 3%."
    )
    assert ch4 == pytest.approx(exp_ch4, rel=0.03), (
        f"AIMIP {year} CH4 = {ch4:.0f} ppbv differs from expected "
        f"{exp_ch4:.0f} ppbv by more than 3%."
    )
    assert n2o == pytest.approx(exp_n2o, rel=0.03), (
        f"AIMIP {year} N2O = {n2o:.1f} ppbv differs from expected "
        f"{exp_n2o:.0f} ppbv by more than 3%."
    )
