"""GYRE namelist validation refuses non-integer sizes and inert edits."""

import pytest

from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    GYRE_R4_NAMELIST,
    GYRE_SHIPPED_NAMELIST,
    validate_gyre_namelist,
)


def test_shipped_and_r4_namelists_validate():
    validate_gyre_namelist(GYRE_SHIPPED_NAMELIST)
    validate_gyre_namelist(GYRE_R4_NAMELIST)


@pytest.mark.parametrize("value", [2.5, 4.0, True, 0])
def test_nn_gyre_must_be_a_positive_int(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_gyre_namelist(GYRE_R4_NAMELIST._replace(nn_GYRE=value))


@pytest.mark.parametrize("field", ["rn_csmc", "rn_minfac", "rn_maxfac"])
def test_laplacian_mode_refuses_inert_smagorinsky_edits(field):
    assert GYRE_SHIPPED_NAMELIST.ln_dynldf_lap
    with pytest.raises(ValueError, match="does not read them"):
        validate_gyre_namelist(GYRE_SHIPPED_NAMELIST._replace(**{field: 9.0}))


@pytest.mark.parametrize("field", ["nn_itend", "nn_stock", "nn_write", "ptimincr"])
def test_run_controls_must_be_integers(field):
    value = getattr(GYRE_R4_NAMELIST, field) + 0.5
    with pytest.raises(ValueError, match="must be an integer"):
        validate_gyre_namelist(GYRE_R4_NAMELIST._replace(**{field: value}))
