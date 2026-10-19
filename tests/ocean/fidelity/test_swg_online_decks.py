"""Direct tests for the online-learning resolution-control decks."""
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
)
from legoesm.ocean.fidelity.swg_online_decks import (
    SWG_DG50_NU80_NAMELIST,
    SWG_DG3125_NAMELIST,
)


def test_nu80_changes_only_viscosity_and_dg3125_preserves_courant():
    changed = {
        field for field in SWG_DG50_NAMELIST._fields
        if getattr(SWG_DG50_NAMELIST, field)
        != getattr(SWG_DG50_NU80_NAMELIST, field)
    }
    assert changed == {"rn_Lv_m"}
    assert SWG_DG50_NU80_NAMELIST.rn_Lv_m == 80.0
    assert SWG_DG3125_NAMELIST.nn_SWG == 32
    assert SWG_DG3125_NAMELIST.rn_Dt_s == 25.0
    fine_courant_clock = (
        SWG_MADDISON_NAMELIST.rn_Dt_s
        / (SWG_MADDISON_NAMELIST.rn_dx_m / SWG_MADDISON_NAMELIST.nn_SWG))
    sensitivity_courant_clock = (
        SWG_DG3125_NAMELIST.rn_Dt_s
        / (SWG_DG3125_NAMELIST.rn_dx_m / SWG_DG3125_NAMELIST.nn_SWG))
    assert sensitivity_courant_clock == fine_courant_clock
