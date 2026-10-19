"""Pre-registered SWG online-learning experiment-control decks."""
from __future__ import annotations

from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
)

# Resolution-control viscosity.  This is an experiment control, not the
# NEMO-certified dg50 training model.
SWG_DG50_NU80_NAMELIST = SWG_DG50_NAMELIST._replace(rn_Lv_m=80.0)

# Reference-adequacy sensitivity: double the fine deck's resolution and halve
# its clock so the gravity-wave Courant number is unchanged.
SWG_DG3125_NAMELIST = SWG_MADDISON_NAMELIST._replace(
    nn_SWG=32,
    nn_itend=3_732_480,
    nn_write=3_456,
    nn_stock=3_732_480,
    rn_Dt_s=25.0,
)

__all__ = ["SWG_DG50_NU80_NAMELIST", "SWG_DG3125_NAMELIST"]
