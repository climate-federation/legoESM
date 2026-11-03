"""IFS 49r1 CLOUDSC's advected cloud fraction as a tracer (vote VC18a, A3):
the registry entry, its place outside the water ledgers and the FV3-duo
column lane's slot contract for the extra passenger.  No CLOUDSC dependence:
the registry branch is gated on the scheme name only."""
from __future__ import annotations

from legoesm.core.tracers import make_full_moisture_registry
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import (
    moisture_registry_for, registry_scheme_config)


# --- registry ---------------------------------------------------------------

def test_registry_appends_the_cloud_fraction_only_for_ifs_cloudsc():
    full = make_full_moisture_registry().names
    ifs = moisture_registry_for("ifs_cloudsc")
    assert ifs.names == full + ("cloud_fraction_ifs",)
    info = ifs.tracers[-1]
    assert (info.units, info.positive_definite, info.transported) == ("1", True, True)
    for scheme in ("none", "kessler", "sundqvist", "morrison", "thompson", "p3"):
        names = moisture_registry_for(
            scheme, registry_scheme_config(ExperimentConfig(microphysics=scheme))).names
        assert "cloud_fraction_ifs" not in names, scheme


def test_cloud_fraction_is_outside_the_water_ledgers():
    from legoesm.core.conservation import is_borrow_eligible_tracer
    from legoesm.diagnostics.process_ledger import LEDGER_WATER_SPECIES
    assert "cloud_fraction_ifs" not in LEDGER_WATER_SPECIES
    assert not is_borrow_eligible_tracer("cloud_fraction_ifs")   # plain floor


# --- FV3-duo column lane: the registry names are passenger slots ------------

def test_fv3_duo_column_takes_the_ifs_registry_as_nwat6_plus_passengers():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    dyn = FV3DuoDynamicsModel(create_fv3_duo_grid(12, 3),
                              FV3DuoConfig(km=5, n_split=2, moist=True))
    names = moisture_registry_for("ifs_cloudsc").names
    col = FV3DuoColumnModel(dyn, tracer_names=names)
    assert col.nwat == 6
    assert col.tracer_names[6:] == ("N_c", "N_r", "N_i", "cloud_fraction_ifs")
