"""Tests for the morrison SAM/MG flavor switch (E3SM Morrison-Gettelman vs SAM).

The "mg" flavor must override the parameters where E3SM MG (micro_mg_utils.F90)
differs from SAM, and the mg_ferrier ice->snow path must run finite. "sam"
must leave the SAM/gSAM defaults untouched (the plane-CRM validation target).
"""
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    resolve_morrison_flavor, morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


def test_default_flavor_is_mg_global():
    # Global/GCM default per project policy.
    assert MorrisonConfig().morrison_flavor == "mg"


def test_sam_flavor_unchanged():
    sam = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="sam"))
    assert sam.lami_max == 1.0 / 1.0e-6
    assert sam.snow_aggregation_eii == 0.1
    assert sam.rho_snow == 100.0
    assert sam.fall_b_i == 0.865
    assert sam.ice_to_snow_scheme == "m2005_autoconv"


def test_mg_flavor_overrides_match_e3sm():
    mg = resolve_morrison_flavor(MorrisonConfig(morrison_flavor="mg"))
    assert mg.lami_max == pytest.approx(1.0 / 10.0e-6)   # LAMMAXI
    assert mg.snow_aggregation_eii == 0.5                # eii
    assert mg.rho_snow == 250.0                          # rhosn
    assert mg.fall_b_i == 1.0                            # bi
    assert mg.ice_to_snow_scheme == "mg_ferrier"
    # warm rain + ice deposition already MG-faithful in both flavors
    assert mg.warm_rain_scheme == "kk2000"
    assert mg.ice_deposition_scheme == "m2005"


def test_unknown_flavor_raises():
    with pytest.raises(ValueError):
        resolve_morrison_flavor(MorrisonConfig(morrison_flavor="bogus"))


def _run(flavor):
    ncol, nlev = 4, 1
    c = lambda x: jnp.full((ncol, nlev), x)
    hyd = HydrometeorState(
        q_c=c(1e-4), q_r=c(1e-5), q_i=c(5e-5), q_s=c(2e-5), q_g=c(0.0),
        N_c=c(0.0), N_r=c(1e4), N_i=c(2e6), N_s=c(1e3), N_g=c(0.0))
    return morrison_microphysics(
        c(250.0), c(1.2e-3), hyd, c(4e4), c(4e4), c(0.4), c(300.0), 6.0,
        MorrisonConfig(morrison_flavor=flavor))


def test_both_flavors_finite_and_differ():
    sam_out, mg_out = _run("sam"), _run("mg")
    for out, fl in ((sam_out, "sam"), (mg_out, "mg")):
        for a in ("dT_dt", "dq_v_dt", "dq_i_dt", "dq_s_dt", "dN_i_dt"):
            assert bool(jnp.all(jnp.isfinite(getattr(out, a)))), (fl, a)
    # The flavor must actually change the ice->snow conversion: SAM PRCI
    # (supersaturation-driven) and MG Ferrier (180-s) give different ice
    # tendencies on the same column.
    assert float(jnp.max(jnp.abs(sam_out.dq_i_dt - mg_out.dq_i_dt))) > 0.0
