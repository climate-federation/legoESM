"""NEMO's river-runoff WATER, and the dilution operand it must stay out of.

NEMO applies the runoff volume in two places -- the horizontal divergence
(``sbcrnf.F90``'s surface arm, called from ``divhor.F90``) and the barotropic
sea-surface forcing ``r1_rho0*(emp - rnf)`` (``stp2d.F90:278-281``) -- and the
tracer content it carries in a third (``trasbc.F90``'s river-runoff block).
The stage-1/2 concentration/dilution term reads ``emp`` ALONE
(``trasbc.F90:282-288``), and NEMO's ``emp`` never carries the runoff.

legoESM's net freshwater flux is ``precip - evap + runoff + ice_fw``, which is
the right operand for the sea surface and the wrong one for the dilution: with
the runoff left in, the dilution deposits ``rnf*T_top/rho0/h``, a second and
nearly identical copy of the runoff's own heat ``MAX(sst,0)*rnf/rho0/h``.

These are the controls for that statement.  The bitwise proof on the
production step, against the ORCA2 record's own frames, is the round-13 water
gate; what is locked here is the helper's spelling and the fact that the model
reads the emp channel rather than the net one.
"""

import numpy as np
import pytest

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_lock_exchange_zco_card,
)
from legoesm.ocean.freshwater import FreshwaterForcing, net_freshwater_flux


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _fields(shape, seed):
    rng = np.random.default_rng(seed)
    zeros = np.zeros(shape, dtype=np.float64)
    evaporation = rng.uniform(-2.0e-5, 2.0e-5, shape)
    runoff = np.zeros(shape, dtype=np.float64)
    flat = runoff.reshape(-1)
    mouths = max(3, flat.size // 5)
    flat[rng.choice(flat.size, size=mouths, replace=False)] = rng.uniform(
        1.0e-6, 5.0e-5, size=mouths)
    return zeros, evaporation, runoff


def test_the_emp_channel_leaves_the_runoff_out_of_the_sum():
    """Excluded from the sum, never subtracted after it.

    ``(x + r) - r`` is not ``x``.  The first spelling of this statement
    subtracted the runoff from the assembled net flux and the ORCA2 gate
    caught it on 328 cells, so the failure mode is locked here with a case
    where the two spellings differ by everything.
    """
    tiny = np.array([1.0e-30], dtype=np.float64)
    one = np.array([1.0], dtype=np.float64)
    zero = np.array([0.0], dtype=np.float64)
    forcing = FreshwaterForcing(precip=zero, evap=-tiny, runoff=one,
                                ice_fw=zero, restoring=zero)
    emp_channel = net_freshwater_flux(forcing, include_runoff=False)
    np.testing.assert_array_equal(_bits(emp_channel), _bits(tiny))
    subtract_after = net_freshwater_flux(forcing) - forcing.runoff
    assert float(subtract_after[0]) == 0.0
    assert _bits(emp_channel)[0] != _bits(subtract_after)[0]


def test_the_default_still_carries_the_runoff():
    """The sea-surface channel needs it: ``stp2d.F90:278-281`` is emp - rnf."""
    shape = (4, 5)
    zeros, evaporation, runoff = _fields(shape, seed=7)
    forcing = FreshwaterForcing(precip=zeros, evap=evaporation, runoff=runoff,
                                ice_fw=zeros, restoring=zeros)
    with_runoff = net_freshwater_flux(forcing)
    without = net_freshwater_flux(forcing, include_runoff=False)
    assert int(np.count_nonzero(_bits(with_runoff) != _bits(without))) > 0
    np.testing.assert_allclose(
        np.asarray(with_runoff) - np.asarray(without), runoff, atol=1.0e-18)


def test_a_card_that_resolves_no_runoff_is_bit_identical():
    """The scoping claim GYRE's identity rests on.

    Every card but ORCA2 supplies a runoff of exactly +0.0, and ``x + 0.0`` is
    bitwise ``x`` for every value this sum can hold, so excluding it moves
    nothing there.
    """
    shape = (4, 5)
    zeros, evaporation, _ = _fields(shape, seed=11)
    forcing = FreshwaterForcing(precip=zeros, evap=evaporation, runoff=zeros,
                                ice_fw=zeros, restoring=zeros)
    np.testing.assert_array_equal(
        _bits(net_freshwater_flux(forcing)),
        _bits(net_freshwater_flux(forcing, include_runoff=False)))


@pytest.fixture(scope="module")
def lock_arms():
    """Two steps whose NET freshwater is identical and whose dilution is not.

    Arm P carries the river water as precipitation, arm Q as runoff.  Their
    net fluxes are bitwise equal (floating-point addition is commutative), so
    the sea surface sees the same forcing in both.  Under NEMO's statement the
    DILUTION differs -- ``precip`` is in ``emp`` and the runoff is not -- so
    the two steps must part company.  With the runoff left in the dilution
    operand the two arms become the same calculation.
    """
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    state = card.recipe.initial_state
    shape = np.asarray(state.eta.data).shape
    zeros, evaporation, runoff = _fields(shape, seed=5)
    arms = {}
    for name, forcing in (
        ("as_precipitation", FreshwaterForcing(
            precip=runoff, evap=evaporation, runoff=zeros,
            ice_fw=zeros, restoring=zeros)),
        ("as_runoff", FreshwaterForcing(
            precip=zeros, evap=evaporation, runoff=runoff,
            ice_fw=zeros, restoring=zeros)),
        ("no_water", FreshwaterForcing(
            precip=zeros, evap=evaporation, runoff=zeros,
            ice_fw=zeros, restoring=zeros)),
    ):
        model = model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                expose_tracer_stage=1))
        arms[name] = model.step(state, dt=card.dt_s, freshwater=forcing)
    arms["net_is_identical"] = np.array_equal(
        _bits(net_freshwater_flux(FreshwaterForcing(
            precip=runoff, evap=evaporation, runoff=zeros,
            ice_fw=zeros, restoring=zeros))),
        _bits(net_freshwater_flux(FreshwaterForcing(
            precip=zeros, evap=evaporation, runoff=runoff,
            ice_fw=zeros, restoring=zeros))))
    return arms


def test_the_two_arms_really_do_share_one_net_flux(lock_arms):
    """The control: without this the next test is not a controlled pair."""
    assert lock_arms["net_is_identical"]


@pytest.mark.parametrize("tracer", ("T", "S"))
def test_the_stage_dilution_operand_is_nemos_emp(lock_arms, tracer):
    """River water dilutes as ``emp`` only when it is called precipitation.

    Fails when the statement is reverted: with the runoff back in the operand
    both arms compute the same dilution and the stage-1 tracer fields become
    bitwise equal.
    """
    as_precip = np.asarray(getattr(lock_arms["as_precipitation"],
                                   tracer).data)
    as_runoff = np.asarray(getattr(lock_arms["as_runoff"], tracer).data)
    moved = float(np.abs(as_precip - as_runoff).max())
    assert moved > 1.0e-8, (
        f"stage-1 {tracer} did not distinguish river water called "
        f"precipitation from river water called runoff (max {moved:g}); the "
        "dilution operand is not NEMO's emp")


def test_the_runoff_water_reaches_the_step_at_all(lock_arms):
    """Non-vacuity for the WATER half: it is not an inert field."""
    with_water = np.asarray(lock_arms["as_runoff"].eta.data)
    without = np.asarray(lock_arms["no_water"].eta.data)
    assert int(np.count_nonzero(_bits(with_water) != _bits(without))) > 0
