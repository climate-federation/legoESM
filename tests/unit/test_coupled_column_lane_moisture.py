"""The coupled drivers read the atmosphere's CURRENT moisture on every lane.

The column loop (MPAS grid or ``dycore.column_lane``) advances the tracers it
carries in ``state.tracers``; the model driver's own tracer store keeps their
initial copy there.  The coupler's near-surface humidity (ocean bulk fluxes,
moist density) and water diagnostics must read the advanced field.
"""
from __future__ import annotations

import types

import jax.numpy as jnp
import pytest

INITIAL = jnp.full((4, 3), 1.0e-3)
ADVANCED = jnp.full((4, 3), 7.0e-3)


def _atm(grid_type, column_lane, carried=("q_v", "q_c")):
    from legoesm.driver.model_driver import ModelDriver
    atm = types.SimpleNamespace(
        config=types.SimpleNamespace(
            grid=types.SimpleNamespace(grid_type=grid_type),
            dycore=types.SimpleNamespace(column_lane=column_lane)),
        tracers={"q_v": INITIAL, "q_c": INITIAL},
        state=types.SimpleNamespace(tracers={
            k: types.SimpleNamespace(data=ADVANCED) for k in carried} or None),
    )
    helper = getattr(ModelDriver, "current_tracer", None)
    if helper is not None:
        atm.current_tracer = types.MethodType(helper, atm)
    return atm


def _coupled_q_v(atm):
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    atm.q_v = atm.tracers["q_v"]          # the driver store's accessor
    return CoupledESMDriver.q_v.fget(types.SimpleNamespace(_atm=atm))


@pytest.mark.parametrize("grid_type, column_lane", [
    ("mpas", False), ("latlon", True), ("cubed_sphere", True)])
def test_column_loop_reads_advanced_moisture(grid_type, column_lane):
    atm = _atm(grid_type, column_lane)
    assert jnp.array_equal(_coupled_q_v(atm), ADVANCED)
    assert jnp.array_equal(atm.current_tracer("q_c"), ADVANCED)


def test_other_lanes_read_the_driver_store():
    atm = _atm("latlon", False)
    assert jnp.array_equal(_coupled_q_v(atm), INITIAL)


def test_tracer_the_loop_does_not_carry_comes_from_the_driver_store():
    """A dry column-loop run (state.tracers None) keeps the store's q_v."""
    atm = _atm("mpas", False, carried=())
    assert jnp.array_equal(atm.current_tracer("q_v"), INITIAL)
    assert atm.current_tracer("q_g") is None
