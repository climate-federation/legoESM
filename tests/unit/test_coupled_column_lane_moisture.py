"""The coupler reads the atmosphere's CURRENT moisture on every lane.

The column loop (MPAS grid or ``dycore.column_lane``) advances moisture in
``state.tracers``; the driver's own tracer store keeps the initial copy there.
The coupler's near-surface humidity (ocean bulk fluxes) and water diagnostics
must read the advanced field, not the initial one.
"""
from __future__ import annotations

import types

import jax.numpy as jnp
import pytest


def _stub(grid_type, column_lane):
    initial = jnp.full((4, 3), 1.0e-3)
    advanced = jnp.full((4, 3), 7.0e-3)
    atm = types.SimpleNamespace(
        config=types.SimpleNamespace(
            grid=types.SimpleNamespace(grid_type=grid_type),
            dycore=types.SimpleNamespace(column_lane=column_lane)),
        q_v=initial, q_c=initial,
        state=types.SimpleNamespace(tracers={
            "q_v": types.SimpleNamespace(data=advanced),
            "q_c": types.SimpleNamespace(data=advanced)}),
    )
    return types.SimpleNamespace(_atm=atm), initial, advanced


def _bind(stub):
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    helper = getattr(CoupledESMDriver, "_atm_tracer", None)
    if helper is not None:
        stub._atm_tracer = types.MethodType(helper, stub)
    return CoupledESMDriver.q_v.fget(stub)


@pytest.mark.parametrize("grid_type, column_lane", [
    ("mpas", False), ("latlon", True), ("cubed_sphere", True)])
def test_column_loop_reads_advanced_moisture(grid_type, column_lane):
    stub, _, advanced = _stub(grid_type, column_lane)
    assert jnp.array_equal(_bind(stub), advanced)
    assert jnp.array_equal(stub._atm_tracer("q_c"), advanced)


def test_other_lanes_read_the_driver_store():
    stub, initial, _ = _stub("latlon", False)
    assert jnp.array_equal(_bind(stub), initial)


def test_species_not_carried_is_none():
    stub, _, _ = _stub("mpas", False)
    _bind(stub)
    assert stub._atm_tracer("q_g") is None
