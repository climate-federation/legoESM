"""Controls for the ORCA2 round-157 external-mode exchange split."""

from __future__ import annotations

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round157_exchange_split_gate as gate,
)


def test_component_registry_is_compiled_order() -> None:
    assert gate.COMPONENTS == ("u_cyclic", "u_fold", "v_cyclic", "v_fold")


def test_valid_component_selection() -> None:
    for component in gate.COMPONENTS:
        gate.validate_selection(component=component)


@pytest.mark.parametrize(
    "plant", ("comparison-bit", "signed-zero", "selector", "overlap"))
def test_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError, match="plant fired|unknown|overlap"):
        gate.run_plant(plant)
