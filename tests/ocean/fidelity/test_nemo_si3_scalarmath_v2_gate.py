"""Controls for the C1D scalar-math oracle V2 provenance gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_si3_scalarmath_v2_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_scalarmath_v2_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_bulk_byte_offset_maps_to_registered_field() -> None:
    assert gate._bulk_location(36) == {
        "step": 1,
        "frame": "POST_BLK_ICE_1",
        "field": "u_air",
        "byte_in_binary64": 0,
    }


def test_first_byte_plant_is_non_destructive(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    payload = bytes(range(128))
    left.write_bytes(payload)
    right.write_bytes(payload)
    assert gate._first_differing_byte(left, right) is None
    assert gate._first_differing_byte(left, right, plant=True) == 40
    assert right.read_bytes() == payload


def test_column_card_defaults_to_scalar_math_v2() -> None:
    from legoesm.ice.c1d_omip_l3 import (
        ORACLE_V2_ROOT,
        ORACLE_VERSION,
        build_c1d_omip_l3_card,
    )

    card = build_c1d_omip_l3_card()
    assert ORACLE_VERSION == "V2_SCALAR_MATH"
    assert card.oracle_root == ORACLE_V2_ROOT == gate.V2_A_ROOT


def test_binary_zgv_control_uses_retained_vectorized_executable() -> None:
    symbols = gate._nm_zgv(gate.V1_ROOT / "nemo.exe")
    assert symbols
    assert all(symbol.startswith("_ZGV") for symbol in symbols)


@pytest.mark.skipif(
    not (gate.V2_A_ROOT / "oracle_si3_exchange_frames.bin").exists(),
    reason="retained scalar-math V2 oracle is not mounted",
)
def test_full_v2_gate_and_plants() -> None:
    result = gate.evaluate()
    assert result["status"] == "REPRODUCIBLE"
    assert result["different_v1_v2_streams"] == 0
    assert result["reproducible_streams"] == 8
    assert all(
        result["comparisons"][name]["v1_v2"] == "IDENTICAL"
        for name in gate.STREAMS
    )
    for plant in (
        "stream_bit", "inventory", "binary_zgv", "source_drift", "v1_as_v2",
        "run_deck", "forcing",
    ):
        with pytest.raises(gate.GateError):
            gate.evaluate(plant=plant)
