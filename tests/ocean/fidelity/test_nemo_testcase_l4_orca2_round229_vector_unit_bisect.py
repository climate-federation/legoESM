from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as gate,
)


def _row(maximum: float, unequal: int = 1) -> dict[str, object]:
    return {
        "support": 1,
        "unequal": unequal,
        "signed_zero_only": 0,
        "max_abs": maximum,
        "argmax": [0] if unequal else [0],
        "first_nonfinite": None,
    }


def _variant(s_value: float, *, zfv_unequal: int = 0) -> dict:
    result = {
        "T": _row(0.177 if s_value > 1.0 else 0.001),
        "S": _row(s_value),
        "e3t": _row(0.0, 0),
        "tmask": _row(0.0, 0),
        "zFv_recorded_slab": _row(0.0, zfv_unequal),
        "zFv_pivot": _row(0.0, zfv_unequal),
    }
    for name in ("T", "S", "e3t"):
        for suffix in ("pivot", "south1", "south2"):
            result[f"{name}_{suffix}"] = result[name]
    return result


def _scenario(label: str) -> dict:
    variants = {
        "off": _variant(0.001),
        "full": _variant(3.28),
        "drop_slow_v_pair": _variant(3.0),
        "drop_vector_v_mask": _variant(2.9),
        "drop_association": _variant(2.8),
        "drop_v_transport": _variant(0.001),
    }
    return {
        "status": "PASS_R229_SCENARIO",
        "label": label,
        "support": {
            "v_pivot": _row(0.0, 0),
            "t_pivot_right_half": _row(0.0, 0),
            "t_halo_source_check": _row(0.0, 0),
        },
        "recorded_transport_shape": [152, 94, 31],
        "recorded_transport_bottom": _row(0.0, 0),
        "variants": variants,
    }


def _variant_report(label: str, variant: str) -> dict:
    scenario = _scenario(label)
    return {
        "status": "PASS_R229_VARIANT",
        "label": label,
        "variant": variant,
        "worktree": {"commit": "test"},
        "support": scenario["support"],
        "recorded_transport_shape": scenario["recorded_transport_shape"],
        "recorded_transport_bottom": scenario["recorded_transport_bottom"],
        "score": scenario["variants"][variant],
    }
def test_compact_support_uses_distinct_v_source_and_t_pivot_rows() -> None:
    perm = np.array([2, 1, 0, 5, 4, 3])
    fold = SimpleNamespace(perm_T=perm, perm_v=perm)
    card = SimpleNamespace(recipe=SimpleNamespace(
        grid=SimpleNamespace(fold=fold)))
    v = np.zeros((4, 6, 1), dtype=np.float64)
    v[-2, :, 0] = np.arange(6, dtype=np.float64)
    v[-1] = -v[-2, perm]
    t = np.zeros_like(v)
    t[-1, :3, 0] = (1.0, 2.0, 3.0)
    t[-1, 3:, 0] = t[-1, perm[3:], 0]
    result = gate.derive_compact_support(card, {"v": v, "T": t}, plant="none")
    assert result["v_pivot"]["unequal"] == 0
    assert result["t_pivot_right_half"]["unequal"] == 0
    assert result["t_halo_source_check"]["unequal"] == 0
    assert result["mapping"]["v_pivot_source"] == -2
    assert result["mapping"]["t_halo_source"] == -2


@pytest.mark.parametrize(
    "plant", ("v-source-row", "v-sign", "special-longitude"))
def test_compact_support_plants_move_the_exact_v_row(plant: str) -> None:
    perm = np.array([2, 1, 0, 5, 4, 3])
    fold = SimpleNamespace(perm_T=perm, perm_v=perm)
    card = SimpleNamespace(recipe=SimpleNamespace(
        grid=SimpleNamespace(fold=fold)))
    v = np.zeros((4, 6, 1), dtype=np.float64)
    v[-3, :, 0] = 20.0 + np.arange(6)
    v[-2, :, 0] = np.arange(6)
    v[-1] = -v[-2, perm]
    t = np.zeros_like(v)
    result = gate.derive_compact_support(card, {"v": v, "T": t}, plant=plant)
    assert result["v_pivot"]["unequal"] > 0


def test_compact_t_halo_source_plant_moves_the_source_row() -> None:
    perm = np.array([2, 1, 0, 5, 4, 3])
    fold = SimpleNamespace(perm_T=perm, perm_v=perm)
    card = SimpleNamespace(recipe=SimpleNamespace(
        grid=SimpleNamespace(fold=fold)))
    t = np.zeros((4, 6, 1), dtype=np.float64)
    t[-2, :, 0] = np.arange(6)
    t[-1, :, 0] = 10.0 + np.arange(6)
    result = gate.derive_compact_support(
        card, {"v": np.zeros_like(t), "T": t}, plant="t-halo-source")
    assert result["t_halo_source_check"]["unequal"] > 0


def test_classifier_names_transport_as_exposure_not_wrong_statement() -> None:
    result = gate.classify([
        _scenario("independent"), _scenario("given_nemo_entry")])
    assert result["status"] == "HELD_R229_MISSING_TRACER_FOLD_OWNER_CANDIDATE"
    assert all(row["exposure_part"] == "v_transport"
               for row in result["rows"].values())


@pytest.mark.parametrize("plant", ("label-coverage", "leave-one-out"))
def test_classifier_structural_plants_refuse(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify([
            _scenario("independent"), _scenario("given_nemo_entry")],
            plant=plant)


def test_one_bit_transport_violation_changes_the_verdict() -> None:
    result = gate.classify([
        _scenario("independent"), _scenario("given_nemo_entry")],
        plant="operand-bit")
    assert result["status"] == "HELD_R229_VECTOR_TRANSPORT_UNRESOLVED"


def test_part_registry_and_leave_one_out_sets_are_exact() -> None:
    assert gate._enabled("off") == set()
    assert gate._enabled("full") == set(gate.PARTS)
    for part in gate.PARTS:
        assert gate._enabled(f"drop_{part}") == set(gate.PARTS) - {part}


def test_isolated_variant_reports_assemble_in_frozen_order() -> None:
    reports = [_variant_report("independent", variant)
               for variant in gate.VARIANTS]
    assembled = gate.assemble_scenario(reports)
    assert tuple(assembled["variants"]) == gate.VARIANTS
    with pytest.raises(gate.GateError):
        gate.assemble_scenario(list(reversed(reports)))


def test_transport_reader_obeys_self_describing_level_count(tmp_path) -> None:
    path = tmp_path / "transport.bin"
    nx, ny, nz = 3, 4, 5
    values = np.arange(3 * nx * ny * nz, dtype=np.float64)
    with path.open("wb") as handle:
        handle.write(b"NEMO_L1_TRANSP_1")
        handle.write(struct.pack("=8i", 1, 1, 1, 1, nx, ny, nz, 64))
        values.tofile(handle)
    u, v = gate._read_transport_self_describing(path)
    assert u.shape == (ny, nx, nz)
    assert v.shape == (ny, nx, nz)
    assert np.array_equal(
        u, values[:nx * ny * nz].reshape((nx, ny, nz), order="F").transpose(1, 0, 2))
