from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l2_gyre_round54_tke_operands.py")
SPEC = importlib.util.spec_from_file_location("round54_tke_operands", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(path: Path) -> None:
    parts = [MODULE.MAGIC, struct.pack("=13i", 2, 2, 3, 3, 32, 22, 31, 30,
                                      1, 32, 1, 22, 64)]
    shape = (32, 22, 31)
    for name in MODULE.FIELDS:
        parts.append(name.ljust(16).encode("ascii"))
        if name in {"rn_Dt", "rn_ediff", "rn_ediss", "rn_ebb", "rn_emin",
                    "rn_emin0", "rmxl_min", "rn_mxl0", "rn_bshear", "rn_lc",
                    "nn_pdl", "nn_mxl", "ln_mxl0", "nn_etau", "nn_htau",
                    "nn_eice", "ln_lc"}:
            values = {
                "rn_Dt": 14400.0, "rn_ediff": 0.1, "rn_ediss": 0.7,
                "rn_ebb": 67.83, "rn_emin": 1.0e-6, "rn_emin0": 1.0e-4,
                "rmxl_min": 0.01, "rn_mxl0": 0.01, "rn_bshear": 1.0e-20,
                "rn_lc": 0.15, "nn_pdl": 1.0, "nn_mxl": 3.0,
                "ln_mxl0": 1.0, "nn_etau": 0.0, "nn_htau": 1.0,
                "nn_eice": 0.0, "ln_lc": 1.0,
            }
            value = values[name]
            parts.extend((struct.pack("=4i", 0, 1, 1, 1),
                          struct.pack("=d", value)))
        elif name == "taum_entry":
            parts.extend((struct.pack("=4i", 2, 32, 22, 1),
                          np.zeros((32, 22), dtype="=f8").tobytes(order="F")))
        else:
            parts.append(struct.pack("=4i", 3, *shape))
            value = np.zeros(shape, dtype="=f8")
            if name in {"tmask", "wmask"}:
                value.fill(1.0)
            elif name == "rn2":
                value.fill(1.0)
            elif name == "pdlr":
                value[:, :, 1:30] = 1.0
            elif name == "matrix_diag":
                value[:, :, :30] = 1.0
                value[:, :, 0] = 1.0e4
            elif name == "matrix_lower":
                value[:, :, 0] = 1.0
            elif name == "rhs_pre_sweep":
                value[:, :, 1:30] = 1.0e-6
                value[:, :, 1] = 2.0e-6
            elif name == "en_post_sweep":
                value[:, :, 0] = 1.0e-4
                value[:, :, 1:30] = 1.0e-6
                value[:, :, 1] = 2.0e-6
            elif name in {"mxl_momentum", "mxl_dissipation"}:
                value[:, :, :30] = 0.01
            elif name == "avm_floor":
                value[:, :, :30] = 1.2e-4
            elif name == "avt_floor":
                value[:, :, :30] = 1.2e-5
            elif name in {"avm_closure", "avm_pre_evd"}:
                value[:, :, :30] = 1.2e-4
            elif name in {"avt_closure", "avt_pre_evd"}:
                value[:, :, :30] = 1.2e-5
            elif name == "dissl_output":
                value[:, :, 0] = 1.0
                value[:, :, 1:30] = 0.1
                value[:, :, 1] = np.sqrt(2.0e-6) / 0.01
            parts.append(value.tobytes(order="F"))
    path.write_bytes(b"".join(parts))


def test_reader_accepts_complete_schema(tmp_path):
    path = tmp_path / "record.bin"
    _record(path)
    result = MODULE.read_record(path)
    assert result["header"]["kt"] == 2
    assert tuple(result["arrays"]["avt_pre_evd"].shape) == (32, 22, 31)
    assert tuple(result["arrays"]["taum_entry"].shape) == (32, 22)
    assert result["calibration"] == {
        "wet_solve_cells": 32 * 22 * 29,
        "wet_mixing_cells": 32 * 22 * 30,
        "en_unequal": 0,
        "mxl_momentum_unequal": 0,
        "mxl_dissipation_unequal": 0,
        "wet_closure_cells": 32 * 22 * 30,
        "wet_prandtl_cells": 32 * 22 * 29,
        "prandtl_unequal": 0,
        "avm_unequal": 0,
        "avt_unequal": 0,
        "dissl_unequal": 0,
    }


@pytest.mark.parametrize(
    "plant", ["header", "slot", "truncation", "nan", "config", "copy",
              "shape", "sweep", "prandtl"])
def test_reader_plants_fail(tmp_path, plant):
    path = tmp_path / "record.bin"
    _record(path)
    with pytest.raises(MODULE.GateError):
        MODULE.read_record(path, plant=plant)


def test_stamp_plant_exits_nonzero_before_report_emission(tmp_path):
    path = tmp_path / "record.bin"
    producer = tmp_path / "producer_commit.txt"
    _record(path)
    producer.write_text("1" * 40 + "\n")
    assert MODULE.main([
        "--record", str(path),
        "--expect-commit", "1" * 40,
        "--producer-commit", str(producer),
        "--plant", "stamp",
    ]) == 1


def test_operand_score_is_bit_exact_and_masked():
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    actual = expected.copy()
    actual[0] = np.nextafter(actual[0], np.inf)
    actual[2] = 99.0
    score = MODULE._operand_score(
        actual, expected, np.array([True, True, False]))
    assert score["compared_cells"] == 2
    assert score["unequal"] == 1
    assert score["max_abs"] == np.spacing(1.0)
    assert not score["exact"]


def test_kh_entry_ulp_plant_selects_a_consumed_output_response():
    energy = np.linspace(1.0e-6, 1.0e-2, 256, dtype=np.float64)
    baseline = np.linspace(1.0, 2.0, 256, dtype=np.float64)
    bumped = baseline.copy()
    bumped[17] = np.nextafter(bumped[17], np.inf)
    bumped[200] = np.nextafter(bumped[200], np.inf)
    mask = np.ones_like(energy, dtype=bool)
    planted, index = MODULE._plant_entry_ulp_at_changed_output(
        energy, baseline, bumped, mask)

    assert np.count_nonzero(planted.view(np.uint64)
                            != energy.view(np.uint64)) == 1
    assert planted[index] == np.nextafter(energy[index], np.inf)
    assert index == (200,)


def test_kh_execution_labels_do_not_call_an_isolated_jit_production():
    assert MODULE.ISOLATED_EAGER_LABEL == "isolated-closure eager"
    assert MODULE.ISOLATED_JIT_LABEL == "isolated-closure JIT"
    assert MODULE.PRODUCTION_STEP_LABEL == (
        "recorded-entry production step (_step_jitted)")
    assert MODULE.PRODUCTION_INJECTION_LABEL == (
        "recorded-entry production step with NEMO en_post_sweep injection "
        "(_step_jitted)")

    source = SCRIPT.read_text()
    assert '"production_jit_call_site"' not in source
    assert '"production-JIT source-order K_H rows' not in source
    assert "production_tke_only=True" in source
    assert 'response_rows[PRODUCTION_STEP_LABEL]' in source
    assert '"physical_range_sanity": physical_range_sanity' in source
    assert 'production_result["forcing_kt"]' in source
    assert 'production_tke_taum=jnp.asarray(yx("taum_entry"))' in source
    assert "production_tke_post_sweep=post_sweep" in source
    assert '"causal_attribution": causal_attribution' in source
    assert "PRODUCTION_INJECTION_LABEL: production_injection_row" in source
    assert 'production_plant_field = "tke_avm"' in source


def test_positive_ulp_distance_counts_binary64_steps():
    expected = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
    actual = expected.copy()
    actual[1] = np.nextafter(np.nextafter(actual[1], np.inf), np.inf)
    mask = np.asarray([True, True, False])

    assert MODULE._positive_ulp_distance(actual, expected, mask) == 2
