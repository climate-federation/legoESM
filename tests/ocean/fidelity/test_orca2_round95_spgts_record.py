"""Controls for the ORCA2 round-95 self-describing SPG record."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record as record,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round98_coriolis_residual as residual,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round98_een_coeff_acquisition import (
    check_record as r98_record,
)


def write_group(payload: bytearray, name: str, rank: int) -> None:
    shape = (3, 2) if rank == 2 else (4,)
    n1, n2 = shape if rank == 2 else (shape[0], 1)
    payload.extend(name.encode("ascii").ljust(16, b" "))
    payload.extend(struct.pack("=4i", rank, n1, n2, 1))
    payload.extend(np.zeros(n1 * n2, dtype=np.float64).tobytes())


def synthetic_record(path: Path, *, icycle: int = 2) -> None:
    payload = bytearray(record.MAGIC.encode("ascii").ljust(16, b" "))
    payload.extend(struct.pack(
        "=18i", 1, 1, 1, 1, 3, 3, 0, 4, 4, 31, icycle,
        1, 1, 2, 2, 3, 3, 64,
    ))
    for name in record.ENTRY:
        write_group(payload, f"i000_{name}", 1 if name in {"wgtbtp1", "wgtbtp2", "entry_sc"} else 2)
    for substep in range(1, icycle + 1):
        for name in record.SUBSTEP:
            write_group(payload, f"j{substep:03d}_{name}", 1 if name in {"ext_coef", "bck_coef", "sum_coef"} else 2)
    for name in record.EXIT:
        write_group(payload, f"o000_{name}", 2)
    path.write_bytes(payload)


def test_record_derives_frames_and_payloads_from_its_header(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    parsed = record.read_record(path)
    assert parsed["icycle"] == 2
    assert parsed["frames"] == 4
    assert parsed["groups"] == len(record.ENTRY) + 2 * len(record.SUBSTEP) + len(record.EXIT)


@pytest.mark.parametrize(
    "plant", ("header", "field-name", "field-dims", "truncation", "missing-frame"),
)
def test_record_plants_refuse(tmp_path: Path, plant: str) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    with pytest.raises(record.Refusal):
        record.read_record(path, plant)


def test_swapped_rank_plant_is_observable(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    assert record.read_record(path, "swapped-rank")["rank"] == 1


def test_round96_launcher_stages_admitted_deck_before_decision83_patch() -> None:
    launcher = Path(
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round95_spgts_acquisition/run.sh"
    ).read_text(encoding="utf-8")
    stage = 'cp "$SOURCE_RUN/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"'
    pin = "pin \"$SOURCE_NML_SHA\" \"$TARGET_ROOT/EXP00/namelist_cfg\""
    patch = 'patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/EXP00" <"$DECISION83_PATCH"'
    assert launcher.index(stage) < launcher.index(pin) < launcher.index(patch)
    assert "readonly TARGET_CFG=ORCA2_OMIP_L4_R96SPG" in launcher
    assert "STATUS PLANT-FIRED source-deck" in launcher


def test_round97_launcher_pins_producer_content_not_commit_object() -> None:
    launcher = Path(
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round95_spgts_acquisition/run.sh"
    ).read_text(encoding="utf-8")
    verifier = launcher.split("verify_recorded_tools() {", 1)[1].split("\n}\n", 1)[0]
    assert 'manifest=${2:-$TARGET_RUN/toolchain.sha256}' in verifier
    assert '"$PATCH" "$DECISION83_PATCH" "$WRITER" "$GATE" "$PREREG"' in verifier
    assert '"$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90"' in verifier
    assert 'pin "$recorded_digest" "$path"' in verifier
    assert "git cat-file" not in verifier
    assert "git show" not in verifier
    assert "STATUS PLANT-FIRED toolchain" in launcher


def test_round97_walk_extracts_validated_self_describing_payload(tmp_path: Path) -> None:
    path = tmp_path / "oracle_r95_spg_rank0000_kt00000001.bin"
    synthetic_record(path)
    metadata, values = walk._payload(path)
    assert metadata["icycle"] == 2
    assert set(values) == record.required_names(2)
    assert values["i000_ssh_frc"].shape == (3, 2)
    assert values["i000_wgtbtp1"].shape == (4,)


def test_round97_walk_first_nonbit_uses_compiled_order() -> None:
    rows = {name: {"bit_exact": True} for name in walk.SOURCE_ORDER}
    rows["transport_u"] = {"bit_exact": False, "differing_cells": 1}
    rows["after_ssh"] = {"bit_exact": False, "differing_cells": 2}
    assert walk._first_nonbit(rows) == {
        "boundary": "transport_u", "bit_exact": False, "differing_cells": 1,
    }


def test_round97_walk_accepts_owned_only_and_haloed_groups() -> None:
    owned = np.arange(6, dtype=np.float64).reshape(3, 2)
    haloed = np.zeros((7, 6), dtype=np.float64)
    haloed[2:5, 2:4] = owned
    np.testing.assert_array_equal(
        walk._owned_block(owned, 3, 3, 5, 4), owned.T)
    np.testing.assert_array_equal(
        walk._owned_block(haloed, 3, 3, 5, 4), owned.T)
    with pytest.raises(walk.GateError, match="neither owned"):
        walk._owned_block(np.zeros((4, 2)), 3, 3, 5, 4)


def test_round98_strict_application_keeps_written_pair_order() -> None:
    u = np.zeros((3, 4), dtype=np.float64)
    v = np.zeros((4, 3), dtype=np.float64)
    u[1, 2] = 2.0
    v[2, 1] = -3.0
    coefficients = {
        name: np.full((3, 3), index + 1.0, dtype=np.float64)
        for index, name in enumerate(residual.COEFFICIENTS)
    }
    cor_u, cor_v, products = residual.strict_application(u, v, coefficients)
    np.testing.assert_array_equal(
        cor_u,
        (products["u_nw"] + products["u_ne"])
        + (products["u_sw"] + products["u_se"]),
    )
    np.testing.assert_array_equal(
        cor_v,
        -((products["v_sw"] + products["v_se"])
          + (products["v_nw"] + products["v_ne"])),
    )


def test_round98_coefficient_movement_separates_fold_support() -> None:
    base = {name: np.zeros((3, 4), dtype=np.float64)
            for name in residual.COEFFICIENTS}
    candidate = {name: np.array(value, copy=True)
                 for name, value in base.items()}
    candidate["ffu_nw"][-1, 2] = 1.0
    movement = residual.coefficient_movement(base, candidate)
    assert movement["ffu_nw"] == {
        "differing_cells": 1,
        "non_fold_differing_cells": 0,
        "fold_differing_cells": 1,
        "maximum_absolute": 1.0,
    }
    assert all(movement[name]["differing_cells"] == 0
               for name in residual.COEFFICIENTS if name != "ffu_nw")


def test_round98_one_ulp_coefficient_plant_reaches_output() -> None:
    u = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
    v = np.arange(12, dtype=np.float64).reshape(4, 3) + 1.0
    coefficients = {
        name: np.full((3, 3), index + 1.0, dtype=np.float64)
        for index, name in enumerate(residual.COEFFICIENTS)
    }
    planted, location = residual.one_ulp_sensitive_coefficient(
        {"u_mid": [u], "v_mid": [v]}, coefficients,
        {"u": np.ones((3, 3), dtype=bool),
         "v": np.ones((3, 3), dtype=bool)},
    )
    changed = planted[location["name"]] != coefficients[location["name"]]
    assert np.count_nonzero(changed) == 1


def synthetic_r98_coefficient_record(path: Path, rank: int) -> None:
    payload = bytearray(r98_record.MAGIC.encode("ascii").ljust(16, b" "))
    payload.extend(struct.pack(
        "=16i", 1, 1, 3, 3, rank, 94, 152, 31,
        1 + 90 * rank, 1, 3, 3, 92, 150, 64, 8,
    ))
    for index, name in enumerate(r98_record.FIELDS):
        values = np.full((90, 148), index + 1.0, dtype=np.float64)
        payload.extend(name.encode("ascii").ljust(16, b" "))
        payload.extend(struct.pack("=4i", 2, 90, 148, 1))
        payload.extend(values.tobytes(order="F"))
    path.write_bytes(payload)


def test_round98_ranked_coefficient_record_uses_declared_owned_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle_r98_een_coeff_rank0000_kt00000001.bin"
    synthetic_r98_coefficient_record(path, 0)
    record = r98_record.read_record(path)
    assert record["shape"] == [94, 152, 31]
    assert {tuple(group["shape"]) for group in record["fields"].values()} == {
        (90, 148),
    }


@pytest.mark.parametrize(
    "plant", ("header", "field-name", "field-dims", "truncation",
              "missing-field", "zero-payload"),
)
def test_round98_ranked_coefficient_record_plants_refuse(
    tmp_path: Path, plant: str,
) -> None:
    path = tmp_path / "oracle_r98_een_coeff_rank0000_kt00000001.bin"
    synthetic_r98_coefficient_record(path, 0)
    with pytest.raises(r98_record.Refusal):
        r98_record.read_record(path, plant)


def test_round98_ranked_coefficient_launcher_is_content_pinned() -> None:
    launcher = Path(
        "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round98_een_coeff_acquisition/run.sh"
    ).read_text(encoding="utf-8")
    assert "readonly TARGET_CFG=ORCA2_OMIP_L4_R98EENCOEFF" in launcher
    assert "SOURCE_DYNSPG_SHA=" in launcher
    assert "RECORDED_RUN_SHA=" in launcher
    assert "RECORDED_CHECKER_SHA=" in launcher
    assert "verify_recorded_tools" in launcher
    assert "git cat-file" not in launcher
    assert "git show" not in launcher
    assert "STATUS PLANT-FIRED layout" in launcher
    assert "STATUS PLANT-FIRED toolchain" in launcher
    assert "/usr/bin/time" not in launcher
