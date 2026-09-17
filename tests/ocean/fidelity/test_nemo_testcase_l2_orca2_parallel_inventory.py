"""Fail-closed tests for the zero-cost ORCA2 inventory probe."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_orca2_parallel_inventory",
    TESTCASES / "nemo_testcase_l2_orca2_parallel_inventory.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _write_tke(path: Path, *, trailing: bool = False) -> None:
    jpi, jpj, jpk = 1, 1, 1
    payload = probe.TKE_3D_FIELDS + probe.TKE_2D_FIELDS
    header = (
        1,
        2,
        3,
        3,
        jpi,
        jpj,
        jpk,
        64,
        probe.TKE_3D_FIELDS,
        probe.TKE_2D_FIELDS,
        payload,
        1,
        1,
        3,
        1,
    )
    extents = (jpi, jpj, jpk) * probe.TKE_3D_FIELDS + (jpi, jpj, 1) * probe.TKE_2D_FIELDS
    raw = (
        probe.TKE_MAGIC.encode("ascii").ljust(16, b" ")
        + struct.pack(f"={probe.TKE_HEADER_INTS}i", *header)
        + struct.pack(f"={3 * probe.TKE_FIELDS}i", *extents)
        + b"\0" * (8 * payload)
    )
    path.write_bytes(raw + (b"X" if trailing else b""))


def _write_entry_stage(path: Path, *, kt: int, stage: int | None) -> None:
    if stage is None:
        magic = probe.ENTRY_MAGIC
        level = 1 if kt % 2 else 3
        header = (1, kt, level, probe.NX, probe.NY, probe.NZ, probe.NTR, 64)
    else:
        magic = probe.STAGE_MAGIC
        level = 2 if stage == 2 else (3 if kt % 2 else 1)
        header = (
            1,
            kt,
            stage,
            level,
            probe.NX,
            probe.NY,
            probe.NZ,
            probe.NTR,
            64,
        )
    payload = 4 * probe.NX * probe.NY * probe.NZ + probe.NX * probe.NY
    expected_bytes = 16 + 4 * len(header) + 8 * payload
    with path.open("wb") as handle:
        handle.write(magic.encode("ascii").ljust(16, b" "))
        handle.write(struct.pack(f"={len(header)}i", *header))
        handle.seek(expected_bytes - 1)
        handle.write(b"\0")


def test_current_dispatch_keys_are_read_without_importing_model():
    source = """
def build_nemo_testcase_card(case):
    builders = {"GYRE-zco": build_gyre, "LOCK_EXCHANGE-zco": build_lock}
    return builders[case]()
"""
    assert probe._dispatch_keys(source) == ("GYRE-zco", "LOCK_EXCHANGE-zco")


def test_historical_execution_guard_is_read_from_ast():
    source = """
def build_orca2_zps_card():
    return NEMOTestcaseCard("ORCA2-zps", unmeasured_features=("one", "two"))
"""
    assert probe._historical_unmeasured_features(source) == ("one", "two")


def test_tke_reader_reaches_exact_eof_and_rejects_trailing_byte(tmp_path):
    exact = tmp_path / "exact.bin"
    _write_tke(exact)
    row = probe._read_tke_schema(exact)
    assert row["exact_eof"] is True
    trailing = tmp_path / "trailing.bin"
    _write_tke(trailing, trailing=True)
    with pytest.raises(probe.InventoryError, match="trailing bytes"):
        probe._read_tke_schema(trailing)


@pytest.mark.parametrize(("kt", "stage"), [(1, None), (2, 1), (2, 2), (1, 3)])
def test_entry_and_stage_readers_parse_schema_and_exact_eof(tmp_path, kt, stage):
    record = tmp_path / f"entry-stage-{kt}-{stage}.bin"
    _write_entry_stage(record, kt=kt, stage=stage)
    row = probe._read_entry_stage_schema(record, kt, stage)
    assert row["exact_eof"] is True
    assert row["fields"] == ["T", "S", "u", "v", "ssh"]
    with record.open("ab") as handle:
        handle.write(b"X")
    with pytest.raises(probe.InventoryError, match="exact EOF"):
        probe._read_entry_stage_schema(record, kt, stage)


def test_phase2v_field_map_names_every_field_and_only_boundary_is_missing():
    tke = {
        "fields_3d": list(probe.TKE_3D_FIELD_NAMES),
        "fields_2d": list(probe.TKE_2D_FIELD_NAMES),
    }
    zdf = {
        "fields_3d": list(probe.ZDF_3D_FIELDS),
        "fields_2d": list(probe.ZDF_2D_FIELDS),
    }
    contract = probe._tke_contract_map(tke, zdf)
    assert [row["phase2v_field"] for row in contract["phase2v_field_map"]] == list(
        probe.TKE_FIELD_NAMES
    )
    assert contract["missing"] == ["en_after_boundaries"]


def test_missing_boundary_and_provenance_plants_flip_and_refuse():
    predicates = {
        "stream_counts": (101, 101),
        "schemas_valid": True,
        "entry_stage_twins_exact": True,
        "producer_hashes_match": True,
    }
    for plant in ("missing-boundary", "provenance-mismatch"):
        with pytest.raises(probe.InventoryError, match=f"{plant} plant flipped decision"):
            probe._run_decision_plant(plant, **predicates)


def test_phase2v_decision_emits_binding_missing_boundary_verdict():
    decision = probe._reuse_decision(
        stream_counts=(101, 101),
        schemas_valid=True,
        entry_stage_twins_exact=True,
        producer_hashes_match=True,
        available_tke_fields=set(probe.ROUND101_TKE_CONTRACT)
        - {"en_after_boundaries"},
    )
    assert decision["entry_stage_reusable"] is True
    assert decision["acquisition_needed"] is True
    assert decision["verdict"] == (
        "REUSABLE FOR entry/stage; TKE boundary frame MISSING"
    )


def test_boundary_acquisition_is_new_np2_write_only_passivity_gate():
    acquisition = (
        TESTCASES
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "run.sh"
    ).read_text()
    assert "REFERENCE_CFG=ORCA2_ICE_PISCES" in acquisition
    assert "TARGET_CFG=ORCA2_OMIP_L4_P2VBND" in acquisition
    assert "mpirun -np 2" in acquisition
    assert "gfortran -fsyntax-only" in acquisition
    assert "EXPECTED_BASELINE_STREAMS=101" in acquisition
    assert 'cmp -s "$BASELINE_RUN/$name" "$TARGET_RUN/$name"' in acquisition
    assert "write-only passivity failed" in acquisition


def test_require_fails_closed():
    with pytest.raises(probe.InventoryError, match="named predicate"):
        probe.require(False, "named predicate")
