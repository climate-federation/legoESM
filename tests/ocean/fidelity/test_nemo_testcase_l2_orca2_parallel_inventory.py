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


def test_require_fails_closed():
    with pytest.raises(probe.InventoryError, match="named predicate"):
        probe.require(False, "named predicate")
