"""Focused controls for the round-336 producer-value audit."""

from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_producer_value_audit as audit,
)


def test_record_table_covers_every_requested_admitted_record() -> None:
    assert [record.label for record in audit.RECORDS] == [
        "round 328 dyn walk",
        "round 328 ZDF terms",
        "round 331 TKE",
        "round 332 source order",
        "round 332 raw V",
        "round 333 tracer stage",
        "round 333 internal LDF",
        "round 336 tra_zdf",
    ]
    assert all(record.verdict in {
        "values production-identical CONFIRMED",
        "values production-identical PLAUSIBLE",
        "values differ",
    } for record in audit.RECORDS)


def test_bundle_digest_is_ordered_and_mutation_sensitive(tmp_path: Path) -> None:
    (tmp_path / "z").write_bytes(b"last")
    (tmp_path / "a").write_bytes(b"first")
    first = audit.bundle(tmp_path, "*")
    second = audit.bundle(tmp_path, "*")
    assert first == second
    (tmp_path / "a").write_bytes(b"changed")
    assert audit.bundle(tmp_path, "*")["sha256"] != first["sha256"]


def test_opcode_classifiers_distinguish_arithmetic_and_fma() -> None:
    assert audit.FP_RE.match("addsd")
    assert audit.FP_RE.match("fmul")
    assert not audit.FP_RE.match("movsd")
    assert audit.FMA_RE.match("vfmadd132sd")
    assert not audit.FMA_RE.match("mulsd")


def test_fast_math_family_is_exhaustively_named() -> None:
    assert "-ffast-math" in audit.FORBIDDEN_FLAGS
    assert "-Ofast" in audit.FORBIDDEN_FLAGS
    assert "-mfma" in audit.FORBIDDEN_FLAGS
    assert "-ffp-contract=fast" in audit.FORBIDDEN_FLAGS
