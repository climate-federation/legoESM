from pathlib import Path
import struct
import sys


ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
sys.path.insert(0, str(SCRIPTS))
import nemo_si3_stream_header_gate as gate


def _write_reassoc(path: Path, claimed: int = 18) -> None:
    with path.open("wb") as out:
        out.write(b"NEMO_L3REA_001  ")
        out.write(struct.pack("=8i", 1, 3, 1, 1, 3, 3, 64, claimed))
        out.write(struct.pack("=18d", *range(18)))


def test_reassociation_header_is_derived(tmp_path):
    _write_reassoc(tmp_path / "oracle_si3_reassoc_operands.bin")
    result = gate.validate_root(tmp_path)
    assert result["status"] == "VALID"
    assert result["streams"][0]["records"] == 1


def test_planted_bad_header_fails_at_row(tmp_path):
    _write_reassoc(tmp_path / "oracle_si3_reassoc_operands.bin", claimed=43)
    try:
        gate.validate_root(tmp_path)
    except gate.HeaderError as exc:
        assert "claimed 43, derived 18" in str(exc)
    else:
        raise AssertionError("planted invalid payload count did not bind")


def test_round16_counted_headers_are_registered_and_fail_closed(tmp_path):
    tracer = tmp_path / "oracle_rung36_tracer_owner_frames.bin"
    with tracer.open("wb") as out:
        out.write(b"NEMO_L3TR16_001 ")
        out.write(struct.pack("=9i", 1, 1, 1, 2, 3, 3, 1, 18, 64))
        out.write(struct.pack("=18d", *range(18)))
    result = gate.READERS[tracer.name](tracer)
    assert result["records"] == 1

    trazdf = tmp_path / "oracle_rung36_trazdf_owner_frames.bin"
    with trazdf.open("wb") as out:
        out.write(b"NEMO_L3TZ16_001 ")
        out.write(struct.pack("=9i", 1, 1, 1, 2, 3, 3, 3, 15, 64))
        out.write(struct.pack("=14d", *range(14)))
    try:
        gate.READERS[trazdf.name](trazdf)
    except gate.HeaderError as exc:
        assert "wanted 120 bytes, got 112" in str(exc)
    else:
        raise AssertionError("round-16 false payload count did not bind")
