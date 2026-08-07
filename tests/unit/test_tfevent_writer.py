"""Direct tests for the dependency-free TensorBoard event writer.

The encoder is hand-rolled wire format, so it gets two independent checks:

1. an INDEPENDENT decoder written here from the TFRecord/protobuf spec, so the
   encoder is never validated against itself; and
2. when the real ``tensorboard`` package is importable, a round trip through
   ``event_accumulator`` — the only evidence that actually matters, since that
   is the code the user will point at the directory.

(2) is skipped in the ``diffesm`` environment, which deliberately has no
TensorBoard installed; it was run manually against the ``pytorch`` env and is
kept here so it runs wherever the package IS available.
"""

from __future__ import annotations

import math
import struct

import pytest
from legoesm.training.tfevent_writer import (
    EVENT_FILE_NAME,
    ScalarPoint,
    crc32c,
    encode_event_file,
    encode_scalar_event,
    frame_record,
    masked_crc32c,
    write_scalar_event_file,
)

# --- an independent decoder, written from the .proto definitions ------------


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _parse_fields(buf: bytes) -> list[tuple[int, int, object]]:
    """Return ``(field_number, wire_type, payload)`` for every field."""
    out, pos = [], 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
        else:
            raise AssertionError(f"unexpected wire type {wire}")
        out.append((field, wire, value))
    return out


def decode_event_file(data: bytes) -> list[tuple[str, int, float, float]]:
    """``(tag, step, value, wall_time)`` for every scalar, checking every CRC."""
    scalars, pos = [], 0
    while pos < len(data):
        header = data[pos:pos + 8]
        assert len(header) == 8, "truncated length header"
        (length,) = struct.unpack("<Q", header)
        (header_crc,) = struct.unpack_from("<I", data, pos + 8)
        assert header_crc == masked_crc32c(header), "length CRC mismatch"
        payload = data[pos + 12:pos + 12 + length]
        (payload_crc,) = struct.unpack_from("<I", data, pos + 12 + length)
        assert payload_crc == masked_crc32c(payload), "payload CRC mismatch"
        pos += 12 + length + 4

        wall_time, step, summary = 0.0, 0, None
        for field, _wire, value in _parse_fields(payload):
            if field == 1:
                wall_time = value
            elif field == 2:
                step = value
            elif field == 5:
                summary = value
        if summary is None:
            continue                       # the file_version header record
        for field, _wire, value in _parse_fields(summary):
            assert field == 1, "Summary carries only repeated value[]"
            tag, simple = None, None
            for sub_field, _sub_wire, sub_value in _parse_fields(value):
                if sub_field == 1:
                    tag = sub_value.decode("utf-8")
                elif sub_field == 2:
                    simple = sub_value
            scalars.append((tag, step, simple, wall_time))
    return scalars


# --- checksum -------------------------------------------------------------


def test_crc32c_matches_the_standard_vectors():
    """Castagnoli, not the IEEE polynomial zlib implements."""
    assert crc32c(b"123456789") == 0xE3069283
    assert crc32c(b"") == 0x00000000
    assert crc32c(b"a") == 0xC1D04330
    # A CRC-32C that had accidentally been zlib's CRC-32 would give this:
    import zlib
    assert crc32c(b"123456789") != zlib.crc32(b"123456789")


def test_masked_crc_is_the_rotate_plus_offset():
    raw = crc32c(b"hello")
    expected = ((((raw >> 15) | (raw << 17)) & 0xFFFFFFFF) + 0xA282EAD8)
    assert masked_crc32c(b"hello") == expected & 0xFFFFFFFF


def test_frame_record_layout():
    payload = b"abcd"
    framed = frame_record(payload)
    assert len(framed) == 8 + 4 + len(payload) + 4
    assert struct.unpack("<Q", framed[:8])[0] == len(payload)
    assert framed[12:16] == payload


# --- encoding round trip through the independent decoder -------------------


def test_round_trip_through_an_independent_decoder():
    points = [
        ScalarPoint("loss/ensemble_mean", 0, 12.5, 100.0),
        ScalarPoint("loss/ensemble_mean", 1, 8.25, 200.0),
        ScalarPoint("parameters/a.b.c/bound_fraction", 1, 0.9700000286102295,
                    200.0),
    ]
    decoded = decode_event_file(encode_event_file(points))
    assert decoded == [(p.tag, p.step, p.value, p.wall_time) for p in points]


def test_float32_is_the_declared_precision():
    """simple_value is a float32; the test states the tolerance rather than
    comparing exactly, because 0.1 is not representable."""
    (decoded,) = decode_event_file(
        encode_event_file([ScalarPoint("t", 0, 0.1, 0.0)]))
    assert decoded[2] == pytest.approx(0.1, rel=1e-7)


def test_non_finite_values_are_skipped_not_written():
    """A NaN would rescale a TensorBoard plot to nothing and hide the rest."""
    points = [ScalarPoint("t", 0, float("nan"), 0.0),
              ScalarPoint("t", 1, float("inf"), 0.0),
              ScalarPoint("t", 2, 3.0, 0.0)]
    decoded = decode_event_file(encode_event_file(points))
    assert [(d[1], d[2]) for d in decoded] == [(2, 3.0)]


def test_file_version_header_is_present_and_first():
    data = encode_event_file([])
    (length,) = struct.unpack("<Q", data[:8])
    payload = data[12:12 + length]
    strings = [v for f, _w, v in _parse_fields(payload) if f == 3]
    assert strings == [b"brain.Event:2"]


def test_rejects_a_negative_step_and_an_empty_tag():
    with pytest.raises(ValueError, match="step"):
        encode_scalar_event("t", 1.0, -1, 0.0)
    with pytest.raises(ValueError, match="tag"):
        encode_scalar_event("", 1.0, 0, 0.0)


# --- on-disk behaviour ------------------------------------------------------


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = write_scalar_event_file(
        tmp_path / "tb", [ScalarPoint("t", 0, 1.0, 0.0)])
    assert path.endswith(EVENT_FILE_NAME)
    assert not (tmp_path / "tb" / f"{EVENT_FILE_NAME}.tmp").exists()
    assert list((tmp_path / "tb").iterdir()) == [tmp_path / "tb" /
                                                 EVENT_FILE_NAME]


def test_rewrite_extends_the_file_without_duplicating_or_rewriting_history():
    """The whole-file rewrite contract.

    A second render must produce a file whose PREFIX is byte-identical to the
    first — otherwise a TensorBoard already reading the file would see history
    change under it — and must not repeat the steps it already contained.
    """
    first = encode_event_file([ScalarPoint("t", 0, 1.0, 10.0)])
    second = encode_event_file([ScalarPoint("t", 0, 1.0, 10.0),
                                ScalarPoint("t", 1, 2.0, 20.0)])
    assert second.startswith(first), "a rewrite must only append"
    assert [(d[1], d[2]) for d in decode_event_file(second)] == [(0, 1.0),
                                                                (1, 2.0)]


def test_a_single_fixed_filename_so_scalars_are_never_doubled(tmp_path):
    """TensorBoard merges every event file in a directory: two files would show
    each scalar twice."""
    write_scalar_event_file(tmp_path, [ScalarPoint("t", 0, 1.0, 0.0)])
    write_scalar_event_file(tmp_path, [ScalarPoint("t", 0, 1.0, 0.0),
                                       ScalarPoint("t", 1, 2.0, 0.0)])
    events = [p for p in tmp_path.iterdir() if "tfevents" in p.name]
    assert len(events) == 1, f"expected one event file, found {events}"


def test_the_name_is_discoverable_by_tensorboard():
    """``IsTensorFlowEventsFile`` matches on this substring in the basename."""
    assert "tfevents" in EVENT_FILE_NAME


@pytest.mark.parametrize("value", [0.0, -1.5, 1e30, -1e-30])
def test_values_survive_the_round_trip(value):
    (decoded,) = decode_event_file(
        encode_event_file([ScalarPoint("t", 0, value, 0.0)]))
    assert decoded[2] == pytest.approx(value, rel=1e-6, abs=1e-35)


# --- the real thing, where it is installed ---------------------------------


def test_real_tensorboard_reads_the_file(tmp_path):
    """The only check that proves the format; skipped where TB is absent."""
    ea = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator",
        reason="tensorboard is not installed in this environment")
    write_scalar_event_file(tmp_path, [
        ScalarPoint("loss/ensemble_mean", 0, 12.5, 100.0),
        ScalarPoint("loss/ensemble_mean", 1, 8.25, 200.0),
        ScalarPoint("loss/best_member", 0, 9.5, 100.0),
    ])
    accumulator = ea.EventAccumulator(str(tmp_path))
    accumulator.Reload()
    assert set(accumulator.Tags()["scalars"]) == {"loss/ensemble_mean",
                                                  "loss/best_member"}
    series = accumulator.Scalars("loss/ensemble_mean")
    assert [s.step for s in series] == [0, 1]
    assert [s.value for s in series] == pytest.approx([12.5, 8.25])
    assert [s.wall_time for s in series] == pytest.approx([100.0, 200.0])


def test_encode_event_file_handles_an_empty_stream():
    """A campaign with nothing logged still writes a valid, readable file."""
    assert decode_event_file(encode_event_file([])) == []


def test_math_helpers_do_not_leak_nan_into_the_stream():
    """Guard against a future edit dropping the finiteness filter."""
    data = encode_event_file([ScalarPoint("t", 0, math.nan, 0.0)])
    assert decode_event_file(data) == []


def test_a_finite_value_outside_float32_range_is_skipped_not_raised(capsys):
    """``simple_value`` is a float32; ~1e41 is reachable from a squared
    normalised residual with a small sigma, i.e. from a member that is
    diverging but has not yet gone NaN. Raising there used to take the whole
    event file, and the dashboard page with it."""
    points = [ScalarPoint("loss/ensemble_mean", 0, 10.0, 0.0),
              ScalarPoint("loss/ensemble_mean", 1, 4.2e41, 0.0),
              ScalarPoint("loss/ensemble_mean", 2, -1e300, 0.0),
              ScalarPoint("loss/ensemble_mean", 3, 7.0, 0.0)]
    decoded = decode_event_file(encode_event_file(points))
    assert [(d[1], d[2]) for d in decoded] == [(0, 10.0), (3, 7.0)]
    assert "not representable as a float32" in capsys.readouterr().err


def test_a_negative_step_is_skipped_rather_than_raising(capsys):
    decoded = decode_event_file(encode_event_file(
        [ScalarPoint("t", -1, 1.0, 0.0), ScalarPoint("t", 0, 2.0, 0.0)]))
    assert [(d[1], d[2]) for d in decoded] == [(0, 2.0)]
    assert "negative step" in capsys.readouterr().err


def test_concurrent_writers_never_collide_on_a_shared_temp_name(tmp_path):
    """--watch may render the same directory alongside the live tracker."""
    import threading

    points = [ScalarPoint("t", i, float(i), 0.0) for i in range(200)]
    errors: list[BaseException] = []

    def write():
        try:
            for _ in range(10):
                write_scalar_event_file(tmp_path, points)
        except BaseException as exc:               # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, f"concurrent writes raised: {errors}"
    assert decode_event_file((tmp_path / EVENT_FILE_NAME).read_bytes()) == [
        (p.tag, p.step, p.value, p.wall_time) for p in points]
    assert [p.name for p in tmp_path.iterdir()] == [EVENT_FILE_NAME]
