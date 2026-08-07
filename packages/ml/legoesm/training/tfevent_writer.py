"""Dependency-free writer for TensorBoard scalar event files.

Why this exists instead of ``torch.utils.tensorboard`` / ``tensorboardX``
------------------------------------------------------------------------
The calibration campaign runs in the ``diffesm`` environment, which has NEITHER
writer installed (verified 2026-08-07: ``tensorboard``, ``tensorboardX`` and
``torch`` all ``ModuleNotFoundError`` there; only the separate ``pytorch`` env
has them).  Depending on an absent package would mean the production path is an
untested fallback while the tested path never runs — the worst of both.  So the
event file is emitted directly.

That is affordable because the format is small and frozen: a TFRecord stream of
serialised ``Event`` protobufs, and we only ever emit the ``file_version`` header
and ``simple_value`` scalars.  Nothing here re-implements TensorBoard; it
implements ~40 bytes of wire format.

    record := u64 len | u32 masked_crc32c(len bytes) | payload | u32 masked_crc32c(payload)
    Event  := wall_time(1,double) step(2,int64) {file_version(3,string) | summary(5,msg)}
    Summary := value(1,repeated msg{ tag(1,string) simple_value(2,float) })

Field numbers are from ``tensorboard/compat/proto/{event,summary}.proto``.  The
encoding is verified end-to-end in ``tests/unit/test_tfevent_writer.py`` by
reading the produced file back with the REAL TensorBoard implementation
(``event_accumulator``) out of the ``pytorch`` env — a hand-rolled encoder that
has only ever been checked against itself is not evidence.

Whole-file rewrite, not append
------------------------------
:func:`write_scalar_event_file` writes the COMPLETE file every call, atomically
(temp + ``os.replace``).  That keeps the event file a pure function of the JSONL
source of truth — the three renderings in
:mod:`legoesm.training.calibration_tracking` cannot drift apart — and it is safe
for a reader that is already tailing the file, because with deterministic
``wall_time`` values the new file is byte-identical to the old one in its prefix
and only grows.  Do not "optimise" this into an append: an append after a job
was killed mid-write would duplicate or corrupt steps.

This module itself imports no JAX and no NumPy — only the stdlib — so it costs
nothing to pull in and works in any environment. (Reaching it as
``legoesm.training.tfevent_writer`` still triggers ``legoesm.training``'s own
eager JAX imports; import the file directly if you need it somewhere without a
JAX install, as ``tests/unit/test_tfevent_writer.py`` documents.)
"""

from __future__ import annotations

import contextlib
import math
import os
import struct
import sys
import tempfile
from collections.abc import Iterable, Sequence
from typing import NamedTuple

# TensorBoard discovers event files by this substring in the BASENAME
# (``IsTensorFlowEventsFile``), so any name containing it works. A FIXED name
# (rather than the usual host/pid/timestamp suffix) is required by the
# rewrite-in-place contract above: a new name per render would leave the old
# file in the directory and TensorBoard merges every file in a directory, so
# every scalar would appear twice.
EVENT_FILE_NAME = "events.out.tfevents.calibration"

# ``brain.Event:2`` is the version string every writer emits as record 0.
_FILE_VERSION = "brain.Event:2"

_CRC32C_POLY = 0x82F63B78  # Castagnoli, reflected. NOT zlib's CRC-32 (IEEE).


def _build_crc32c_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


_CRC32C_TABLE = _build_crc32c_table()


def crc32c(data: bytes) -> int:
    """CRC-32C (Castagnoli) of ``data``.

    ``zlib.crc32`` is the IEEE polynomial and produces a DIFFERENT value; using
    it here would yield an event file every reader rejects. Checked against the
    standard vector ``crc32c(b"123456789") == 0xE3069283``.
    """
    crc = 0xFFFFFFFF
    for byte in data:
        crc = (crc >> 8) ^ _CRC32C_TABLE[(crc ^ byte) & 0xFF]
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    """The rotate-and-offset mask TFRecord applies to every checksum."""
    crc = crc32c(data)
    rotated = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


# --- minimal protobuf wire encoding ----------------------------------------
# Wire types: 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32.


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"negative varint {value!r}: no field here is signed")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _len_delim(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _double(field: int, value: float) -> bytes:
    return _tag(field, 1) + struct.pack("<d", float(value))


def _float32(field: int, value: float) -> bytes:
    return _tag(field, 5) + struct.pack("<f", float(value))


def _int64(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(int(value))


def _string(field: int, value: str) -> bytes:
    return _len_delim(field, value.encode("utf-8"))


def encode_scalar_event(tag: str, value: float, step: int,
                        wall_time: float) -> bytes:
    """Serialise one ``Event`` carrying a single ``simple_value`` summary."""
    if not tag:
        raise ValueError("scalar tag must be a non-empty string")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step!r}")
    summary_value = _string(1, tag) + _float32(2, value)   # Summary.Value
    summary = _len_delim(1, summary_value)                 # Summary.value[0]
    return _double(1, wall_time) + _int64(2, step) + _len_delim(5, summary)


def encode_file_version_event(wall_time: float = 0.0) -> bytes:
    """Serialise the ``file_version`` header every event file starts with."""
    return _double(1, wall_time) + _int64(2, 0) + _string(3, _FILE_VERSION)


def frame_record(payload: bytes) -> bytes:
    """Wrap a serialised protobuf in the TFRecord length+CRC framing."""
    header = struct.pack("<Q", len(payload))
    return (header
            + struct.pack("<I", masked_crc32c(header))
            + payload
            + struct.pack("<I", masked_crc32c(payload)))


class ScalarPoint(NamedTuple):
    """One (tag, step, value) sample; ``wall_time`` should be deterministic."""

    tag: str
    step: int
    value: float
    wall_time: float = 0.0


def encode_event_file(points: Iterable[ScalarPoint],
                      header_wall_time: float = 0.0) -> bytes:
    """Full event-file bytes for ``points``.

    Two classes of value are SKIPPED rather than written:

    * **non-finite** — a NaN in a TensorBoard scalar stream rescales the whole
      plot to nothing, hiding the finite points that matter. A blown-up member
      is reported through the blow-up count instead (see
      ``calibration_tracking``);
    * **outside float32 range** — ``simple_value`` is a float32, so a finite
      value above ~3.4e38 makes ``struct.pack`` raise ``OverflowError``. That
      is NOT hypothetical for this campaign: a squared normalised residual with
      a small sigma reaches 1e41 from a member that is diverging but has not
      yet gone NaN — precisely the state the dashboard exists to show. Letting
      it raise would take the whole event file, and (before the caller was
      reordered) the page with it. A skipped point is a visible gap; a raised
      exception is a dashboard frozen for the rest of a multi-day campaign.

    Skips are reported on stderr so a systematically-dropped tag is noticed.
    """
    out = bytearray(frame_record(encode_file_version_event(header_wall_time)))
    for point in points:
        value = float(point.value)
        if not math.isfinite(value):
            continue
        if point.step < 0:
            print(f"[tfevent_writer] skipping {point.tag!r} at negative step "
                  f"{point.step}", file=sys.stderr)
            continue
        try:
            record = encode_scalar_event(point.tag, value, point.step,
                                         point.wall_time)
        except (OverflowError, struct.error) as exc:
            print(f"[tfevent_writer] skipping {point.tag!r} at step "
                  f"{point.step}: {value!r} is not representable as a float32 "
                  f"({exc})", file=sys.stderr)
            continue
        out += frame_record(record)
    return bytes(out)


def write_scalar_event_file(directory: str | os.PathLike[str],
                            points: Sequence[ScalarPoint],
                            *, header_wall_time: float = 0.0) -> str:
    """Write the whole event file into ``directory``; returns its path.

    Atomic: the bytes land in a temp file that is ``os.replace``d onto the
    final name, so a kill mid-write can never leave a half-written record for
    TensorBoard to choke on.
    """
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, EVENT_FILE_NAME)
    payload = encode_event_file(points, header_wall_time=header_wall_time)
    # A UNIQUE temp name, not f"{path}.tmp": the CLI's --watch mode is
    # documented to run alongside the live tracker, and two renderers sharing
    # one temp path interleave their writes into a file matching neither, or
    # lose the race on os.replace with FileNotFoundError.
    handle_fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{EVENT_FILE_NAME}.",
                                      suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)     # mkstemp is 0600; the file is meant to be read
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)       # never leave a stray temp behind
        raise
    return path
