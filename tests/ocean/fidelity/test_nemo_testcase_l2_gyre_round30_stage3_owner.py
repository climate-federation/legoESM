"""Guards for the round-30 GYRE stage-3 momentum-RHS boundary reader.

What is guarded here is everything that does not need a GYRE model run: that
the pre-``dyn_ldf`` reader returns the writer's own arrays in the campaign's
interior layout, that it FAILS CLOSED on a wrong magic, a wrong header and a
short payload rather than returning a plausible frame, and that both round-29
dumps are registered in the fail-closed time-level registry.

Scoring the model against those frames needs the oracle records and a full
GYRE step, so it lives in the gate, not here; the numbers it produced are in
the round-30 receipt with their SHA-256.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round30_stage3_owner",
    TESTCASES / "nemo_testcase_l2_gyre_round30_stage3_owner.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = gate.DIMS
HEADER = (1, 1, 3, 1, 2, 3, 3, NX, NY, NZ, 64)


def _write(directory: Path, *, magic="NEMO_L2_RKPLD_1", header=HEADER, payload=None):
    # The fixture MUST carry the writer's own basename: the reader looks the
    # record up in the fail-closed time-level registry, which keys on it.
    path = directory / gate.PRE_LDF_RECORD
    if payload is None:
        payload = np.arange(2 * NX * NY * NZ, dtype="<f8")
    with path.open("wb") as handle:
        handle.write(f"{magic:<16}".encode("ascii"))
        handle.write(struct.pack("=11i", *header))
        handle.write(np.asarray(payload, dtype="<f8").tobytes())
    return path


def test_reader_returns_the_campaign_interior_layout(tmp_path):
    """One known element proves the strip-halo-and-transpose is the gate's."""
    values = np.arange(2 * NX * NY * NZ, dtype="<f8")
    frame = gate.read_pre_ldf(_write(tmp_path, payload=values))
    assert frame["u"].shape == frame["v"].shape == (NY - 4, NX - 4, NZ)
    # Fortran (ji, jj, jk) 0-based -> python [jj - 2, ji - 2, jk]; the halo
    # strip is 2 cells on each side of i and j, and the transpose swaps them.
    flat = (0 + 2) + NX * (2 + 2) + NX * NY * 4
    assert frame["u"][2, 0, 4] == values[flat]
    assert frame["v"][2, 0, 4] == values[NX * NY * NZ + flat]


@pytest.mark.parametrize("kwargs", [
    {"magic": "NEMO_L2_ZDFMX_1"},                       # another campaign record
    {"header": (2,) + HEADER[1:]},                      # a future writer version
    {"header": HEADER[:2] + (2,) + HEADER[3:]},         # the stage-2 frame
    {"header": HEADER[:-1] + (32,)},                    # not 64-bit
    {"payload": np.zeros(2 * NX * NY * NZ - 1)},        # short by one value
    {"payload": np.full(2 * NX * NY * NZ, np.nan)},     # non-finite
])
def test_reader_fails_closed(tmp_path, kwargs):
    """A malformed record must get a VERDICT, never a plausible frame."""
    with pytest.raises(gate.require.__globals__["GateError"]):
        gate.read_pre_ldf(_write(tmp_path, **kwargs))


def test_an_unregistered_record_name_is_refused(tmp_path):
    """The registry is fail-closed: an unknown basename never gets a level."""
    stray = tmp_path / "oracle_not_registered_kt00000001.bin"
    stray.write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="no registered NEMO time level"):
        gate.read_pre_ldf(stray)


def test_both_round29_dumps_are_registered():
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    assert time_level_for_dump(gate.PRE_LDF_RECORD) == "now"
    assert time_level_for_dump(gate.ZDF_MATRIX_RECORD) == "now"
