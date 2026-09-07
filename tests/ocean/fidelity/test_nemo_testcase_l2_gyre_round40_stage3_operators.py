"""Guards for the round-40 stage-3 operator reader.

Everything here runs without a GYRE model step and without the oracle roots:
that the ``NEMO_L2_RKTS3_1`` reader returns the writer's own arrays in the
campaign's interior layout, that it FAILS CLOSED on a wrong magic, a wrong
header, a truncated payload, a non-finite payload and a missing operator
frame rather than returning a plausible one, and that the gate's operator
list matches the exposure hook's.

Scoring the model against those frames needs the record and a full GYRE step,
so it lives in the gate; its numbers are in the round-40 receipt.
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
    "nemo_testcase_l2_gyre_round40_stage3_operators",
    TESTCASES / "nemo_testcase_l2_gyre_round40_stage3_operators.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = gate.DIMS
HEADER = (1, 1, 3, 1, 2, 3, 3, NX, NY, NZ, 30, 3, 34, 3, 24, 64)
FRAMES = ("before_u", "before_v", "after_hpg_u", "after_hpg_v",
          "after_vor_u", "after_vor_v", "after_adv_u", "after_adv_v")


def _write(directory: Path, *, magic="NEMO_L2_RKTS3_1", header=HEADER,
           frames=FRAMES, truncate=False, nonfinite=False) -> Path:
    path = directory / "oracle_rkstage3_terms_kt00000001.bin"
    n3 = NX * NY * NZ
    with path.open("wb") as handle:
        handle.write(magic.ljust(16).encode("ascii"))
        handle.write(struct.pack("=16i", *header))
        for index, name in enumerate(frames):
            block = np.arange(n3, dtype=np.float64) + index
            if nonfinite and index == 0:
                block[0] = np.nan
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=4i", 3, NX, NY, NZ))
            payload = block.tobytes()
            handle.write(payload[:-8] if (truncate and index == 0) else payload)
            if truncate and index == 0:
                return path
        handle.write("r3f".ljust(16).encode("ascii"))
        handle.write(struct.pack("=4i", 2, NX, NY, 1))
        handle.write(np.zeros(NX * NY, dtype=np.float64).tobytes())
        handle.write("rDt".ljust(16).encode("ascii"))
        handle.write(struct.pack("=4i", 0, 1, 1, 1))
        handle.write(np.asarray([14400.0], dtype=np.float64).tobytes())
    return path


def test_reader_returns_the_campaign_interior_layout(tmp_path):
    record = gate.read_stage3_terms(_write(tmp_path))
    arrays = record["arrays"]
    assert set(FRAMES) <= set(arrays)
    for name in FRAMES:
        assert arrays[name].shape == (NY - 4, NX - 4, NZ)
    assert arrays["r3f"].shape == (NY - 4, NX - 4)
    assert arrays["rDt"] == 14400.0
    # the writer's own value at native (ji, jj, jk) = (2, 2, 0) is the first
    # interior point, and the reader must put it at (jj, ji) = (0, 0)
    expected = float(np.arange(NX * NY * NZ, dtype=np.float64)
                     .reshape((NX, NY, NZ), order="F")[2, 2, 0])
    assert arrays["before_u"][0, 0, 0] == expected


@pytest.mark.parametrize("kwargs, message", [
    ({"magic": "NEMO_L2_RKTS3_9"}, "bad magic"),
    ({"header": (1, 1, 2) + HEADER[3:]}, "bad header"),
    ({"header": HEADER[:7] + (99, NY, NZ) + HEADER[10:]}, "bad header"),
    ({"truncate": True}, "truncated array"),
    ({"nonfinite": True}, "non-finite payload"),
    ({"frames": ("before_u", "before_v")}, "operator frames are missing"),
])
def test_reader_fails_closed(tmp_path, kwargs, message):
    with pytest.raises(Exception, match=message):
        gate.read_stage3_terms(_write(tmp_path, **kwargs))


def test_operator_list_matches_the_models_exposure_hook():
    """A renamed bucket must not leave the gate scoring nothing."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card("GYRE-zco")
    for name in gate.OPERATORS:
        assert LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_operator=name,
                expose_momentum_operator_stage=3))
    with pytest.raises(ValueError, match="expose_momentum_operator"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_operator="not_an_operator",
                expose_momentum_operator_stage=3))
