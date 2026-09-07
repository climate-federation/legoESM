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


# Each case names the message it must produce, so six cases cannot all be
# firing on ONE shared guard while five of the six checks are dead.
@pytest.mark.parametrize("kwargs, message", [
    ({"magic": "NEMO_L2_ZDFMX_1"}, "bad magic"),        # another campaign record
    ({"header": (2,) + HEADER[1:]}, "bad header"),      # a future writer version
    ({"header": HEADER[:2] + (2,) + HEADER[3:]}, "bad header"),   # stage-2 frame
    ({"header": HEADER[:-1] + (32,)}, "bad header"),    # not 64-bit
    ({"payload": np.zeros(2 * NX * NY * NZ - 1)}, "bad payload"),  # short by one
    ({"payload": np.full(2 * NX * NY * NZ, np.nan)}, "non-finite payload"),
])
def test_reader_fails_closed(tmp_path, kwargs, message):
    """A malformed record must get a VERDICT, never a plausible frame."""
    with pytest.raises(gate.require.__globals__["GateError"], match=message):
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


@pytest.mark.parametrize("hooks, message", [
    ({"expose_stage3_momentum_rhs": "pre_zdf"}, "must be ''"),
    ({"expose_stage3_momentum_rhs": "post_ldf",
      "expose_stage2_momentum_rhs": True}, "cannot be combined"),
    ({"expose_stage3_momentum_rhs": "pre_ldf",
      "expose_momentum_stage": 2}, "cannot be combined"),
])
def test_stage3_rhs_hook_refuses_a_frame_it_cannot_deliver(hooks, message):
    """Two momentum exposures share the returned u/v, so both is REFUSED.

    Without this the later substitution wins silently and a gate scores the
    wrong frame under the right name -- the exact shape of defeat this
    campaign's gates exist to make impossible.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card("GYRE-zco")
    with pytest.raises(ValueError, match=message):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))


def test_the_production_default_still_constructs():
    """Synthetic-violation companion: the guard must not refuse the default."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card("GYRE-zco")
    assert LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)


@pytest.mark.parametrize("stage", [0, 1, 4, -1])
def test_operator_exposure_stage_is_dispatch_hardened(stage):
    """An unknown operator-exposure stage RAISES; it never silently does nothing.

    Stages 1 and 3-with-lateral-mixing are served by other hooks, so a gate
    that asks for one of them here would otherwise get an empty exposure and
    score the ordinary prognostic velocity under an operator's name.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card("GYRE-zco")
    with pytest.raises(ValueError, match="expose_momentum_operator_stage"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_operator="hpg",
                expose_momentum_operator_stage=stage))


@pytest.mark.parametrize("stage", [2, 3])
def test_operator_exposure_accepts_the_two_walked_stages(stage):
    """Synthetic-violation companion: 2 and 3 must both construct."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card("GYRE-zco")
    assert LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_momentum_operator="vorticity",
            expose_momentum_operator_stage=stage))
