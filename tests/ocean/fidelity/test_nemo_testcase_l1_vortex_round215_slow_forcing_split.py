"""Direct test for round 215's forcing-split probe and the new case guards.

Two load-bearing behaviours, each from a defect this round actually had:

* the probe's halo/axis slice.  It reads NEMO's record array, which is
  ``(i, j, k)`` with a two-cell halo, and compares it against the card's
  ``(j, i, k)`` interior.  A transposed or mis-sliced read would still have
  the right SHAPE on this grid -- the seamount mesh is square -- so the
  control uses a NON-SQUARE plane with distinct values, which a transpose
  cannot survive.

* the card/record guard.  The substep walk and the per-term probe used to
  accept a seamount card walked against the flat card's acquisition, which
  nothing downstream can catch (the records carry no case stamp and the two
  decks share a grid size).  Both must now REFUSE.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, TESTCASES / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


split = _load("nemo_testcase_l1_vortex_round215_slow_forcing_split.py",
              "r215_split")
walk = _load("nemo_testcase_l1_vortex_round196_spgts_walk.py", "r215_walk")


def test_interior_strips_the_halo_and_transposes_to_the_card_layout():
    ni, nj, nk, nlev = 7, 5, 4, 3          # deliberately non-square
    halo = split.HALO
    padded = np.arange((ni + 2 * halo) * (nj + 2 * halo) * nk,
                       dtype=np.float64).reshape(
        (ni + 2 * halo, nj + 2 * halo, nk))
    out = split._interior(padded, nlev)
    assert out.shape == (nj, ni, nlev)
    # value identity, not just shape: a transpose or an off-by-one halo
    # would keep the shape on a square plane and change every value here.
    for j in range(nj):
        for i in range(ni):
            for k in range(nlev):
                assert out[j, i, k] == padded[i + halo, j + halo, k]


def test_the_walk_refuses_a_seamount_card_against_the_flat_record():
    with pytest.raises(walk.GateError) as excinfo:
        walk.run(Path("/nonexistent/round196/oracle_spgts_substeps"),
                 case="VORTEX_SMT_VEC-zps", allow_dirty=True)
    assert "must be walked against its own acquisition" in str(excinfo.value)


def test_the_walk_refuses_the_flat_card_against_a_seamount_record():
    with pytest.raises(walk.GateError) as excinfo:
        walk.run(
            Path("/nonexistent/vortex_smt/round5/"
                 "VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts"),
            case="VORTEX_VEC-zco", allow_dirty=True)
    assert "is a seamount acquisition" in str(excinfo.value)


def test_the_guard_is_not_vacuous_on_a_matching_pair():
    """A MATCHING card and root must get past the guard, and fail later.

    Without this the two refusals above would also pass if the guard simply
    refused everything.
    """
    with pytest.raises(Exception) as excinfo:
        walk.run(
            Path("/nonexistent/vortex_smt/round5/"
                 "VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts"),
            case="VORTEX_SMT_VEC-zps", allow_dirty=True)
    assert "must be walked against its own acquisition" not in str(
        excinfo.value)
    assert "is a seamount acquisition" not in str(excinfo.value)
