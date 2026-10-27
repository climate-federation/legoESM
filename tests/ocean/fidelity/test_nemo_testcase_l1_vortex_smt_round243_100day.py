"""Fail-closed controls for the deferred SMT-2 100-day comparison."""
from __future__ import annotations

import sys
from pathlib import Path


TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_round210_100day_comparison as comparison  # noqa: E402


def test_smt2_uses_round243_oracle_and_latest_certified_ladder():
    case, oracle = comparison.CARDS["smt2"]
    assert case == "VORTEX_SMT2_VEC-zps"
    assert oracle == Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/"
        "oracle_vortex_smt2/day100")
    assert comparison.CERTIFIED_LADDER["smt2"] == Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/inert/"
        "VORTEX_SMT2_VEC-zps.json")


def test_acquisition_wrapper_reuses_the_registered_common_arm():
    wrapper = (TOOLS /
        "nemo_testcase_l1_vortex_smt_round31_smt2_100day" / "run.sh")
    text = wrapper.read_text()
    assert "--variant smt2vec100d" in text
    assert "round243/oracle_vortex_smt2/day100" in text
    assert "/usr/bin/time" not in text
    assert "mpirun" not in text
    assert "status --porcelain" in text
    assert "REFUSE:" in text


def test_movie_renderer_accepts_smt2_from_shared_registry():
    import nemo_testcase_l1_vortex_round210_movie as movie

    assert movie.CARDS is comparison.CARDS
    assert "smt2" in movie.CARDS
