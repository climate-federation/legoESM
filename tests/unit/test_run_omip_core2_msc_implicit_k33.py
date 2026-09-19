"""--gm-msc-stabilize must bring implicit_K33 with it, or it cannot be used.

gm_redi_tracer_tendency_latlon RAISES when msc_stabilize is on and
implicit_K33 is off: the capped akz has to be applied by the implicit
vertical solve, or its contribution to the a33 diagonal is silently dropped.

implicit_K33 is a GMRediConfig field with no CLI flag and a False default, so
until this pairing was added, --gm-msc-stabilize was UNUSABLE on every lane --
any run selecting it died at the first step. The gap 9 composition check did
not catch it because it built and validated the config without taking a step,
and this guard lives in the tendency, not the validator. Running the actual
production card is what found it.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def test_the_pairing_is_in_build_tripole():
    """Read the source: build_tripole needs a mesh, so constructing the real
    config here is not possible, but the pairing must be present and must be
    INSIDE the msc_stabilize branch rather than unconditional."""
    src = RUNNER.read_text()
    assert '_gm_kw["implicit_K33"] = True' in src
    i_msc = src.index('_gm_kw["msc_stabilize"] = bool(gm_msc_stabilize)')
    i_k33 = src.index('_gm_kw["implicit_K33"] = True')
    assert i_k33 > i_msc, "implicit_K33 must be set after msc_stabilize"
    between = src[i_msc:i_k33]
    assert "if gm_msc_stabilize:" in between, (
        "implicit_K33 must be conditional on msc_stabilize being ON, not "
        "forced for every run")


def test_the_tendency_still_enforces_the_requirement():
    """The pairing above is only safe because the model still refuses the
    illegal combination. If that guard were deleted the akz term would be
    silently dropped instead of raising, so pin it."""
    gm = (pathlib.Path(__file__).resolve().parents[2] / "packages" / "ocean"
          / "legoesm" / "ocean" / "physics" / "lateral_mixing"
          / "gm_redi_latlon_cgrid.py").read_text()
    # The message is split across source lines, so match a phrase that is
    # unambiguous and contiguous rather than the whole sentence.
    assert "the capped akz must be applied by the" in gm
    assert "msc_stabilize" in gm


def test_implicit_K33_defaults_off():
    """If this ever defaults True the pairing becomes redundant -- and, more
    importantly, every existing run's numerics would have changed."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    assert GMRediConfig().implicit_K33 is False


def test_msc_stabilize_defaults_off_so_nothing_moved():
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    assert GMRediConfig().msc_stabilize is False


def test_build_tripole_still_takes_the_flag():
    for node in ast.walk(ast.parse(RUNNER.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "build_tripole":
            names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert "gm_msc_stabilize" in names
            return
    pytest.fail("build_tripole not found")
