"""Gates for the GLM authoring-prompt builder.

The prompt is what the authored module is written against, so a defect
here is a defect in every module it produces -- and the failure mode is
silent: a prompt that omits a callee, or carries a signature the code no
longer has, still looks like a prompt.  The module-2 prompt shipped with
a docstring for a routine that module never calls, which is what this
builder exists to stop.

So the gates are: the blocks that must be present are present, the
extracted signature is the one in the source RIGHT NOW, ``--only``
really restricts, and every "cannot find it" path raises instead of
quietly emitting a prompt with a hole in it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = (_REPO / "scripts" / "experiment"
           / "fv3_build_glm_authoring_prompt.py")

# Real files, because the point of the builder is that it reads the
# tree rather than a fixture: a stub would let the extractor drift from
# what the modules actually contain.
_SPEC = "packages/core/legoesm/core/fv3_native_dsw_tail_3d.py"
_SIBLING = "packages/core/legoesm/core/fv3_dsw_phase_3d.py"


def _load():
    spec = importlib.util.spec_from_file_location("_glm_prompt", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(tmp_path, *extra):
    mod = _load()
    out = tmp_path / "prompt.txt"
    rc = mod.main([
        "--spec", str(_REPO / _SPEC),
        "--sibling", str(_REPO / _SIBLING),
        "--target", "packages/core/legoesm/core/fv3_example.py",
        "--out", str(out), *extra,
    ])
    assert rc == 0
    return out.read_text()


def test_script_exists():
    assert _SCRIPT.is_file(), _SCRIPT


def test_prompt_carries_every_block(tmp_path):
    txt = _build(tmp_path)
    for marker in ("FILE: packages/core/legoesm/core/fv3_example.py",
                   "THE NUMPY MODULE IS THE SPECIFICATION",
                   "SIBLING MODULE", "SPECIFICATION ("):
        assert marker in txt, marker


def test_spec_is_included_whole_by_default(tmp_path):
    txt = _build(tmp_path)
    for fn in ("def dsw_tail_phase_3d", "def dgrid_pressure_phase_3d",
               "def dgrid_nh_pressure_phase_3d"):
        assert fn in txt, fn


def test_only_restricts_the_spec_but_keeps_the_header(tmp_path):
    txt = _build(tmp_path, "--only", "dgrid_pressure_phase_3d")
    assert "def dgrid_pressure_phase_3d" in txt
    assert "def dsw_tail_phase_3d" not in txt
    # the deck constant lives above the first def and must survive, or
    # the authored module would restate it (convention C7)
    assert "DUO_TAIL_CFG" in txt


def test_only_raises_on_a_routine_that_is_not_there(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        _build(tmp_path, "--only", "no_such_routine")


def test_callee_signature_is_the_one_in_the_source_now(tmp_path):
    """The whole point: the signature comes from the installed source,
    not from memory.  Compared against the file, so a rename or a new
    keyword-only argument makes this test go red rather than making the
    NEXT authored module wrong."""
    txt = _build(tmp_path, "--callee",
                 "legoesm.core.fv3_pgrad:geopk,one_grad_p")
    assert "fv3_pgrad.py::geopk" in txt
    assert "fv3_pgrad.py::one_grad_p" in txt
    src = (_REPO / "packages/core/legoesm/core/fv3_pgrad.py").read_text()
    first = [ln for ln in src.splitlines() if ln.startswith("def geopk(")]
    assert first and first[0] in txt, first


def test_callee_raises_on_an_unknown_symbol(tmp_path):
    with pytest.raises(SystemExit, match="no top-level def"):
        _build(tmp_path, "--callee", "legoesm.core.fv3_pgrad:not_a_symbol")


def test_callee_raises_on_an_unknown_module(tmp_path):
    with pytest.raises(SystemExit, match="cannot locate module"):
        _build(tmp_path, "--callee", "legoesm.core.fv3_not_a_module:x")


def test_callee_with_no_symbol_is_refused(tmp_path):
    """A bare module name would silently contribute nothing."""
    with pytest.raises(SystemExit, match="names no symbol"):
        _build(tmp_path, "--callee", "legoesm.core.fv3_pgrad:")


def test_missing_spec_file_raises(tmp_path):
    mod = _load()
    with pytest.raises(SystemExit, match="missing"):
        mod.main(["--spec", str(tmp_path / "nope.py"),
                  "--sibling", str(_REPO / _SIBLING),
                  "--target", "x.py", "--out", str(tmp_path / "p.txt")])
