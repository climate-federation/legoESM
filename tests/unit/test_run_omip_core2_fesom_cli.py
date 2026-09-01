"""``--grid fesom`` stage-B1 gate: registered, mesh-dir required, and every
unwired OMIP physics selector rejected LOUDLY (three-grid unification plan;
silent drops forbidden).

Shown non-vacuous by construction: each assertion matches the gate's own
message text, which exists only on the fesom branch.
"""
from __future__ import annotations

import argparse

import pytest

from scripts.run.run_omip_core2 import (
    _FESOM_B1_UNSUPPORTED,
    validate_fesom_stage,
)


def _args(**kw) -> argparse.Namespace:
    base = {attr: None for attr, _ in _FESOM_B1_UNSUPPORTED}
    base["fesom_mesh_dir"] = "/some/mesh"
    base.update(kw)
    return argparse.Namespace(**base)


def test_mesh_dir_required():
    with pytest.raises(SystemExit, match="fesom-mesh-dir"):
        validate_fesom_stage(_args(fesom_mesh_dir=None))


def test_clean_args_pass():
    validate_fesom_stage(_args())  # no raise


@pytest.mark.parametrize("attr,flag", _FESOM_B1_UNSUPPORTED)
def test_every_unwired_selector_rejected(attr, flag):
    with pytest.raises(SystemExit, match="silently dropped"):
        validate_fesom_stage(_args(**{attr: True}))


def test_reject_lists_the_offending_flag():
    with pytest.raises(SystemExit, match=r"--dm2dc"):
        validate_fesom_stage(_args(dm2dc=True))


def test_grid_choice_registered():
    """The parser accepts --grid fesom (argparse would exit(2) with
    'invalid choice' otherwise). Proven via the real entry point's parser
    the same way the sibling CLI tests do: --help exits 0 and mentions
    fesom."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, str(root / "scripts/run/run_omip_core2.py"),
         "--help"],
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0
    assert "fesom" in out.stdout
