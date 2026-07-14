"""FV3_3D iter 334: AST regression guard for iter-333 PE duogrid
wiring at the FV3 corner-div damping ke_correction halo.

Mirrors the iter-327 NH AST guard pattern.  Without this guard a
future refactor could silently drop the
``duogrid=_pe_dg_ke`` kwarg, restoring the PE ke_correction
duogrid bypass without test failure (the no-duogrid path stays
bit-for-bit baseline + iter-333's "changes-tendency" test uses
``use_duogrid=True`` with the corner-div toolkit).

Tests
-----

1. ``test_pe_dg_ke_helper_assigned`` — ``_pe_dg_ke = grid.duogrid``
   helper present.
2. ``test_pe_ke_correction_pad_passes_duogrid`` — the
   ``_pad_halo_4d_fn(_ke_correction, ..., duogrid=_pe_dg_ke)``
   call pattern present.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


SRC_PATH = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")


@pytest.fixture(scope="module")
def pe_source():
    return SRC_PATH.read_text()


def test_pe_dg_ke_helper_assigned(pe_source):
    """``_pe_dg_ke = grid.duogrid`` helper appears (iter-333 marker)."""
    pat = r"_pe_dg_ke\s*=\s*grid\.duogrid"
    assert re.search(pat, pe_source), (
        "iter-333 marker '_pe_dg_ke = grid.duogrid' missing from "
        f"{SRC_PATH.name}.  A refactor likely dropped the duogrid "
        f"helper assignment, restoring the PE ke_correction halo "
        f"bypass."
    )


def test_pe_ke_correction_pad_passes_duogrid(pe_source):
    """``_pad_halo_4d_fn(_ke_correction, duogrid=_pe_dg_ke)``
    call pattern present."""
    pat = (
        r"_pad_halo_4d_fn\(\s*_ke_correction\s*,\s*"
        r"duogrid\s*=\s*_pe_dg_ke"
    )
    assert re.search(pat, pe_source, re.DOTALL), (
        "iter-333 PE ke_correction halo does not pass "
        "duogrid=_pe_dg_ke.  A refactor likely dropped the kwarg, "
        "restoring O(dx²) cube-edge bias at the PE FV3 corner-div "
        "damp gradient."
    )
