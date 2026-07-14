"""FV3_3D iter 395: docstring-content regression for iter-394
audit clarification.

iter-394 added a source comment in NH ``use_fv3_dynamic_exner``
docstring noting that ``Π_total = Π_ref + π'`` equals FV3
``pkz`` exactly (not a linearization).  iter-395 pins this so
a future refactor doesn't drop the equivalence note.

Tests
-----

1. ``test_nh_dyn_exner_docstring_documents_pkz_equivalence``
"""
from __future__ import annotations

import pytest

from tests.legoesm_paths import legoesm_source_path


NH_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)


def test_nh_dyn_exner_docstring_documents_pkz_equivalence():
    text = NH_SRC.read_text()
    pos = text.find("use_fv3_dynamic_exner: bool = False")
    assert pos >= 0
    next_field = text.find("use_fv3_vector_halo_uv", pos)
    block = text[pos:next_field]
    assert "iter 394" in block, (
        "NH dyn_exner docstring missing iter-394 audit marker."
    )
    assert "FV3 ``pkz``" in block or "FV3 `pkz`" in block, (
        "NH dyn_exner docstring should mention FV3 pkz "
        "equivalence."
    )
    assert "FULL NONLINEAR" in block, (
        "NH dyn_exner docstring should clarify Pi_total is "
        "NONLINEAR (not a linearization)."
    )
