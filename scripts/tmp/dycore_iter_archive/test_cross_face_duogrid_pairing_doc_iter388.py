"""FV3_3D iter 388: regression that iter-385 source comments
document the iter-384 cross_face/duogrid pairing requirement.

iter-384 found ``use_fv3_cross_face_du_proj`` is a no-op when
``use_duogrid=False``.  iter-385 updated PE + NH source
comments to document this.  iter-388 pins the docstring text
so a future refactor doesn't drop the warning.

Tests
-----

1. ``test_pe_source_documents_duogrid_pairing``
2. ``test_nh_source_documents_duogrid_pairing``
"""
from __future__ import annotations

import pytest

from tests.legoesm_paths import legoesm_source_path


PE_SRC = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")
NH_SRC = legoesm_source_path("atmosphere/dynamics/gcm/compressible_euler_cdgrid.py")


def test_pe_source_documents_duogrid_pairing():
    text = PE_SRC.read_text()
    # Find use_fv3_cross_face_du_proj definition.
    pos = text.find("use_fv3_cross_face_du_proj: bool = False")
    assert pos >= 0
    # Find next field def to bound the docstring.
    next_field = text.find("use_fv3_metric_aware_d_con", pos)
    block = text[pos:next_field]
    assert "iter 384" in block, (
        "PE iter-385 doc-update marker missing from "
        "use_fv3_cross_face_du_proj docstring."
    )
    assert "use_duogrid=True" in block, (
        "PE docstring should mention pairing requirement with "
        "use_duogrid=True."
    )


def test_nh_source_documents_duogrid_pairing():
    text = NH_SRC.read_text()
    pos = text.find("use_fv3_cross_face_du_proj: bool = False")
    assert pos >= 0
    next_field = text.find("use_fv3_metric_aware_d_con", pos)
    block = text[pos:next_field]
    assert "iter 384" in block, (
        "NH iter-385 doc-update marker missing."
    )
    assert "use_duogrid=True" in block, (
        "NH docstring should mention pairing requirement."
    )
