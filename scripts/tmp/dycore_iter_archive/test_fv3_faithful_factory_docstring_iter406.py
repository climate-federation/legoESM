"""FV3_3D iter 406: docstring-content regression for iter-392
factories.

Ensures users see clear instruction in factory docstrings about:
- The full flag set enabled
- Pairing requirement with ``use_duogrid=True``
- Override support

Tests
-----

1. ``test_pe_factory_docstring_complete``
2. ``test_nh_factory_docstring_complete``
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_pe_factory_docstring_complete():
    doc = make_fv3_faithful_pe_config.__doc__
    assert doc is not None
    # Mentions every PE flag
    for flag in [
        "use_fv3_a2b_zeta_corner",
        "use_fv3_metric_aware_d_con",
        "use_fv3_cross_face_du_proj",
    ]:
        assert flag in doc, (
            f"PE factory docstring missing {flag} mention."
        )
    # Mentions pairing
    assert "use_duogrid=True" in doc, (
        "PE factory docstring should mention duogrid pairing."
    )
    # Mentions overrides
    assert "overrides" in doc, (
        "PE factory docstring should document overrides kwarg."
    )


def test_nh_factory_docstring_complete():
    doc = make_fv3_faithful_nh_config.__doc__
    assert doc is not None
    for flag in [
        "use_fv3_d_con_cv",
        "use_fv3_vector_halo_uv",
        "use_fv3_dynamic_exner",
        "use_fv3_metric_aware_d_con",
        "use_fv3_cross_face_du_proj",
    ]:
        assert flag in doc, (
            f"NH factory docstring missing {flag} mention."
        )
    assert "use_duogrid=True" in doc
    assert "overrides" in doc
