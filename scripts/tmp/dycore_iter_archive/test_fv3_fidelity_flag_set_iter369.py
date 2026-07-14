"""FV3_3D iter 369: comprehensive presence test for the FV3-
fidelity flag set on PE + NH configs.

Asserts every flag introduced by iter-320 through iter-355 is
present + defaults to False (preserves bit-for-bit baseline).

NH (4 flags):
- use_fv3_d_con_cv               (iter-320)
- use_fv3_vector_halo_uv         (iter-328)
- use_fv3_dynamic_exner          (iter-336/337)
- use_fv3_metric_aware_d_con     (iter-339/344)

PE (2 flags):
- use_fv3_a2b_zeta_corner        (iter-14)
- use_fv3_metric_aware_d_con     (iter-338/344)

Mirror of iter-318/331/343/362 default-asymmetry guards but
in a single consolidated test.

Tests
-----

1. ``test_nh_has_4_fv3_fidelity_flags``
2. ``test_pe_has_2_fv3_fidelity_flags``
3. ``test_all_fv3_flags_default_false``
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)


_NH_FLAGS = (
    "use_fv3_d_con_cv",
    "use_fv3_vector_halo_uv",
    "use_fv3_dynamic_exner",
    "use_fv3_metric_aware_d_con",
    "use_fv3_cross_face_du_proj",   # iter-370
)
_PE_FLAGS = (
    "use_fv3_a2b_zeta_corner",
    "use_fv3_metric_aware_d_con",
    "use_fv3_cross_face_du_proj",   # iter-370
)


def test_nh_has_4_fv3_fidelity_flags():
    cfg = CDGridCompressibleEulerConfig()
    for fld in _NH_FLAGS:
        assert hasattr(cfg, fld), (
            f"NH config missing {fld} — wiring regression."
        )


def test_pe_has_2_fv3_fidelity_flags():
    cfg = CDGridPrimitiveEquationConfig()
    for fld in _PE_FLAGS:
        assert hasattr(cfg, fld), (
            f"PE config missing {fld} — wiring regression."
        )


def test_all_fv3_flags_default_false():
    nh = CDGridCompressibleEulerConfig()
    pe = CDGridPrimitiveEquationConfig()
    for fld in _NH_FLAGS:
        v = getattr(nh, fld)
        assert v is False, (
            f"NH {fld} default = {v}; must be False (preserves "
            f"bit-for-bit baseline)."
        )
    for fld in _PE_FLAGS:
        v = getattr(pe, fld)
        assert v is False, (
            f"PE {fld} default = {v}; must be False."
        )
