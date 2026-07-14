"""FV3_3D iter 437: factory defaults for vorticity + corner-
divergence damping ORDER.

FV3 ``fv_arrays.F90`` production default::

    integer :: nord = 1     ! 0: del-2, 1: del-4, 2: del-6,
                            ! 3: del-8 divergence damping

Previously the factories did not set ``nord_v`` or
``corner_div_damp_nord`` explicitly:
* ``nord_v`` defaulted to 2 (del-6 vorticity damping)
* ``corner_div_damp_nord`` defaulted to 0 (del-2)

iter-437 sets both to 1 in both factories — matches FV3
production ``nord=1`` del-4 across vorticity + corner-div.

Tests
-----

1. ``test_nh_factory_default_nord_v`` — NH factory ``nord_v == 1``.
2. ``test_pe_factory_default_nord_v`` — PE factory ``nord_v == 1``.
3. ``test_nh_factory_default_corner_div_damp_nord`` — NH
   factory ``corner_div_damp_nord == 1``.
4. ``test_pe_factory_default_corner_div_damp_nord`` — PE
   factory ``corner_div_damp_nord == 1``.
5. ``test_nh_factory_override_nord_v`` — explicit override
   recovers other order.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_nh_factory_default_nord_v():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.nord_v == 1, (
        f"NH factory default nord_v expected 1, got {cfg.nord_v}."
    )


def test_pe_factory_default_nord_v():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.nord_v == 1, (
        f"PE factory default nord_v expected 1, got {cfg.nord_v}."
    )


def test_nh_factory_default_corner_div_damp_nord():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.corner_div_damp_nord == 1, (
        f"NH factory default corner_div_damp_nord expected 1, "
        f"got {cfg.corner_div_damp_nord}."
    )


def test_pe_factory_default_corner_div_damp_nord():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.corner_div_damp_nord == 1, (
        f"PE factory default corner_div_damp_nord expected 1, "
        f"got {cfg.corner_div_damp_nord}."
    )


def test_nh_factory_override_nord_v():
    cfg = make_fv3_faithful_nh_config(nord_v=2)
    assert cfg.nord_v == 2
