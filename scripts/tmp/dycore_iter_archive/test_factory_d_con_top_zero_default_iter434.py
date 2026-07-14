"""FV3_3D iter 434: factory defaults for ``d_con_top_zero_levels``.

iter-431/432/433 added the ``d_con_top_zero_levels`` config
field to both NH and PE configs (default 0 = bit-for-bit
baseline).  iter-434 sets the value to ``2`` in the FV3-
faithful factories (``make_fv3_faithful_nh_config`` and
``make_fv3_faithful_pe_config``).

The value 2 matches FV3 production behaviour under the typical
sponge namelist (``d2_bg_k1=0.16, d2_bg_k2=0.05``): FV3
``dyn_core.F90:790/800/804`` zeros ``d_con_k`` at k=1 (always)
and k=2 (d2_bg_k2>0.01) but NOT k=3 (d2_bg_k2 NOT > 0.05).

Tests
-----

1. ``test_nh_factory_default_d_con_top_zero_levels`` — NH
   factory output has ``d_con_top_zero_levels == 2``.
2. ``test_pe_factory_default_d_con_top_zero_levels`` — PE
   factory output has ``d_con_top_zero_levels == 2``.
3. ``test_nh_factory_override_works`` — explicit override
   ``d_con_top_zero_levels=0`` recovers baseline.
4. ``test_pe_factory_override_works`` — same on PE.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_nh_factory_default_d_con_top_zero_levels():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.d_con_top_zero_levels == 2, (
        f"NH factory default d_con_top_zero_levels expected 2, "
        f"got {cfg.d_con_top_zero_levels}."
    )


def test_pe_factory_default_d_con_top_zero_levels():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.d_con_top_zero_levels == 2, (
        f"PE factory default d_con_top_zero_levels expected 2, "
        f"got {cfg.d_con_top_zero_levels}."
    )


def test_nh_factory_override_works():
    cfg = make_fv3_faithful_nh_config(d_con_top_zero_levels=0)
    assert cfg.d_con_top_zero_levels == 0


def test_pe_factory_override_works():
    cfg = make_fv3_faithful_pe_config(d_con_top_zero_levels=0)
    assert cfg.d_con_top_zero_levels == 0
