"""FV3_3D iter 436: factory defaults for ``delt_max``.

Both FV3-faithful factories now expose ``delt_max = 1.0``,
matching FV3 ``fv_arrays.F90`` production default::

    real    :: delt_max = 1.           ! limiter for dissipative heating rate

Previously the factory left ``delt_max`` at its dataclass
default of 0.0 (cap disabled).  Setting it to 1.0 matches the
FV3 production sponge behaviour without requiring the user to
remember the magic value.

Tests
-----

1. ``test_nh_factory_default_delt_max`` — NH factory default
   ``delt_max == 1.0``.
2. ``test_pe_factory_default_delt_max`` — PE factory default
   ``delt_max == 1.0``.
3. ``test_nh_factory_override_delt_max`` — explicit override
   ``delt_max=0.0`` recovers baseline.
4. ``test_pe_factory_override_delt_max`` — same on PE.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_nh_factory_default_delt_max():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.delt_max == 1.0, (
        f"NH factory default delt_max expected 1.0, "
        f"got {cfg.delt_max}."
    )


def test_pe_factory_default_delt_max():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.delt_max == 1.0, (
        f"PE factory default delt_max expected 1.0, "
        f"got {cfg.delt_max}."
    )


def test_nh_factory_override_delt_max():
    cfg = make_fv3_faithful_nh_config(delt_max=0.0)
    assert cfg.delt_max == 0.0


def test_pe_factory_override_delt_max():
    cfg = make_fv3_faithful_pe_config(delt_max=0.0)
    assert cfg.delt_max == 0.0
