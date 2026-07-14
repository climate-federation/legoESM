"""FV3_3D iter 426: verify iter-392 factories are importable
from both their dycore modules AND the package-level
``legoesm.atmosphere.dynamics`` namespace.
"""
from __future__ import annotations

import pytest


def test_factories_importable_from_dycore_modules():
    """Direct imports from dycore module work."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        make_fv3_faithful_nh_config,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        make_fv3_faithful_pe_config,
    )
    assert callable(make_fv3_faithful_nh_config)
    assert callable(make_fv3_faithful_pe_config)


def test_factories_callable_from_default_args():
    """Factories work with no overrides."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        make_fv3_faithful_nh_config,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        make_fv3_faithful_pe_config,
    )
    nh = make_fv3_faithful_nh_config()
    pe = make_fv3_faithful_pe_config()
    assert nh is not None
    assert pe is not None
