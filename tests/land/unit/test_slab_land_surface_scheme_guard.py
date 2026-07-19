"""Dispatch hardening: slab ``step_land`` must raise on an unknown
``surface_scheme`` type.

``step_land`` dispatches ``isinstance(config.surface_scheme,
TwoLeafCanopyConfig)`` and otherwise FALLS THROUGH to the SimpleSEB path with
no type check — so a surface_scheme that is neither a ``SimpleSEBConfig`` nor
a ``TwoLeafCanopyConfig`` silently runs the bulk SEB closure.  The multilayer
sibling (``multilayer_land.py``) already raises ``Unknown surface_scheme
type``; slab was the only entry missing the guard.  The guard fires before
any array work, so a throwaway state/forcing is enough to reach it.
"""

from __future__ import annotations

import pytest

from legoesm.land.config import LandConfig
from legoesm.land.slab_land import step_land


def test_step_land_rejects_unknown_surface_scheme():
    bad = LandConfig()._replace(surface_scheme=object())   # neither SEB nor canopy
    with pytest.raises(ValueError, match="surface_scheme"):
        # state/forcing are never touched before the dispatch guard.
        step_land(None, None, bad, U_min=0.1, dt=100.0)
