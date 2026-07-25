"""Convective-cloud (Slingo 1987) threading on the MPAS standalone path.

2026-07-24 gap closure: the standalone (MPAS) radiation path used to FORCE
``convective_cloud=False`` because ``conv_precip`` was never threaded into
``compute_cloud_properties``.  The closure is a one-step LAG carry — the
convection bridge publishes its column-integrated in-updraft rain production
into ``PhysicsState.conv_precip`` and the radiation module (which runs first
in the combined-physics chain) reads the previous step's value, matching the
FV pipeline's lagged ``conv_precip`` convention.

Seam tests here (the bridge-side publication is pinned next to the rain-split
conservation tests in
``tests/atmosphere/hydrostatic/unit/test_convection_rain_split_bridges.py``):

1. ``update_physics_state`` CARRIES ``conv_precip`` forward when a step's
   convection publishes nothing (the None-regression: dropping it to the
   field default poisons the MPAS checkpoint with an object array).
2. ``_standalone_cloud_config`` honours ``convective_cloud`` ONLY where the
   carry exists (MPAS ``allow_convective_cloud=True``; spectral stays off).
3. The hydrostatic radiation factory becomes a ``phys_state`` reader exactly
   when its cloud config enables ``convective_cloud``.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.physics_state import (
    init_physics_state, update_physics_state,
)
from legoesm.atmosphere.physics.combined import PhysicsConfig


def _mini_phys_state(ncol=4, nlev=3):
    return init_physics_state(ncol, nlev, PhysicsConfig())


def test_update_physics_state_carries_conv_precip_forward():
    ps = _mini_phys_state()
    assert ps.conv_precip is not None and ps.conv_precip.shape == (4,)
    # No update published (e.g. convection scheme without a rain split):
    # the carry must survive UNCHANGED — not fall to the None field default
    # (which np.asarray()s into an unloadable object array in the MPAS
    # checkpoint writer).
    seeded = ps._replace(conv_precip=jnp.arange(4, dtype=jnp.float64))
    out = update_physics_state(seeded, {})
    np.testing.assert_array_equal(np.asarray(out.conv_precip),
                                  np.arange(4, dtype=np.float64))
    # A published update replaces it.
    out2 = update_physics_state(seeded, {"conv_precip": jnp.ones(4)})
    np.testing.assert_array_equal(np.asarray(out2.conv_precip), np.ones(4))


def test_standalone_cloud_config_honours_conv_cloud_on_mpas_only():
    from legoesm.driver.model_driver import _standalone_cloud_config
    cfg = types.SimpleNamespace(convective_cloud=True)
    on = _standalone_cloud_config(cfg, "sundqvist", allow_convective_cloud=True)
    off = _standalone_cloud_config(cfg, "sundqvist")   # spectral path default
    assert on.convective_cloud is True
    assert off.convective_cloud is False
    # A user who never asked for it gets it off on both paths.
    cfg_off = types.SimpleNamespace(convective_cloud=False)
    assert _standalone_cloud_config(
        cfg_off, "sundqvist", allow_convective_cloud=True
    ).convective_cloud is False


def test_radiation_factory_reads_phys_state_iff_conv_cloud_enabled():
    from legoesm.atmosphere.physics.radiation.integration import (
        _make_hydrostatic_radiation,
    )
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config

    rc_on = RadiationConfig(
        scheme="gray", cloud_scheme="sundqvist",
        cloud_config=build_cloud_config("sundqvist", convective_cloud=True),
    )
    fn_on = _make_hydrostatic_radiation(rc_on)
    assert getattr(fn_on, "_wants_phys_state_ro", False) is True, (
        "convective_cloud=True must make the radiation fn a phys_state "
        "reader (the lagged conv_precip carry)")

    rc_off = RadiationConfig(
        scheme="gray", cloud_scheme="sundqvist",
        cloud_config=build_cloud_config("sundqvist", convective_cloud=False),
    )
    fn_off = _make_hydrostatic_radiation(rc_off)
    assert getattr(fn_off, "_wants_phys_state_ro", False) is False, (
        "without convective_cloud the fn must NOT grow a phys_state "
        "dependency (byte-identical contract)")
