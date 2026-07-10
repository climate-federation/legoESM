"""#911: surface_T_sfc_override uses a FINITE "no override" sentinel so the
state never carries NaN (which poisoned JAX_DEBUG_NANS + the realism inspector).
The turbulence resolver must still fall back for unset columns, use a real
physical override where present, AND stay backward-compatible with legacy
checkpoints that were written with the old NaN sentinel.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.physics_state import (
    NO_SFC_T_OVERRIDE,
    init_physics_state,
)
from legoesm.atmosphere.physics.radiation.integration import _apply_T_sfc_override
from legoesm.atmosphere.physics.turbulence.integration import _resolve_T_sfc

_CFG = PhysicsConfig(
    radiation=RadiationConfig(scheme="none"),
    convection=ConvectionConfig(scheme="none"),
    turbulence=TurbulenceConfig(scheme="none"),
    microphysics=MicrophysicsConfig(scheme="none"),
    gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
)


def test_init_override_is_finite():
    ps = init_physics_state(8, 10, _CFG)
    ov = np.asarray(ps.surface_T_sfc_override)
    assert np.all(np.isfinite(ov))           # the whole point of #911
    assert np.allclose(ov, NO_SFC_T_OVERRIDE)


def _resolve(override):
    T_col = jnp.broadcast_to(jnp.asarray([250.0, 260.0, 270.0])[:, None], (3, 5))
    ps = init_physics_state(3, 5, _CFG)
    ps = ps._replace(surface_T_sfc_override=jnp.asarray(override))
    return np.asarray(_resolve_T_sfc(T_col, ps))


def test_sentinel_falls_back_to_lowest_level():
    out = _resolve([NO_SFC_T_OVERRIDE] * 3)
    assert np.allclose(out, [250.0, 260.0, 270.0])   # T_col[:, -1]


def test_physical_override_is_used():
    out = _resolve([288.0, NO_SFC_T_OVERRIDE, 300.0])
    assert np.allclose(out, [288.0, 260.0, 300.0])   # mix: override / fallback


def test_legacy_nan_sentinel_still_falls_back():
    # a restart from a checkpoint written with the OLD NaN sentinel
    out = _resolve([np.nan, 295.0, np.nan])
    assert np.allclose(out, [250.0, 295.0, 270.0])
    assert np.all(np.isfinite(out))


def test_radiation_consumer_shares_predicate():
    # radiation's _apply_T_sfc_override must agree with the turbulence resolver:
    # sentinel/NaN keep the base T_sfc, physical values win.
    T_sfc = jnp.asarray([250.0, 260.0, 270.0])
    override = jnp.asarray([NO_SFC_T_OVERRIDE, 295.0, np.nan])
    out = np.asarray(_apply_T_sfc_override(T_sfc, override))
    assert np.allclose(out, [250.0, 295.0, 270.0])
    assert np.all(np.isfinite(out))
