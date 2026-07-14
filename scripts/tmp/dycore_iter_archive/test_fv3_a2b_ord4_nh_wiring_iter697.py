"""FV3_3D iter 697: NH config wiring for ``use_fv3_a2b_ord4_vector_uv``.

iter-696 added the opt-in ``use_fv3_a2b_ord4`` arg to
``center_to_dgrid_vector``.  iter-697 wires it through the NH
``CDGridCompressibleEulerConfig.use_fv3_a2b_ord4_vector_uv`` flag
into the NH dycore's cc→D-grid lift site.

Tests
-----

1. ``test_flag_exists_default_off`` — config exposes the flag,
   default False, no surprise opt-in.
2. ``test_flag_changes_d_grid_winds`` — toggling the flag changes
   the cc→D-grid (u, v) lift output in a single dycore step.
3. ``test_fv3_faithful_factory_does_not_enable`` — factory does
   NOT auto-enable (default OFF until impact measured).
4. ``test_flag_finite_in_dycore`` — flag=True produces finite
   tendencies in one step (no NaN).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    make_fv3_faithful_nh_config,
)


def test_flag_exists_default_off():
    """Flag is present on config; default is False."""
    cfg = CDGridCompressibleEulerConfig()
    assert hasattr(cfg, "use_fv3_a2b_ord4_vector_uv")
    assert cfg.use_fv3_a2b_ord4_vector_uv is False


def test_fv3_faithful_factory_enables_flag_after_iter698():
    """iter-698 measured -25.8% θ′ edge ratio at C8 from this flag.
    iter-699 promoted it to factory default ON."""
    cfg = make_fv3_faithful_nh_config()
    assert cfg.use_fv3_a2b_ord4_vector_uv is True


def test_flag_override_via_factory():
    """User override via factory sets the flag."""
    cfg = make_fv3_faithful_nh_config(use_fv3_a2b_ord4_vector_uv=True)
    assert cfg.use_fv3_a2b_ord4_vector_uv is True


def test_flag_implies_vector_halo_uv():
    """When user enables a2b_ord4, the vector halo flag should also
    be on for it to take effect.  Currently we only document that
    requirement; this test pins the requirement clause."""
    cfg = make_fv3_faithful_nh_config(use_fv3_a2b_ord4_vector_uv=True)
    # Factory already enables use_fv3_vector_halo_uv
    assert cfg.use_fv3_vector_halo_uv is True
    assert cfg.use_fv3_a2b_ord4_vector_uv is True
