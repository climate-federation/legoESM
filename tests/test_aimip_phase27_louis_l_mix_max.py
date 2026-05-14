"""AIMIP Phase 2.7: ``l_mix_max`` override changes Louis output.

Numerical sentinel for the iter-258 wiring of ``l_mix_max`` through
``legoesm.atmosphere.physics.turbulence.louis.louis_turbulence``.  A
regression that silently drops the override (e.g. forgetting to read
the kwarg, falling back to ``config.l_mix_max`` in the mixing-length
formula) would make ``l_mix_max`` a nominal-only trainable again —
exactly the bug iter-256 reverted out of the system.  This sentinel
runs Louis twice on identical inputs with two different ``l_mix_max``
values and asserts the tendencies differ measurably.

Also pins:

- ``louis_turbulence`` accepts the new ``l_mix_max`` kwarg (signature
  check).
- Passing ``l_mix_max=None`` reproduces the default-``config.l_mix_max``
  path bit-for-bit (backward-compatibility guard).
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence


def _louis_inputs(ncol: int = 4, nlev: int = 8):
    """A minimal physically-plausible Louis input bundle."""
    u = jnp.zeros((ncol, nlev)).at[:, -2].set(10.0)
    v = jnp.zeros((ncol, nlev))
    T = jnp.full((ncol, nlev), 280.0)
    q_v = jnp.full((ncol, nlev), 1e-3)
    p_full = jnp.linspace(20000.0, 95000.0, nlev).reshape(1, nlev).repeat(ncol, axis=0)
    p_half = jnp.linspace(10000.0, 100000.0, nlev + 1).reshape(1, nlev + 1).repeat(ncol, axis=0)
    z_full = jnp.linspace(15000.0, 100.0, nlev).reshape(1, nlev).repeat(ncol, axis=0)
    z_half = jnp.linspace(16000.0, 0.0, nlev + 1).reshape(1, nlev + 1).repeat(ncol, axis=0)
    T_sfc = jnp.full((ncol,), 290.0)
    q_sfc = jnp.full((ncol,), 5e-3)
    rho = jnp.full((ncol, nlev), 1.0)
    return u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho


def test_louis_signature_has_l_mix_max_kwarg():
    """Pin that ``l_mix_max`` is a kwarg of ``louis_turbulence``.

    Catches a regression where the kwarg is silently dropped: without
    it the rest of the AIMIP wiring chain (build_segment_fn ->
    step_unified -> physics_step_no_rad) would forward an unknown
    argument and crash at runtime.
    """
    sig = inspect.signature(louis_turbulence)
    assert "l_mix_max" in sig.parameters, (
        "louis_turbulence is missing the ``l_mix_max`` kwarg added in "
        "iter-258 (AIMIP Phase 2.7).  See new_test_dycores.md."
    )


def test_l_mix_max_override_changes_output():
    """Override changes Louis tendencies measurably.

    Two runs with identical inputs and `l_mix_max` = 100 vs 200 must
    produce different du/dt fields.  The mixing-length formula is
        l = κ·z / (1 + κ·z / l_max)
    so doubling ``l_max`` doubles ``l`` at large z, which roughly
    quadruples ``K_m = l^2 · S`` and the surface-wind tendency.
    """
    cfg = LouisConfig()
    inputs = _louis_inputs()
    out_default = louis_turbulence(*inputs, 600.0, cfg)
    out_override = louis_turbulence(*inputs, 600.0, cfg, l_mix_max=200.0)
    diff = jnp.abs(out_default.du_dt - out_override.du_dt).max()
    base = jnp.abs(out_default.du_dt).max()
    rel = float(diff / (base + 1e-12))
    # require at least 10 % relative change somewhere in the field.
    assert rel > 0.1, (
        f"l_mix_max override produced no observable change "
        f"(rel diff = {rel:.3e}).  Override is not flowing into the "
        f"mixing-length formula — check louis.py read of ``l_mix_max``."
    )


def test_l_mix_max_none_matches_config_default():
    """``l_mix_max=None`` reproduces the static-config path bit-for-bit.

    Pins the backward-compatibility branch: the no-override default
    must remain unchanged so existing non-AIMIP runs are not silently
    perturbed.
    """
    cfg = LouisConfig()
    inputs = _louis_inputs()
    out_no_kw = louis_turbulence(*inputs, 600.0, cfg)
    out_none = louis_turbulence(*inputs, 600.0, cfg, l_mix_max=None)
    np.testing.assert_array_equal(
        np.asarray(out_no_kw.du_dt), np.asarray(out_none.du_dt)
    )
    np.testing.assert_array_equal(
        np.asarray(out_no_kw.dT_dt), np.asarray(out_none.dT_dt)
    )


def test_l_mix_max_override_matches_replaced_config():
    """Passing ``l_mix_max=X`` matches running with a config whose
    ``l_mix_max`` field has been replaced with ``X``.  This is the
    invariant the AIMIP training loop relies on — the gradient w.r.t.
    ``l_mix_max`` is the same regardless of whether we go through the
    kwarg override or rebuild ``LouisConfig``."""
    cfg = LouisConfig()
    cfg_replaced = cfg._replace(l_mix_max=200.0)
    inputs = _louis_inputs()
    out_override = louis_turbulence(*inputs, 600.0, cfg, l_mix_max=200.0)
    out_replaced = louis_turbulence(*inputs, 600.0, cfg_replaced)
    np.testing.assert_allclose(
        np.asarray(out_override.du_dt), np.asarray(out_replaced.du_dt),
        rtol=1e-6, atol=1e-7,
    )
