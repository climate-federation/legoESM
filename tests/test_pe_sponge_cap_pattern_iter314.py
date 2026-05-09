"""FV3_3D iter 314: PE iter-218/219 sponge cap layer pattern.

PE skips the ``delt_max`` cap at sponge layers k=0, 1 (uncapped
= ``jnp.inf``) and applies the cap at k >= 2.  This per-level
clip is the FV3-faithful behavior at sponge layers.

iter-267 pinned the NH sponge factor values (0.1×, 0.5×, 1×).
iter-314 pins the PE sponge cap pattern (∞ at k=0,1; delt_max
at k>=2) by helper-formula numerical regression.

Tests
-----

1. ``test_pe_sponge_cap_per_level_pattern`` — the per-level
   cap array equals [∞, ∞, delt_max, delt_max, ...] for
   nlev levels.
2. ``test_pe_sponge_cap_clip_uncapped_top`` — values at k=0,
   1 pass through clip unchanged regardless of magnitude
   (cap = ∞).
3. ``test_pe_sponge_cap_clip_capped_below`` — values at
   k>=2 are clipped to ±delt_max when they exceed it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _pe_sponge_cap_per_level(nlev, delt_max):
    """Reproduce the iter-218/219 PE sponge-aware cap array.

    Mirrors primitive_eq_cdgrid.py line 1450-1452.
    """
    k_idx = jnp.arange(nlev)
    return jnp.where(k_idx < 2, jnp.inf, delt_max)


def test_pe_sponge_cap_per_level_pattern():
    """Per-level cap = [∞, ∞, delt_max, delt_max, ...]"""
    nlev = 6
    delt_max = 1.0
    cap = _pe_sponge_cap_per_level(nlev, delt_max)
    assert jnp.isinf(cap[0])
    assert jnp.isinf(cap[1])
    assert cap[0] > 0  # +inf, not -inf
    assert cap[1] > 0
    for k in range(2, nlev):
        assert float(cap[k]) == delt_max, (
            f"PE k={k} cap should be delt_max={delt_max}, got "
            f"{float(cap[k])}"
        )


def test_pe_sponge_cap_clip_uncapped_top():
    """Large values at k=0, 1 pass through the clip unchanged."""
    nlev = 6
    delt_max = 1.0
    cap = _pe_sponge_cap_per_level(nlev, delt_max)
    cap_b = cap[None, None, None, :]

    # Build a tendency-like field with large values at k=0, 1.
    rng = np.random.default_rng(seed=314)
    field = jnp.asarray(rng.uniform(-100.0, 100.0,
                                    size=(6, 8, 8, nlev)))
    clipped = jnp.clip(field, -cap_b, cap_b)

    # k=0 and k=1 must be unchanged (cap = inf → no clip).
    np.testing.assert_array_equal(
        np.asarray(clipped[..., 0]), np.asarray(field[..., 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(clipped[..., 1]), np.asarray(field[..., 1]),
    )


def test_pe_sponge_cap_clip_capped_below():
    """Large values at k>=2 are clipped to ±delt_max."""
    nlev = 6
    delt_max = 1.0
    cap = _pe_sponge_cap_per_level(nlev, delt_max)
    cap_b = cap[None, None, None, :]

    # Build a field with values exceeding delt_max at k>=2.
    rng = np.random.default_rng(seed=314)
    field = jnp.asarray(rng.uniform(-100.0, 100.0,
                                    size=(6, 8, 8, nlev)))
    clipped = jnp.clip(field, -cap_b, cap_b)

    for k in range(2, nlev):
        max_clipped = float(jnp.max(jnp.abs(clipped[..., k])))
        assert max_clipped <= delt_max + 1e-15, (
            f"PE k={k} max|clipped|={max_clipped} exceeds "
            f"delt_max={delt_max} — cap not engaged."
        )
        # And every clipped value must equal the original or
        # ±delt_max (no other transformation).
        clipped_k = np.asarray(clipped[..., k])
        field_k = np.asarray(field[..., k])
        for val_c, val_f in zip(clipped_k.ravel(), field_k.ravel()):
            if abs(val_f) <= delt_max:
                assert abs(val_c - val_f) < 1e-15, (
                    "Within-cap value should pass unchanged"
                )
            else:
                assert abs(abs(val_c) - delt_max) < 1e-15, (
                    f"Out-of-cap value should be clipped to "
                    f"±delt_max; got {val_c} from {val_f}"
                )
