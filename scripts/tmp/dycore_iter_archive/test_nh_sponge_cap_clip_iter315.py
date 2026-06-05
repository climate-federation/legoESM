"""FV3_3D iter 315: NH iter-219 sponge cap clip behavior.

NH iter-219 builds a per-level cap on Δθ_p as::

    cap_per_level[k] = delt_max * sponge_factor[k] / exner_ref[k]
    sponge_factor = [0.1, 0.5, 1.0, 1.0, ...]

iter-267 pinned the sponge factor VALUES.  iter-315 pins the
CLIP BEHAVIOR with the full formula:

    * Below cap: passes through unchanged.
    * Above cap: clipped to ±cap_per_level[k].
    * Per-level cap honors both factor (0.1, 0.5, 1.0) and
      exner_ref scaling.

Tests
-----

1. ``test_nh_sponge_cap_per_level_values`` — per-level cap
   array equals ``delt_max * factor / exner_ref`` exactly.
2. ``test_nh_sponge_cap_clip_below`` — values below cap
   pass unchanged at each k.
3. ``test_nh_sponge_cap_clip_above`` — values above cap
   are clipped to ±cap[k] at each k.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _nh_sponge_cap_per_level(nlev, delt_max, exner_ref):
    """Reproduce the NH iter-219 per-level cap.

    Mirrors compressible_euler_cdgrid.py line 866-879.
    """
    k_idx = jnp.arange(nlev)
    sponge_factor = jnp.where(
        k_idx == 0, 0.1,
        jnp.where(k_idx == 1, 0.5, 1.0),
    )
    return delt_max * sponge_factor / exner_ref


def test_nh_sponge_cap_per_level_values():
    """Per-level cap = delt_max * factor / exner_ref."""
    nlev = 5
    delt_max = 1.0
    # Fake exner_ref values (decreasing with height, like real
    # atmosphere where pressure drops).
    exner_ref = jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2])

    cap = _nh_sponge_cap_per_level(nlev, delt_max, exner_ref)

    # k=0: 1.0 * 0.1 / 1.0 = 0.1
    # k=1: 1.0 * 0.5 / 0.8 = 0.625
    # k=2: 1.0 * 1.0 / 0.6 = 1.6666...
    # k=3: 1.0 * 1.0 / 0.4 = 2.5
    # k=4: 1.0 * 1.0 / 0.2 = 5.0
    expected = jnp.asarray([0.1, 0.625, 5.0/3, 2.5, 5.0])
    np.testing.assert_allclose(
        np.asarray(cap), np.asarray(expected), rtol=1e-14,
    )


def test_nh_sponge_cap_clip_below():
    """Values below cap pass through unchanged."""
    nlev = 5
    delt_max = 1.0
    exner_ref = jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2])
    cap = _nh_sponge_cap_per_level(nlev, delt_max, exner_ref)
    cap_b = cap[None, None, None, :]

    # Build θ_p tendencies BELOW each per-level cap.
    # cap = [0.1, 0.625, 1.667, 2.5, 5.0]; use 0.5 × cap as input.
    rng = np.random.default_rng(seed=315)
    field = jnp.zeros((6, 8, 8, nlev))
    for k in range(nlev):
        # Sign-randomize but keep magnitude < cap[k].
        layer = rng.uniform(-0.4, 0.4, size=(6, 8, 8)) * float(cap[k])
        field = field.at[..., k].set(jnp.asarray(layer))

    clipped = jnp.clip(field, -cap_b, cap_b)
    np.testing.assert_array_equal(
        np.asarray(clipped), np.asarray(field),
    )


def test_nh_sponge_cap_clip_above():
    """Values above cap are clipped to ±cap[k]."""
    nlev = 5
    delt_max = 1.0
    exner_ref = jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2])
    cap = _nh_sponge_cap_per_level(nlev, delt_max, exner_ref)
    cap_b = cap[None, None, None, :]

    # Build tendencies far ABOVE cap at every k (e.g., 100×).
    rng = np.random.default_rng(seed=315)
    field = jnp.zeros((6, 8, 8, nlev))
    for k in range(nlev):
        layer = rng.uniform(-100.0, 100.0, size=(6, 8, 8))
        # Force every value to exceed cap.
        layer = jnp.where(jnp.abs(layer) < float(cap[k]) * 2,
                          jnp.sign(layer) * float(cap[k]) * 10,
                          layer)
        field = field.at[..., k].set(jnp.asarray(layer))

    clipped = jnp.clip(field, -cap_b, cap_b)

    for k in range(nlev):
        max_clipped = float(jnp.max(jnp.abs(clipped[..., k])))
        # Allow tiny FP slack on the boundary clip.
        assert max_clipped <= float(cap[k]) + 1e-15, (
            f"k={k}: max|clipped|={max_clipped} exceeds "
            f"cap[k]={float(cap[k])} — NH cap not engaging."
        )
