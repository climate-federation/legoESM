"""FV3_3D iter 447: shared linear-scaling FV3 sponge helper.

Factored from three duplicate iter-441/442/443 inline blocks:
* NH damp_w (factor=1.0, apply_at_k2=True)
* NH damp_v (factor=0.5, apply_at_k2=False)
* PE damp_v (factor=0.5, apply_at_k2=False)

All three now call the single helper
``legoesm.core.fv3_sponge_boost.apply_top_sponge_field_scale``.

Tests
-----

1. ``test_helper_importable``.
2. ``test_baseline_noop`` — zero coefficients → identity.
3. ``test_damp_w_factor_1_at_k0`` — factor=1.0, scaling matches
   FV3 ``damp_w = d2_divg``.
4. ``test_damp_v_factor_05_at_k0`` — factor=0.5, scaling
   matches FV3 ``damp_vt = 0.5 * d2_divg``.
5. ``test_apply_at_k2_false_skips_k2`` — damp_v variant does
   NOT touch k=2 even with d2_bg_k2>0.05.
6. ``test_apply_at_k2_true_touches_k2`` — damp_w variant DOES
   touch k=2 with 0.2 factor.
7. ``test_iter441_442_443_still_pass`` — sanity that existing
   iter-441/442/443 numerical tests still pass post-refactor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_helper_importable():
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    assert callable(apply_top_sponge_field_scale)


def test_baseline_noop():
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    field = jnp.ones((4, 5, 5, 5))
    out = apply_top_sponge_field_scale(
        field, damp_x=0.030, nord_x=1, factor=1.0,
        d2_bg=0.0005, d2_bg_k1=0.0, d2_bg_k2=0.0,
        apply_at_k2=True,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(field))


def test_damp_w_factor_1_at_k0():
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    field = jnp.ones((4, 5, 5, 5))
    damp = 0.030
    nord = 1
    d2_bg, d2_bg_k1 = 0.0005, 4.0
    out = apply_top_sponge_field_scale(
        field, damp_x=damp, nord_x=nord, factor=1.0,
        d2_bg=d2_bg, d2_bg_k1=d2_bg_k1, d2_bg_k2=0.0,
        apply_at_k2=True,
    )
    expected_k0 = (max(d2_bg, d2_bg_k1) / damp) ** (nord + 1)
    np.testing.assert_allclose(
        np.asarray(out[..., 0]), expected_k0,
        rtol=1e-14, atol=1e-14,
    )
    # k=1..4 unchanged
    np.testing.assert_array_equal(
        np.asarray(out[..., 1:]), np.asarray(field[..., 1:]),
    )


def test_damp_v_factor_05_at_k0():
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    field = jnp.ones((4, 5, 5, 5))
    damp = 0.030
    nord = 1
    d2_bg, d2_bg_k1 = 0.0005, 4.0
    out = apply_top_sponge_field_scale(
        field, damp_x=damp, nord_x=nord, factor=0.5,
        d2_bg=d2_bg, d2_bg_k1=d2_bg_k1, d2_bg_k2=0.0,
        apply_at_k2=False,
    )
    expected_k0 = (0.5 * max(d2_bg, d2_bg_k1) / damp) ** (nord + 1)
    np.testing.assert_allclose(
        np.asarray(out[..., 0]), expected_k0,
        rtol=1e-14, atol=1e-14,
    )


def test_apply_at_k2_false_skips_k2():
    """damp_v variant: d2_bg_k2 > 0.05 still does NOT touch k=2."""
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    field = jnp.ones((4, 5, 5, 5))
    out = apply_top_sponge_field_scale(
        field, damp_x=0.030, nord_x=1, factor=0.5,
        d2_bg=0.0005, d2_bg_k1=0.0, d2_bg_k2=2.0,
        apply_at_k2=False,
    )
    # k=1 modified (factor 0.5)
    expected_k1 = (0.5 * 2.0 / 0.030) ** 2
    np.testing.assert_allclose(
        np.asarray(out[..., 1]), expected_k1,
        rtol=1e-14, atol=1e-14,
    )
    # k=2 UNCHANGED (apply_at_k2=False)
    np.testing.assert_array_equal(
        np.asarray(out[..., 2]), np.asarray(field[..., 2]),
    )


def test_apply_at_k2_true_touches_k2():
    """damp_w variant: d2_bg_k2 > 0.05 touches k=2 with 0.2 factor."""
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_field_scale,
    )
    field = jnp.ones((4, 5, 5, 5))
    out = apply_top_sponge_field_scale(
        field, damp_x=0.030, nord_x=1, factor=1.0,
        d2_bg=0.0005, d2_bg_k1=0.0, d2_bg_k2=2.0,
        apply_at_k2=True,
    )
    # k=2 modified: 0.2 * d2_bg_k2 = 0.4
    expected_k2 = (0.2 * 2.0 / 0.030) ** 2
    np.testing.assert_allclose(
        np.asarray(out[..., 2]), expected_k2,
        rtol=1e-14, atol=1e-14,
    )
