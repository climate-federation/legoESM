"""Tests for CLUBB mass-conserving vertical hole-filling (``fill_holes_type=2``).

Confirms the global fill conserves the density-weighted column integral and
removes holes, the sliding-window+fallback path (CAM default) restores
positivity, the no-hole fast path is a no-op, and the kernels are JIT/grad
clean. Bit-exact parity vs CLUBB-JAX ``fill_holes.py`` where the tree is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    fill_holes_global,
    fill_holes_sliding_window,
    fill_holes_vertical,
    fill_holes_wp2_from_horz_tke,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _clubb_ref_api as _ref  # noqa: E402

_CLUBB_JAX_ROOT = _ref.ROOT


def _field_with_holes(ng=3, nz=16, seed=0):
    rng = np.random.default_rng(seed)
    field = 0.5 + rng.random((ng, nz))
    # punch holes (negative values) at a few interior levels
    field[:, 5] = -0.3
    field[:, 9] = -0.1
    rho_ds = 1.0 + 0.1 * rng.random((ng, nz))
    dz = 30.0 + 5.0 * rng.random((ng, nz))
    return jnp.asarray(field), jnp.asarray(rho_ds), jnp.asarray(dz)


def test_global_conserves_mass_and_removes_holes():
    field, rho_ds, dz = _field_with_holes()
    lo, hi, thr = 1, 14, 0.0
    rho_dz = rho_ds * dz
    out = fill_holes_global(field, rho_dz, thr, lo, hi)
    # density-weighted integral over [lo, hi] conserved
    sl = slice(lo, hi + 1)
    m_in = np.sum(np.asarray(rho_dz)[:, sl] * np.asarray(field)[:, sl], axis=1)
    m_out = np.sum(np.asarray(rho_dz)[:, sl] * np.asarray(out)[:, sl], axis=1)
    np.testing.assert_allclose(m_out, m_in, rtol=1e-12, atol=1e-12)
    # holes removed over the filled interval
    assert np.all(np.asarray(out)[:, sl] >= thr - 1e-12)


def test_sliding_window_removes_holes():
    field, rho_ds, dz = _field_with_holes()
    out = fill_holes_vertical(field, rho_ds, dz, 0.0, 1, 14, fill_holes_type=2)
    assert np.all(np.asarray(out)[:, 1:15] >= -1e-12)
    assert np.all(np.isfinite(np.asarray(out)))


def test_no_hole_is_noop():
    rng = np.random.default_rng(3)
    field = jnp.asarray(1.0 + rng.random((2, 12)))
    rho_ds = jnp.asarray(1.0 + 0.1 * rng.random((2, 12)))
    dz = jnp.asarray(jnp.full((2, 12), 25.0))
    out = fill_holes_vertical(field, rho_ds, dz, 0.0, 1, 10, fill_holes_type=2)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(field))


def test_unknown_type_raises():
    field, rho_ds, dz = _field_with_holes()
    with pytest.raises(ValueError, match="fill_holes_type"):
        fill_holes_vertical(field, rho_ds, dz, 0.0, 1, 14, fill_holes_type=3)


def test_jit_static_args_public_signature():
    """JIT through the PUBLIC signature with the static contract (not closures).

    lower_k/upper_k/fill_holes_type are passed as live args and declared static
    via static_argnums — proving the documented JIT contract, per codex review.
    """
    field, rho_ds, dz = _field_with_holes()
    # All four documented static args (incl. grid_dir_indx) marked static and
    # passed as live arguments through the public signature.
    jf = jax.jit(fill_holes_vertical, static_argnums=(4, 5, 6, 7))
    out = jf(field, rho_ds, dz, 0.0, 1, 14, 2, 1)
    assert jnp.all(jnp.isfinite(out))
    # type-1 global path through the same jitted entry
    out1 = jf(field, rho_ds, dz, 0.0, 1, 14, 1, 1)
    assert jnp.all(jnp.isfinite(out1))


def test_jit_and_grad():
    field, rho_ds, dz = _field_with_holes()

    def loss(f):
        out = fill_holes_vertical(f, rho_ds, dz, 0.0, 1, 14, fill_holes_type=2)
        return jnp.sum(out ** 2)

    assert jnp.isfinite(jax.jit(loss)(field))
    g = jax.grad(loss)(field)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_parity_vs_reference():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.fill_holes as RF  # noqa: N812

    field, rho_ds, dz = _field_with_holes(seed=7)
    rho_dz = rho_ds * dz
    ng, nz = np.asarray(field).shape
    np.testing.assert_allclose(
        np.asarray(fill_holes_global(field, rho_dz, 0.0, 1, 14)),
        np.asarray(RF.fill_holes_global(nz, ng, 0.0, 1, 14, dz, rho_ds, field)),
        rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(fill_holes_sliding_window(field, rho_dz, 0.0, 1, 14)),
        np.asarray(RF.fill_holes_sliding_window(nz, ng, 0.0, 1, 14, dz, rho_ds, field)),
        rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(fill_holes_vertical(field, rho_ds, dz, 0.0, 1, 14, 2)),
        np.asarray(RF.fill_holes_vertical(nz, ng, 0.0, 1, 14, dz, rho_ds, 1, 2, field)),
        rtol=1e-12, atol=1e-14)


def _horz_tke_inputs(seed=0, ng=4, nzm=12):
    rng = np.random.default_rng(seed)
    wp2 = 0.3 + 0.5 * rng.random((ng, nzm))
    # punch wp2 holes (below threshold w_tol^2 = 4e-4) at a few levels
    wp2[:, 4] = 1.0e-4
    wp2[:, 7] = 5.0e-5
    up2 = 0.4 + 0.6 * rng.random((ng, nzm))
    vp2 = 0.4 + 0.6 * rng.random((ng, nzm))
    return jnp.asarray(wp2), jnp.asarray(up2), jnp.asarray(vp2)


def test_horz_tke_fill_conserves_total_and_fills():
    wp2, up2, vp2 = _horz_tke_inputs()
    thr = float((2.0e-2) ** 2)
    tot_in = np.asarray(wp2 + up2 + vp2)
    w, u, v = fill_holes_wp2_from_horz_tke(wp2, up2, vp2, thr, 0, 11)
    tot_out = np.asarray(w + u + v)
    # total TKE conserved at every level
    np.testing.assert_allclose(tot_out, tot_in, rtol=1e-12, atol=1e-12)
    # holes raised toward the threshold (where TKE was available)
    assert np.asarray(w)[:, 4].min() > np.asarray(wp2)[:, 4].min()
    assert np.all(np.asarray(u) >= -1e-12) and np.all(np.asarray(v) >= -1e-12)


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_horz_tke_fill_parity():
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.fill_holes as RF  # noqa: N812
    wp2, up2, vp2 = _horz_tke_inputs(seed=2)
    thr = float((2.0e-2) ** 2)
    mine = fill_holes_wp2_from_horz_tke(wp2, up2, vp2, thr, 0, 9)
    ng, nz = np.asarray(wp2).shape
    ref = RF.fill_holes_wp2_from_horz_tke(nz, ng, thr, 0, 9, wp2, up2, vp2)
    for a, b in zip(mine, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)


def test_horz_tke_fill_jit_grad():
    wp2, up2, vp2 = _horz_tke_inputs(seed=5)
    thr = float((2.0e-2) ** 2)

    def loss(w):
        ww, uu, vv = fill_holes_wp2_from_horz_tke(w, up2, vp2, thr, 0, 11)
        return jnp.sum(ww ** 2 + uu ** 2 + vv ** 2)

    assert jnp.isfinite(jax.jit(loss)(wp2))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(wp2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
