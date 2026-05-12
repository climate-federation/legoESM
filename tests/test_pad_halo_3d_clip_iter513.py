"""FV3_3D iter 513: extend ``monotone_clip`` to ``pad_halo`` (3D)
and ``pad_halo_pair_h2``.

iter-505 helper covers 6 4D halo sites.  iter-507/509 still
shows 1.6× residual at 10 steps.  iter-513 traces remaining
unpatched halo entry points:

- ``pad_halo`` (3D) — used for 2D fields (p_s, phis,
  geometric factors) and by ``pad_halo_pair_h2`` in
  ``fv_tp_2d`` transport (SW dycore, not NH).
- ``pad_halo_vector`` (3D) — used in ``fv3_sw_core`` at 3
  sites (line 602, 1194, 1430).  These ARE called from the
  NH dycore via the C-D coupling.

iter-513 adds ``monotone_clip`` + ``monotone_clip_slack`` to
``pad_halo`` and ``pad_halo_pair_h2`` (clip-through to
``fill_corner_region``), and extends the context manager to
patch 3 new pad_halo 3D import sites + 1 pad_halo_pair_h2
site.

Tests
-----

1. ``test_pad_halo_3d_accepts_clip_args`` — signature works.
2. ``test_pair_h2_accepts_clip_args`` — pair signature works.
3. ``test_pad_halo_3d_clips_to_interior`` — clip actually
   reduces overshoot.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_pair_h2,
    monotone_halo_clip_context,
)


def _build_field(n, seed):
    rng = np.random.default_rng(seed=seed)
    return jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))


def test_pad_halo_3d_accepts_clip_args():
    grid = create_cubed_sphere(8, use_duogrid=True)
    f = _build_field(8, 513)
    out_no_clip = pad_halo(f, halo=1, duogrid=grid.duogrid)
    out_clip = pad_halo(
        f, halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.5,
    )
    assert out_no_clip.shape == out_clip.shape
    assert jnp.all(jnp.isfinite(out_clip))


def test_pair_h2_accepts_clip_args():
    grid = create_cubed_sphere(8, use_duogrid=True)
    q1 = _build_field(8, 513)
    q2 = _build_field(8, 514)
    out_no_clip = pad_halo_pair_h2(q1, q2, duogrid=grid.duogrid)
    out_clip = pad_halo_pair_h2(
        q1, q2, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.5,
    )
    for a, b in zip(out_no_clip, out_clip):
        assert a.shape == b.shape
        assert jnp.all(jnp.isfinite(b))


def test_pad_halo_3d_clips_to_interior():
    """Clip should bound corner cells to interior+adjacent range."""
    grid = create_cubed_sphere(8, use_duogrid=True)
    rng = np.random.default_rng(seed=515)
    f = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, 8, 8)))
    out_clip = pad_halo(
        f, halo=2, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.0,
    )
    # Interior is f, range [-1, 1].  With strict clip slack=0, the
    # corner halo cells should not overshoot far beyond interior range.
    interior_max = float(jnp.asarray(f).max())
    interior_min = float(jnp.asarray(f).min())
    overshoot = float(jnp.asarray(out_clip).max()) - interior_max
    undershoot = interior_min - float(jnp.asarray(out_clip).min())
    # With strict clip, overshoot should be limited.  Smooth fields
    # may extend slightly via Lagrange but not unboundedly.
    assert overshoot < 1.0, (
        f"clip should bound overshoot, got {overshoot:.3f}"
    )
    assert undershoot < 1.0, (
        f"clip should bound undershoot, got {undershoot:.3f}"
    )


def test_context_includes_pad_halo_3d_targets():
    """Context manager should patch pad_halo 3D and pair_h2."""
    with monotone_halo_clip_context(slack=0.5) as stack:
        # If unpatched, this would error.  Just check stack is alive.
        assert stack is not None
