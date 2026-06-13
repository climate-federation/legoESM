"""Sub-face tiled FV3 PPM transport sweep SPMD stage (P4 phase-1b, task #3).

Approach C (mirrors :func:`legoesm.parallel.tiled_d2a2c.make_tiled_d2a2c_stage`):
the cross-face D-grid halo + the h3=4 PPM pre-pad run in the global
(face-replicated) view; the per-tile PPM sweep is sharded over the device
mesh — each device slices its tile's sweep window from the replicated
padded field and runs the device-uniform
:func:`legoesm.core.fv3_sw_core._ppm_transport_1d` with
``external_halo=h3, rd_prepadded=True`` (so the padded face already carries
the halo — NO in-stage halo ppermute, the same insight as the d2a2c stage).

The per-tile compute is bit-exact vs the global PPM sweep
(``tests/parallel/test_ppm_transport_tile.py`` U2/U2b); this module proves
it holds INSIDE a real ``shard_map`` (``tests/parallel/
test_tiled_transport_stage.py``).

This first stage tiles the SWEEP axis only (i-sweep ``xtp_u`` on a
``(6, kt)`` ``(face, tile_i)`` mesh, full cross axis) — np = 6*kt.  The 2-D
``(6, kt, kt)`` staggered-cross-axis tiling + the ``ytp_v`` j-sweep + the
real Courant / B-grid corner sync are follow-ups (design:
``docs/scaling/cube_transport_tiling_design.md`` §U3).
"""
from __future__ import annotations

from functools import partial

import jax
from jax.sharding import PartitionSpec as P

try:  # JAX >= 0.8 top-level export
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map


def transport_sweep_tile(vp_g, courant, rd_g, a, nl: int, h3: int = 4):
    """Per-tile i-sweep PPM flux: slice the tile window (SWEEP axis=1; the
    leading face axis and the trailing cross axis are kept) from the
    face-replicated padded inputs at start ``a``, then run the PPM flux.

    Shared by the shard_map stage (``a = axis_index("tile_i") * nl``) and a
    non-shard body test (``a = t * nl``) so the exact slice shapes + the
    ``_ppm_transport_1d`` call are CI-covered WITHOUT a multi-device mesh
    (codex U3 MEDIUM — the shard_map test silently skips below 6*kt host
    devices).  ``vp_g`` is ``(F, n+2*h3, m)`` (F=1 inside the stage shard,
    F=6 in the host test); the tile's i-cells ``[a-h3 : a+nl+h3]`` live at
    padded indices ``[a : a+nl+2*h3]`` (the slice U2/U2b validated).
    ``dynamic_slice_in_dim`` takes a single start index (no mixed
    int32/int64 index-tuple error from ``axis_index``).  Returns
    ``(F, nl+1, m)`` i-interface flux.
    """
    from legoesm.core.fv3_sw_core import _ppm_transport_1d

    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a, nl + 2 * h3, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a, nl + 1, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a, nl + 2, axis=1)
    return _ppm_transport_1d(
        vp_t, c_t, rd_t, 1, external_halo=h3, rd_prepadded=True)


def make_tiled_transport_sweep_stage(mesh, n: int, kt: int, h3: int = 4):
    """Build the sharded i-sweep PPM transport stage for a ``(6, kt)`` mesh.

    mesh : ``jax.sharding.Mesh`` with axis names ``("face", "tile_i")`` and
        shape ``(6, kt)``.

    Returns ``stage(vp_g, courant, rd_g) -> flux`` where
      * ``vp_g`` : ``(6, n+2*h3, M)`` the GLOBAL field already padded to the
        PPM storage halo h3 (cross-face halo + edge-pad), FACE-REPLICATED;
      * ``courant`` : ``(6, n+1, M)`` i-interface Courant numbers;
      * ``rd_g`` : ``(6, n+2, M)`` the depth-1 edge-padded ``rdelta``;
    and the output ``flux`` is the tile-sharded i-interface flux of gathered
    extent ``(6, kt*(nl+1), M)`` — each tile contributes ``nl+1`` interfaces
    DUPLICATING the shared boundary with its neighbour (reassembly: lower
    tile owns the shared interface → the global ``(6, n+1, M)`` PPM sweep).
    """
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    fo = P("face", None, None)          # face-replicated padded inputs
    co = P("face", "tile_i", None)      # tile-i-sharded staggered flux

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo),
             out_specs=co, check_vma=False)
    def _stage(vp_g, courant, rd_g):
        a = jax.lax.axis_index("tile_i") * nl
        return transport_sweep_tile(vp_g, courant, rd_g, a, nl, h3)

    return _stage


# ---------------------------------------------------------------------------
# U3b: full 2-D (6, kt, kt) tiling — BOTH the cross axis tiled AND the
# symmetric j-sweep (ytp_v).  The two PPM sweeps are INDEPENDENT 1-D passes
# (fv3_sw_core._bgrid_ke_transport: axis=2 ytp_v, axis=1 xtp_u — no coupling
# at the sweep level), and each sweep is independent PER cross-row, so the
# CROSS axis is sliced to the tile WITHOUT a halo (only the sweep axis carries
# the h3 PPM halo).  np = 6*kt*kt (kt=2 → the np24 target).
# ---------------------------------------------------------------------------

def transport_sweep_tile_2d(vp_g, courant, rd_g, a_i, a_j, nl: int, h3: int = 4):
    """Per-tile i-sweep (xtp_u, axis=1) PPM flux on a 2-D ``(tile_i, tile_j)``
    tiling.  Slice the SWEEP axis=1 to the window ``[a_i : a_i+nl+2*h3]`` AND
    the CROSS axis=2 to the tile's j-cells ``[a_j : a_j+nl]`` (NO j-halo — the
    i-sweep is an independent 1-D PPM per j-row).  ``vp_g`` is
    ``(F, n+2*h3, n)``; returns ``(F, nl+1, nl)`` i-interface flux."""
    from legoesm.core.fv3_sw_core import _ppm_transport_1d

    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a_i, nl + 2 * h3, axis=1)
    vp_t = jax.lax.dynamic_slice_in_dim(vp_t, a_j, nl, axis=2)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a_i, nl + 1, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(c_t, a_j, nl, axis=2)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a_i, nl + 2, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_t, a_j, nl, axis=2)
    return _ppm_transport_1d(
        vp_t, c_t, rd_t, 1, external_halo=h3, rd_prepadded=True)


def transport_jsweep_tile_2d(vp_g, courant, rd_g, a_i, a_j, nl: int, h3: int = 4):
    """Per-tile j-sweep (ytp_v, axis=2) PPM flux — the symmetric counterpart
    of :func:`transport_sweep_tile_2d` with the sweep/cross axes swapped.
    Slice the SWEEP axis=2 to ``[a_j : a_j+nl+2*h3]`` AND the CROSS axis=1 to
    ``[a_i : a_i+nl]`` (NO i-halo).  ``vp_g`` is ``(F, n, n+2*h3)``; returns
    ``(F, nl, nl+1)`` j-interface flux."""
    from legoesm.core.fv3_sw_core import _ppm_transport_1d

    vp_t = jax.lax.dynamic_slice_in_dim(vp_g, a_j, nl + 2 * h3, axis=2)
    vp_t = jax.lax.dynamic_slice_in_dim(vp_t, a_i, nl, axis=1)
    c_t = jax.lax.dynamic_slice_in_dim(courant, a_j, nl + 1, axis=2)
    c_t = jax.lax.dynamic_slice_in_dim(c_t, a_i, nl, axis=1)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_g, a_j, nl + 2, axis=2)
    rd_t = jax.lax.dynamic_slice_in_dim(rd_t, a_i, nl, axis=1)
    return _ppm_transport_1d(
        vp_t, c_t, rd_t, 2, external_halo=h3, rd_prepadded=True)


def make_tiled_transport_sweep_stage_2d(
    mesh, n: int, kt: int, h3: int = 4, sweep: str = "i",
):
    """Build a sharded PPM transport stage on a ``(6, kt, kt)`` mesh with axis
    names ``("face", "tile_i", "tile_j")``.

    ``sweep="i"`` tiles the i-sweep (xtp_u); ``sweep="j"`` the j-sweep
    (ytp_v).  Inputs are FACE-REPLICATED, pre-padded to the PPM storage halo
    h3 on the SWEEP axis (i: axis=1, j: axis=2).  Output is sharded
    ``P("face","tile_i","tile_j")``; gathered extent
    ``(6, kt*(nl+1), kt*nl)`` for ``i`` / ``(6, kt*nl, kt*(nl+1))`` for ``j``
    — each tile DUPLICATES the shared sweep-interface with its neighbour
    (reassembly: lower tile owns it → the global PPM sweep)."""
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    if sweep not in ("i", "j"):
        raise ValueError(f"sweep must be 'i' or 'j', got {sweep!r}")
    nl = n // kt
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    body = transport_sweep_tile_2d if sweep == "i" else transport_jsweep_tile_2d

    @partial(shard_map, mesh=mesh, in_specs=(fo, fo, fo),
             out_specs=co, check_vma=False)
    def _stage(vp_g, courant, rd_g):
        a_i = jax.lax.axis_index("tile_i") * nl
        a_j = jax.lax.axis_index("tile_j") * nl
        return body(vp_g, courant, rd_g, a_i, a_j, nl, h3)

    return _stage
