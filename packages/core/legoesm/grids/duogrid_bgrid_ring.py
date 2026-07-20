"""FAITHFUL B-grid ring-1 cross-face halo for the JAX FB core.

The forward-backward core's ``d_sw5_corner_divergence`` needs a one-ring
cross-face ghost of the B-grid corner scalar ``divg_d``.  Its two
existing options are approximations (zero-ring ghost; nearest-row copy
that skips the cube_rmp Lagrange remap — see
``_pad_corner_scalar_cross_face``).  The FAITHFUL exchange is the
certified numpy ``ext_scalar_sixface(f6, "B", ectx)`` (mpp index-copy +
k2e Lagrange ring remap + corner-region Lagrange —
``fv3_native_ext_vector``), which is pure numpy/Python loops and cannot
run inside ``jit``.

This module turns that exchange into a STATIC LINEAR MAP:

- ``build_bgrid_ring1_map(n)`` probes the certified exchange with unit
  impulses (linearity: the exchange is exactly linear in the field) and
  records, for every ring-1 destination slot, its source
  ``(face, i, j)`` compute-node weights.  Faithfulness is BY
  CONSTRUCTION — the weights come from running the certified code, not
  from re-deriving any Lagrange math.  The map is grid-static, built
  once per ``n`` and cached to disk (numpy npz).
- ``apply_bgrid_ring1(field, ring_map)`` applies the map inside jitted
  code as a gather + weighted sum, producing the ``(6, n+3, n+3)``
  padded array ``d_sw5`` consumes (compute block + faithful ring;
  pad-diagonal corners keep the edge-copy — the d_sw5 gradient stencil
  never reads them, matching the existing options' convention).

Only sources within ``_BAND`` of a face edge can influence the ring
(k2e nord=4 stencil reach + corner Lagrange); the build asserts
completeness against the certified exchange on a random field.
"""
from __future__ import annotations

import os

import numpy as np

# influencing band: k2e nord-4 along-edge stencil (6 wide) + corner
# Lagrange reach (4) with margin
_BAND = 8

# cache format version: v2 = CREATE-layout native map (codex bgring-r1
# P0: v1 probed in FV3 reference tile layout while the FB core runs
# create_cubed_sphere layout — raw application error 3.158; the map is
# now conjugated by the ED face perm/rot at build time)
_MAP_VERSION = "create-ed-v2"

_CACHE: dict = {}


def _layout_transforms():
    """FV3-reference <-> create face layout (cubed_sphere canonical)."""
    from legoesm.grids.cubed_sphere import (
        _GNOMONIC_ED_FACE_PERM as PERM,
        _GNOMONIC_ED_FACE_ROT as ROT,
    )

    def create_to_ref(field6: np.ndarray) -> np.ndarray:
        out = np.zeros_like(field6)
        for F in range(6):
            out[PERM[F]] = np.rot90(field6[F], -ROT[F])
        return out

    def ref_window_to_create(win6: np.ndarray) -> np.ndarray:
        """(6, m, m) fort windows in ref layout -> create layout."""
        out = np.zeros_like(win6)
        for F in range(6):
            out[F] = np.rot90(win6[PERM[F]], ROT[F])
        return out

    return create_to_ref, ref_window_to_create


def _cache_dir() -> str:
    d = os.environ.get(
        "LEGOESM_BGRID_RING_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache",
                     "legoesm_bgrid_ring"))
    os.makedirs(d, exist_ok=True)
    return d


def _run_certified_exchange(field6: np.ndarray, n: int, ng: int,
                            ectx: dict) -> np.ndarray:
    """Certified ext_scalar B on a CREATE-layout FB field; returns
    ring-1 values IN CREATE LAYOUT.

    ``field6``: (6, n+1, n+1) compute-only B arrays, create face
    layout.  Internally: create->reference layout, the certified
    reference-layout exchange, then the (compute + ring-1) window back
    to create layout (codex bgring-r1 P0 conjugation).  Returns
    (6, 4, n+1): per create-face the four ring strips (west i=0, east
    i=npx+1, south j=0, north j=npx+1) over Fortran nodes 1..npx.
    """
    from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface

    create_to_ref, ref_window_to_create = _layout_transforms()
    m_b = n + 2 * ng + 1
    npx = n + 1
    ref_field = create_to_ref(field6)
    f6 = [np.zeros((m_b, m_b)) for _ in range(6)]
    sl = slice(ng, ng + npx)
    for t in range(6):
        f6[t][sl, sl] = ref_field[t]
    ext_scalar_sixface(f6, "B", ectx)
    # fort window rows/cols 0..npx+1 (compute + ring-1), per ref face
    wsl = slice(ng - 1, ng + npx + 1)
    ref_win = np.stack([f6[t][wsl, wsl] for t in range(6)])
    win = ref_window_to_create(ref_win)         # (6, npx+2, npx+2)
    out = np.zeros((6, 4, npx))
    out[:, 0] = win[:, 0, 1:npx + 1]            # west ring
    out[:, 1] = win[:, npx + 1, 1:npx + 1]      # east ring
    out[:, 2] = win[:, 1:npx + 1, 0]            # south ring
    out[:, 3] = win[:, 1:npx + 1, npx + 1]      # north ring
    return out


def build_bgrid_ring1_map(n: int, ng: int = 3, *,
                          use_disk_cache: bool = True) -> dict:
    """Extract the faithful ring-1 linear map by impulse probing.

    Returns numpy arrays (jnp-converted at apply time):
      ``starts``  (6*4*(n+1)+1,) int32 CSR row starts per dest slot,
      ``src_face``/``src_i``/``src_j`` (K,) int32 source compute nodes,
      ``w`` (K,) float64 weights.
    Dest slot order: face-major, then strip (W,E,S,N), then position.
    """
    if n < 2 * _BAND:
        raise ValueError(
            f"build_bgrid_ring1_map: n={n} too small for the probe "
            f"band (_BAND={_BAND}); the certified probe context also "
            "needs ng=3 <= n//2")
    key = (_MAP_VERSION, n, ng)
    if key in _CACHE:
        return _CACHE[key]
    path = None
    if use_disk_cache:
        path = os.path.join(
            _cache_dir(), f"bgrid_ring1_{_MAP_VERSION}_n{n}_ng{ng}.npz")
    if use_disk_cache and os.path.exists(path):
        z = np.load(path)
        m = {k: z[k] for k in ("starts", "src_face", "src_i", "src_j",
                               "w")}
        _CACHE[key] = m
        return m

    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True)
    ectx = ctx["ectx"]

    npx = n + 1
    n_dst = 6 * 4 * npx
    cols: list[list] = [[] for _ in range(n_dst)]

    # probe only edge-band sources (the ring cannot see deeper nodes);
    # completeness is asserted below on a random field
    band_nodes = [(i, j) for i in range(npx) for j in range(npx)
                  if (i < _BAND or i >= npx - _BAND
                      or j < _BAND or j >= npx - _BAND)]
    for face in range(6):
        for (i, j) in band_nodes:
            probe = np.zeros((6, npx, npx))
            probe[face, i, j] = 1.0
            ring = _run_certified_exchange(probe, n, ng, ectx)
            nz = np.argwhere(np.abs(ring) > 1e-300)
            for (t, s, p) in nz:
                dst = (t * 4 + s) * npx + p
                cols[dst].append((face, i, j, ring[t, s, p]))

    starts = np.zeros(n_dst + 1, dtype=np.int32)
    src_face, src_i, src_j, w = [], [], [], []
    for d in range(n_dst):
        starts[d + 1] = starts[d] + len(cols[d])
        for (f, i, j, ww) in cols[d]:
            src_face.append(f)
            src_i.append(i)
            src_j.append(j)
            w.append(ww)
    m = {"starts": starts,
         "src_face": np.array(src_face, dtype=np.int32),
         "src_i": np.array(src_i, dtype=np.int32),
         "src_j": np.array(src_j, dtype=np.int32),
         "w": np.array(w, dtype=np.float64)}

    # completeness tripwire: random field, full certified exchange vs
    # the extracted map (catches an under-sized _BAND loudly)
    rng = np.random.default_rng(0)
    test = rng.standard_normal((6, npx, npx))
    want = _run_certified_exchange(test, n, ng, ectx)
    got = _apply_numpy(test, m, n)
    err = float(np.max(np.abs(got - want)))
    if err > 1e-12:
        raise AssertionError(
            f"bgrid ring-1 map incomplete (err {err:.3e}) — widen _BAND")

    if use_disk_cache:
        np.savez_compressed(path, **m)
    _CACHE[key] = m
    return m


def _apply_numpy(field6: np.ndarray, m: dict, n: int) -> np.ndarray:
    """Reference numpy application (tests / completeness tripwire)."""
    npx = n + 1
    flat = field6.reshape(6, -1)
    src = flat[m["src_face"], m["src_i"] * npx + m["src_j"]] * m["w"]
    out = np.zeros(6 * 4 * npx)
    for d in range(6 * 4 * npx):
        a, b = m["starts"][d], m["starts"][d + 1]
        out[d] = src[a:b].sum()
    return out.reshape(6, 4, npx)


def apply_bgrid_ring1(field, ring_map: dict, n: int):
    """Faithful ring-1 pad inside jitted code.

    ``field``: (6, n+1, n+1) jnp B-grid scalar.  Returns
    (6, n+3, n+3): edge-copy base pad with the four ring strips
    overwritten by the faithful cross-face values.  Segment-sum over the
    CSR map keeps everything shape-static for jit.
    """
    import jax.numpy as jnp

    npx = n + 1
    n_dst = 6 * 4 * npx
    starts = ring_map["starts"]
    # per-entry destination ids (static numpy -> constant under jit)
    dst_ids = np.repeat(np.arange(n_dst, dtype=np.int32),
                        np.diff(starts))
    src_face = jnp.asarray(ring_map["src_face"])
    src_flat = jnp.asarray(ring_map["src_i"].astype(np.int64) * npx
                           + ring_map["src_j"])
    wgt = jnp.asarray(ring_map["w"], dtype=field.dtype)

    flat = field.reshape(6, -1)
    contrib = flat[src_face, src_flat] * wgt
    import jax
    ring = jax.ops.segment_sum(contrib, jnp.asarray(dst_ids),
                               num_segments=n_dst)
    ring = ring.reshape(6, 4, npx)

    padded = jnp.pad(field, [(0, 0), (1, 1), (1, 1)], mode="edge")
    padded = padded.at[:, 0, 1:npx + 1].set(ring[:, 0])       # west
    padded = padded.at[:, npx + 1, 1:npx + 1].set(ring[:, 1])  # east
    padded = padded.at[:, 1:npx + 1, 0].set(ring[:, 2])       # south
    padded = padded.at[:, 1:npx + 1, npx + 1].set(ring[:, 3])  # north
    return padded
