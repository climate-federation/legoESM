"""Cross-process agreement checks for per-process-recomputed SPMD geometry.

Every multi-controller SPMD lane faces the same hazard: each process rebuilds
the band/tile geometry from the same config, then hands it to a REPLICATED
``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
bit-identical on every process, and per-process XLA autotuning on
device-derived grid fields makes the last ULPs differ at larger sizes (job
26450848: LL576 np=4, area-scale fields differing at 1e-7 relative), which
trips that assert.

The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
silently paper over a REAL cross-process inconsistency (a different wet
domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
fingerprint must agree first, and a disagreement RAISES.

This module is the ONE implementation of that protocol.  It was extracted
from ``ocean.dynamics.sharded_ocean_step`` (where it was developed and
hardened over five rounds of adversarial review) so the atmosphere lat-lon
lane — which had the identical defect (#1362) — reuses it instead of growing
a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
lanes MUST call these helpers rather than re-derive the fingerprints.

Sequencing contract, in this order:

1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
   fixed-shape collective that every process reaches.  A process-dependent
   field selection (e.g. an optional mask present on some ranks only) would
   otherwise DESYNCHRONIZE the per-field gathers below instead of failing
   with a clear message.
2. :func:`broadcast_checked` per field, in an order identical on every
   process.

NO DEADLOCK RISK: every process fingerprints the same fields in the same
order and derives its verdict from the SAME gathered array, so the refusal is
symmetric — all raise or none.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np

__all__ = [
    "content_hash48",
    "schema_fingerprint",
    "assert_schema_agrees",
    "broadcast_checked",
]

# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
# is expected there; quantize-then-assert-equal false-positived on a rounding
# boundary (job 26453240), so compare with a tolerance instead.
_FLOAT_RTOL = 1e-5


def content_hash48(arr) -> float:
    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.

    Used to compare EXACT-dtype arrays (masks, index tables) across
    processes: unlike moment fingerprints, a byte digest is positional, so a
    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
    under 2**53 so it survives the float64 ``process_allgather`` payload
    exactly. Not cryptographic — collision-resistance at 2**-48 is far
    beyond the ~10 setup-time comparisons this guard makes.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(a.tobytes(), digest_size=6)
    return float(int.from_bytes(h.digest(), "big"))


def schema_fingerprint(names, n_dev) -> np.ndarray:
    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.

    Gathered ONCE before the per-field loop so a process-dependent field
    selection is caught by a collective every process reaches, instead of
    desynchronizing the per-field gathers (codex round-5 findings 3/4).
    """
    joined = ",".join(names).encode()
    digest = float(int.from_bytes(
        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
    return np.array(
        [float(len(names)), digest, float(bool(jax.config.jax_enable_x64)),
         float(n_dev)], dtype=np.float64)


def assert_schema_agrees(names, n_dev, *, context: str) -> None:
    """Raise unless every process agrees on the geometry field SCHEMA.

    ``names`` must be an ORDERED sequence — the per-field
    :func:`broadcast_checked` calls that follow are matched positionally
    across processes, so a reordering is itself a divergence worth catching.

    No-op when ``jax.process_count() == 1``.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    gathered = multihost_utils.process_allgather(
        schema_fingerprint(list(names), n_dev))
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: the band-geometry SCHEMA differs across processes "
            f"(field list / x64 setting / device count — gathered "
            f"{gathered.tolist()}). Fix the per-process config before "
            f"sharding; the per-field checks assume one schema.")


def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.

    Returns a host ``np.ndarray`` that is bit-identical on every process, safe
    to hand to a replicated ``device_put``.  Single-process: returns the host
    view unchanged, no collectives.

    The fingerprint compares structural entries exactly; value entries
    EXACTLY for integer/bool arrays and to ``rtol=1e-5`` for float arrays.

    Integer/bool arrays (masks, index tables) are exact data, not autotuned
    arithmetic: their BYTES are fingerprinted so a positional difference is
    caught.  Moment-only compares are blind to a permutation — a bool mask's
    ``(sum, sumsq, absmax)`` is identical for every arrangement with the same
    true-count (codex round-5).  A mask that genuinely differs across
    processes means different wet domains = different physics: refusing is
    the correct outcome, not a false alarm.

    Residual, documented: a float divergence preserving sum, sum-of-squares
    AND absmax to ``rtol`` is not detected.  Band grids are analytic in
    lat/lon, so any real inconsistency moves those moments.
    """
    host = np.asarray(arr)
    if jax.process_count() <= 1:
        return host
    from jax.experimental import multihost_utils

    flat = host.ravel()
    is_exact = host.dtype.kind in "biu"
    struct = np.array(
        [float(host.ndim), *map(float, host.shape),
         float(np.dtype(host.dtype).num)], dtype=np.float64)
    if is_exact:
        vals = np.array([content_hash48(host)], dtype=np.float64)
    else:
        finite = flat[np.isfinite(flat)]
        f64 = finite.astype(np.float64)
        # Non-finite COUNT is structural: a NaN appearing on one process only
        # must not be averaged away by the moment compare below.
        struct = np.concatenate([struct, [float(flat.size - finite.size)]])
        vals = np.array(
            [float(f64.sum()) if f64.size else 0.0,
             float((f64 * f64).sum()) if f64.size else 0.0,
             float(np.abs(f64).max()) if f64.size else 0.0],
            dtype=np.float64)

    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0],
                                   rtol=_FLOAT_RTOL, atol=0.0))
    if not (struct_ok and vals_ok):
        raise RuntimeError(
            f"{context}: geometry field {name!r} DIVERGES across processes "
            f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
            f"config/grid inconsistency, not autotune noise; refusing to "
            f"broadcast process 0 over it.")
    return np.asarray(multihost_utils.broadcast_one_to_all(host))
