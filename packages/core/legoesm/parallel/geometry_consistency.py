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
    "name_digest48",
    "schema_fingerprint",
    "assert_schema_agrees",
    "assert_flags_agree",
    "broadcast_checked",
]

# Every per-field collective payload is padded to these FIXED widths.  A
# payload whose LENGTH depends on rank-local data (dtype class, ndim,
# non-finite count) would let two processes enter `process_allgather` with
# different shapes and DEADLOCK -- the exact failure this module exists to
# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
# inherited from the pre-extraction ocean implementation, so fixing it here
# fixes BOTH lanes).
_STRUCT_WIDTH = 8
_VALS_WIDTH = 3

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


def name_digest48(names) -> float:
    """Order-sensitive, UNAMBIGUOUS digest of a sequence of names.

    Uses a NUL separator, which cannot occur in a Python identifier or any
    legoESM field name, so ``["a,b", "c"]`` and ``["a", "b,c"]`` cannot
    collide.  A plain ``",".join`` COULD (codex 2026-07-29, minor 5): those
    two lists have the same length, so a count check does not separate them
    either.
    """
    joined = "\x00".join(names).encode()
    return float(int.from_bytes(
        hashlib.blake2b(joined, digest_size=6).digest(), "big"))


def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
    """Fixed-shape schema digest gathered ONCE before the per-field loop.

    Covers the field-name list (order-sensitive), the count, the x64 flag,
    ``n_dev``, and -- critically -- the per-field DTYPE CLASS and NDIM.

    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
    exact dtypes to a 1-value digest and float dtypes to a 3-moment
    fingerprint, and its struct entry depends on ndim.  If the schema gate
    did not cover those, a field that is bool on one process and float on
    another would PASS the gate and then deadlock inside the per-field
    gather with mismatched payloads.  Catching it here converts that hang
    into a clean symmetric RuntimeError (codex 2026-07-29, blocker 2).

    ``dtype_kinds``/``ndims`` default to empty for callers that have not yet
    resolved the arrays; passing them is strongly preferred.
    """
    return np.array(
        [float(len(names)),
         name_digest48(names),
         float(bool(jax.config.jax_enable_x64)),
         float(n_dev),
         name_digest48([str(k) for k in dtype_kinds]),
         name_digest48([str(int(n)) for n in ndims])],
        dtype=np.float64)


def assert_flags_agree(names, values, *, context: str) -> None:
    """Raise unless every process agrees on a tuple of rank-local CONFIG flags.

    Call this BEFORE any rank-local ``raise`` that inspects per-process
    config.  Otherwise one process can reject its config and exit while its
    peers proceed into a collective and block forever — a collective-ORDER
    violation whose symptom (hang vs backend error) is backend-dependent
    (codex 2026-07-29, blocker 1).

    ``names`` and ``values`` must be STATIC tuples written at the call site,
    so the payload length is fixed by the code path rather than by data.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    payload = np.array(
        [float(len(values)), name_digest48(names),
         *(float(v) for v in values)], dtype=np.float64)
    gathered = multihost_utils.process_allgather(payload)
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: per-process CONFIG differs across processes "
            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
            f"process must be built from the same config; refusing before "
            f"any rank-local rejection so the failure is symmetric rather "
            f"than a hang.")


def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
    """Raise unless every process agrees on the geometry field SCHEMA.

    ``names`` must be an ORDERED sequence — the per-field
    :func:`broadcast_checked` calls that follow are matched positionally
    across processes, so a reordering is itself a divergence worth catching.

    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
    bool-vs-float disagreement slip past this gate and deadlock in the
    per-field gather instead of raising here.

    No-op when ``jax.process_count() == 1``.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    names = list(names)
    if arrays is None:
        kinds, ndims = (), ()
    else:
        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
        # array exposes both without materialising, so forcing a host copy
        # here would add a device sync per field AND could itself fail
        # (transfer error / OOM) BEFORE the collective below — reintroducing
        # the very "one rank exits while a peer blocks" hazard this gate
        # exists to remove (codex round-2 minor). `broadcast_checked` does
        # the single real materialisation later.
        kinds = [("exact" if np.dtype(a.dtype).kind in "biu" else "inexact")
                 for a in arrays]
        ndims = [int(getattr(a, "ndim", np.ndim(a))) for a in arrays]
    gathered = multihost_utils.process_allgather(
        schema_fingerprint(names, n_dev, kinds, ndims))
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: the band-geometry SCHEMA differs across processes "
            f"(field list / x64 setting / device count / per-field dtype "
            f"class / ndim — gathered {gathered.tolist()}). Fix the "
            f"per-process config before sharding; the per-field checks "
            f"assume one schema.")


def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.

    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
    every process, safe to hand to a replicated ``device_put``.
    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
    host round trip, no dtype/weak-type change.

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
    # EARLY return, BEFORE np.asarray: single process has nothing to compare,
    # and converting here would force a device->host->device round trip and
    # strip weak-type metadata on a 1-process mesh. The ocean lane already
    # held host arrays so it was unaffected, but the atmosphere lane passes
    # `jnp.stack` results straight in and WAS regressed by an unconditional
    # conversion (codex 2026-07-29, major 3). Return the caller's object
    # untouched.
    if jax.process_count() <= 1:
        return arr
    from jax.experimental import multihost_utils

    host = np.asarray(arr)
    flat = host.ravel()
    is_exact = host.dtype.kind in "biu"
    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
    # must never depend on rank-local data, or two processes can enter this
    # collective with different shapes and hang. Shape is folded in as a
    # digest rather than splatted, so an ndim difference cannot change the
    # length either.
    struct = np.zeros(_STRUCT_WIDTH, dtype=np.float64)
    struct[0] = float(host.ndim)
    struct[1] = float(np.dtype(host.dtype).num)
    struct[2] = float(1.0 if is_exact else 0.0)
    struct[3] = float(host.size)
    struct[4] = name_digest48([str(d) for d in host.shape])
    vals = np.zeros(_VALS_WIDTH, dtype=np.float64)
    if is_exact:
        vals[0] = content_hash48(host)
    else:
        finite = flat[np.isfinite(flat)]
        f64 = finite.astype(np.float64)
        # Non-finite COUNT is structural: a NaN appearing on one process only
        # must not be averaged away by the moment compare below.
        struct[5] = float(flat.size - finite.size)
        vals[0] = float(f64.sum()) if f64.size else 0.0
        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0

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
