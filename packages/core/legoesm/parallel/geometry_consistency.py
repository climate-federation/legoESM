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
    "coerce_count",
    "coerce_bool",
    "config_digest48",
    "tree_schema_digest48",
    "safe_repr",
    "FLAG_ABSENT",
    "FLAG_UNCOERCIBLE",
    "FLAG_OUT_OF_RANGE",
    "FLAG_NEGATIVE",
    "FLAG_MAX_EXACT",
    "FLAG_DIGEST_FAILED",
]

# --- entry-gate payload sentinels -------------------------------------------
# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
# into a fixed-width float payload.  Building that payload must NEVER raise:
# a rank that dies in `int(n_steps)` while its peers block in
# `process_allgather` is a HANG, which is strictly worse than the bug the gate
# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
# mapped to a SENTINEL that travels through the collective; every rank then
# sees it in the gathered payload and the raise that follows is symmetric.
#
# The sentinels are large-magnitude NEGATIVE values that NO legitimate count
# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
# to be ``-1.0``, so a rank passing ``segment_steps=None`` and a peer passing
# ``-1`` produced the SAME payload entry, agreed, and then diverged downstream
# (codex round-4, blocker 1).  Counts are validated non-negative, so every
# sentinel is unreachable from valid data AND distinct from every other.
FLAG_ABSENT = -6.0e15
FLAG_UNCOERCIBLE = -8.0e15
FLAG_OUT_OF_RANGE = -7.0e15
FLAG_NEGATIVE = -5.0e15
FLAG_DIGEST_FAILED = -4.0e15
# 2**53 is the largest integer whose successor is exactly representable in
# float64.  Above it two DIFFERENT counts alias to the same payload entry, so
# the gate would pass a real divergence (codex round-3, minor 2).  Values past
# the bound are refused rather than silently compared.
FLAG_MAX_EXACT = 2.0 ** 53
_MAX_EXACT_INT = 2 ** 53


def safe_repr(value, limit: int = 120) -> str:
    """``repr(value)`` that cannot raise and cannot blow up the message.

    A user object whose ``__repr__`` raises would otherwise propagate out of
    the payload build — the very pre-collective throw the gates exist to
    remove (codex round-4, blocker 1).
    """
    try:
        text = repr(value)
    except Exception:                       # pragma: no cover - defensive
        try:
            text = f"<unrepresentable {type(value).__name__}>"
        except Exception:                   # pragma: no cover - defensive
            text = "<unrepresentable>"
    return text if len(text) <= limit else text[:limit] + "..."


def coerce_count(value, *, absent: float = FLAG_ABSENT):
    """Map a rank-local COUNT to an exactly-comparable entry-gate payload float.

    Returns ``(payload, problem)``.  ``problem`` is ``None`` when the value is
    usable; otherwise it is a human-readable clause naming the offending value,
    which the caller must raise AFTER its collective so the refusal is
    symmetric across processes.

    This function NEVER raises.  That is the whole point: it is called while
    ASSEMBLING a collective payload, upstream of the collective itself, where a
    raise deadlocks the peers (codex round-3, blocker 3).

    STRICT by type, not by coercibility (codex round-4, blocker 1).  Only a
    real non-negative Python/NumPy integer is accepted:

    * ``3.5`` is REJECTED.  ``int(3.5) == 3`` made a rank carrying ``3.5``
      indistinguishable from a peer carrying ``3``; the payloads agreed and
      then ``range(3.5)`` blew up on one rank alone while its peer entered the
      step collective.
    * ``bool`` is REJECTED.  ``True`` is not a step count, and silently
      encoding it as ``1`` hides a caller bug.
    * Arrays (even size-1) are REJECTED: ``int(arr)`` succeeds for size 1 and
      raises for size > 1, so accepting them makes the gate's behaviour depend
      on rank-local shape.
    * NEGATIVE integers get their OWN sentinel, so they can never collide with
      the "absent" encoding.

    ``None`` maps to ``absent`` (default :data:`FLAG_ABSENT`, itself outside
    the valid range) so a call site that does not carry the value still emits a
    FIXED-WIDTH payload.
    """
    if value is None:
        return float(absent), None
    # `bool` is a subclass of `int`, so it must be excluded FIRST.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return FLAG_UNCOERCIBLE, (
            "must be a non-negative Python/NumPy integer (got type "
            f"{type(value).__name__}: {safe_repr(value)}); every process must "
            "be launched with the same value")
    try:
        as_int = int(value)
    except Exception:                       # pragma: no cover - defensive
        return FLAG_UNCOERCIBLE, (
            f"could not be read as an integer ({safe_repr(value)})")
    if as_int < 0:
        return FLAG_NEGATIVE, (
            f"must be non-negative (got {safe_repr(value)})")
    # Range check on the INTEGER: converting to float first would already have
    # collapsed 2**53+1 onto 2**53, so the very aliasing this guards against
    # would be invisible to the guard.
    if as_int > _MAX_EXACT_INT:
        return FLAG_OUT_OF_RANGE, (
            f"is outside the exactly-comparable range n <= 2**53 (got "
            f"{safe_repr(value)}); beyond that bound two different counts "
            f"alias to the same float64 payload entry and the cross-process "
            f"agreement check would pass a real divergence")
    return float(as_int), None


def coerce_bool(value, *, absent: float = FLAG_ABSENT):
    """Strict, NON-THROWING tri-state encoder for a rank-local BOOLEAN flag.

    Returns ``(payload, problem)`` exactly like :func:`coerce_count`.
    ``None`` -> ``absent`` ("not applicable at this call site").

    Only a real ``bool`` / ``np.bool_`` is accepted.  ``bool(value)`` on an
    arbitrary object RAISES for a multi-element array ("truth value of an array
    is ambiguous") — inside a gate that is a pre-collective throw, i.e. a hang
    (codex round-4, blocker 2).  Anything else becomes a sentinel that travels
    through the collective and is refused symmetrically afterwards.
    """
    if value is None:
        return float(absent), None
    if isinstance(value, (bool, np.bool_)):
        return (1.0 if value else 0.0), None
    return FLAG_UNCOERCIBLE, (
        f"must be a bool (got type {type(value).__name__}: "
        f"{safe_repr(value)})")


def _canonical_config_terms(obj, prefix: str = "", depth: int = 0,
                            out=None, seen=None):
    """Flatten a config object into ORDERED ``"path=value"`` strings.

    Covers the STATIC scalars that select a compiled program: scheme literals,
    integrator names, and every feature-gating bool (``fix_mass``,
    ``fix_moisture``, ``use_polar_filter``, ...).  Arrays contribute only
    ``dtype`` + ``shape`` — comparing their VALUES is the job of
    :func:`broadcast_checked`, not of a cheap fixed-width entry gate.

    Never raises: any unreadable field becomes a ``<unreadable>`` term, which
    still participates in the comparison.
    """
    if out is None:
        out, seen = [], set()
    if depth > 4 or len(out) > 512:          # bounded work, bounded payload
        return out
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    fields = getattr(obj, "_fields", None)   # NamedTuple
    if fields is None:
        dc = getattr(obj, "__dataclass_fields__", None)
        fields = tuple(dc) if dc else None
    if fields is None:
        return out
    for name in fields:
        try:
            val = getattr(obj, name)
        except Exception:                    # pragma: no cover - defensive
            out.append(f"{prefix}{name}=<unreadable>")
            continue
        path = f"{prefix}{name}"
        if val is None or isinstance(val, (bool, int, float, str, np.bool_,
                                           np.integer, np.floating)):
            out.append(f"{path}={safe_repr(val, 64)}")
        elif hasattr(val, "dtype") and hasattr(val, "shape"):
            out.append(f"{path}=array:{safe_repr(val.dtype, 32)}:"
                       f"{safe_repr(tuple(val.shape), 64)}")
        elif getattr(val, "_fields", None) or getattr(
                val, "__dataclass_fields__", None):
            _canonical_config_terms(val, path + ".", depth + 1, out, seen)
        else:
            out.append(f"{path}=<{type(val).__name__}>")
    return out


def config_digest48(obj) -> float:
    """One fixed-width, order-sensitive digest of a config's STATIC scalars.

    Why a digest instead of a hand-listed set of flags: an entry gate that
    enumerates ``fold``/``anchor``/``polar`` by hand agrees only the fields
    somebody remembered.  ``fix_mass`` gates a global-area psum,
    ``outer_integrator`` selects a different program, ``fix_moisture`` adds a
    reduction — each was MISSING from the hand-written list (codex round-4,
    blocker 3).  Digesting every static scalar closes the class instead of the
    three instances, and costs ONE payload entry.

    Never raises; an internal failure returns :data:`FLAG_DIGEST_FAILED`,
    which still compares equal across ranks that fail identically and unequal
    against a rank that succeeded.
    """
    try:
        return name_digest48(_canonical_config_terms(obj))
    except Exception:                        # pragma: no cover - defensive
        return FLAG_DIGEST_FAILED


def tree_schema_digest48(tree) -> float:
    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.

    The gather/scatter entry points run one cross-process replication PER
    NON-``None`` LEAF, so the NUMBER and ORDER of those collectives is
    rank-local data: a state whose tracer dict differs across processes (extra
    species, different insertion order, different shape) produces mismatched
    schedules and hangs (codex round-4, blocker 5).  Folding the whole leaf
    schema into ONE fixed-width float makes that a clean symmetric raise.

    ``jax.tree_util`` key paths give a canonical, ORDER-SENSITIVE description
    (dict keys are sorted by ``tree_flatten_with_path``, so an insertion-order
    difference alone does not false-positive, while a KEY-SET difference does
    move the digest).  Never raises.
    """
    try:
        from jax.tree_util import tree_flatten_with_path, keystr
        leaves, _ = tree_flatten_with_path(tree)
        terms = []
        for path, leaf in leaves:
            dtype = getattr(leaf, "dtype", None)
            shape = getattr(leaf, "shape", None)
            terms.append(
                f"{keystr(path)}:{safe_repr(dtype, 32)}:"
                f"{safe_repr(tuple(shape) if shape is not None else None, 64)}")
        return name_digest48(terms)
    except Exception:                        # pragma: no cover - defensive
        return FLAG_DIGEST_FAILED


def _dtype_kind_and_ndim(a):
    """``(kind, ndim)`` for the schema digest, tolerant of a plain scalar.

    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
    both without materialising, so no device sync).  A plain Python scalar or
    list has neither; falling through to ``np.asarray`` there is FREE (it is
    already host data) and, critically, keeps this function from dying with a
    bare ``AttributeError`` BEFORE :func:`assert_schema_agrees` reaches its
    collective — a rank-local raise ahead of a collective is a HANG, so
    "fail explicitly" here must NOT mean "raise here" (codex round-3, minor 3).

    An unsupported dtype class (object/str) is reported as ``unsupported:<k>``
    rather than being silently bucketed with the float fields, so a
    disagreement about it is visible in the digest and a same-on-all-ranks
    unsupported field fails later in :func:`broadcast_checked` with its own
    message instead of here.
    """
    dtype = getattr(a, "dtype", None)
    if dtype is None:
        host = np.asarray(a)
        dtype, ndim = host.dtype, host.ndim
    else:
        ndim = int(getattr(a, "ndim", np.ndim(a)))
    k = np.dtype(dtype).kind
    if k in "biu":
        kind = "exact"
    elif k in "fc":
        kind = "inexact"
    else:
        kind = f"unsupported:{k}"
    return kind, int(ndim)


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
        #
        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
        # bare `a.dtype` read did not (codex round-3, minor 3): no production
        # caller passes one today, but an AttributeError HERE would be a
        # rank-local raise BEFORE the collective, i.e. a hang rather than a
        # clear failure.
        described = [_dtype_kind_and_ndim(a) for a in arrays]
        kinds = [d[0] for d in described]
        ndims = [d[1] for d in described]
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
