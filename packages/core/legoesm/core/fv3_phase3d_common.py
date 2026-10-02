"""Entry gates and assembly helpers shared by the 3-D duo phase modules.

The Phase-2 (3-D) layer of the FV3 duo JAX lane is a stack of phase
modules -- :mod:`legoesm.core.fv3_cgrid_phase_3d` (C-grid half step),
:mod:`legoesm.core.fv3_dsw_phase_3d` (D-grid transport + barrier 1),
:mod:`legoesm.core.fv3_dsw_tail_3d` (barrier 2 + the D tail) -- that all
take the same face-stacked containers and therefore all need the same
five entry gates and two assembly helpers.

Those seven had been copied privately into each module, because a
private (``_``-prefixed) cross-module import is banned by the
empty-allowlist ratchet ``tests/test_no_private_cross_imports.py``.  The
copies carried their own FOLLOW-UP note asking for exactly this
promotion ("this is now the fifth"); this module is it.  Nothing here is
new behaviour -- each function is the union of the previously duplicated
copies, and where the copies differed the SUPERSET was taken:

* :func:`require_nord` takes the parameter ``name`` (the C-grid copy
  hard-coded ``"nord"``), so a module with two damping orders can say
  which one is wrong.
* :func:`stack_levels` accepts an optional ``want2d``: given, every
  level is checked against that DECLARED shape (the C-grid copy's
  behaviour, which is the stricter one); omitted, level 0's shape is the
  reference (the D-grid copy's, used where no shape table exists).
* :func:`validate_stacked` maps names through :data:`CSW_OUT_LIKE`
  before calling ``field_shape``; the mapping is the identity for every
  name outside it, so this is a superset of the un-mapped copy.

:data:`CSW_OUT_LIKE` lives here rather than in the C-grid module for an
import-cycle reason, not a taxonomic one: :func:`validate_stacked` needs
it, and every phase module imports this one.  ``fv3_cgrid_phase_3d``
re-exports it so existing importers do not move.

WHAT THIS MODULE IS NOT
-----------------------
It holds no numerics and no oracle citations.  Every function here is a
GATE (raises, or returns its input unchanged) or a pure index copy
(:func:`stack_levels` / :func:`stack_faces` are ``jnp.stack``).  A
routine that computes a physical quantity does not belong here; it
belongs in the kernel module that owns it.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.core.fv3_native_state_3d import field_shape

__all__ = [
    "CSW_OUT_LIKE",
    "PER_FACE_FLAG_FIELDS",
    "batch_size",
    "build_batched_gs",
    "require_f64_jax",
    "require_uniform_float_jax",
    "require_bool",
    "require_km",
    "require_nord",
    "validate_stacked",
    "stack_levels",
    "stack_levels_batched",
    "stack_faces",
]

# Which field's shape each c_sw output shares.  Re-stated rather than
# imported because the spec's copy (`_out_like` in
# `fv3_native_cgrid_phase_3d`) is private and a private cross-module
# import is banned by the empty-allowlist ratchet
# `tests/test_no_private_cross_imports.py`.  Pinned against the spec's
# copy by `test_csw_out_like_matches_the_spec`, so the two cannot drift.
CSW_OUT_LIKE = {
    "divg_d": "divgd", "uc": "uc", "vc": "vc",
    "delpc": "delp", "ptc": "pt", "ua": "ua", "va": "va",
    "ut": "ut", "vt": "vt", "wc": "w",
    # `pkc` is the C-stage FULL interface pressure, allocated
    # `(m_a, m_a, km+1)` at fv3_native_cgrid_phase_3d.py:253 -- the same
    # shape `field_shape` declares for `pk`/`gz`.  Added because the NH
    # D-grid tail validates `csw_press` through this table and the
    # lookup raised "unknown field 'pkc'" (job 9417536); the message was
    # right that the fix belongs HERE and not in an undeclared shape at
    # the call site.
    "pkc": "pk",
}


# ---------------------------------------------------------------------
# face-batched context view (face-batching ladder step 1)
# ---------------------------------------------------------------------

# The ONLY GridFlags fields that legitimately differ per face.  They are
# used ARITHMETICALLY inside the kernels (``(damp_c*da_min)**(nord+1)``
# and ``(da_min_c*d4_bg)**(nord+1)``), so a face-batched arm may carry
# them as ``(6,)`` arrays; every OTHER field is a STATIC Python branch
# selector, and :func:`build_batched_gs` REFUSES a context on which one
# of those differs across faces -- silently broadcasting face 1's value
# to all six would be plausible wrong physics, not an error message.
PER_FACE_FLAG_FIELDS = ("da_min", "da_min_c")


def batch_size(ctx) -> int:
    """Length of the leading batch axis this ctx steps: 6 for the cube's
    faces, ``6*kt*kt`` for a window ctx (fv3_duo_windows).  Read off
    ``ctx.gs6``; a context without gridstructs (the shape-validator tests'
    stubs) is the cube."""
    gs6 = getattr(ctx, "gs6", None)
    return 6 if gs6 is None else len(gs6)


def build_batched_gs(ctx) -> dict:
    """Face-BATCHED view of ``ctx.gs6`` / ``ctx.flags6``, built once.

    The 3-D phases' certified path is a Python ``for t in range(6)``
    over per-face kernel calls.  Under SPMD that loop is the measured
    2-GPU wall (every device computes all six faces, and each traced
    ``x[t]`` read costs a masked select + all-reduce); the fix is to
    ``jax.vmap`` the per-face kernel over a leading ``(6, ...)`` batch
    axis, which GSPMD then partitions with zero communication.  This
    function provides the stacked operands that vmap arm consumes:

    ``"gs"``
        ``{key: (6, ...)}`` -- the six per-face metric dicts stacked on
        a new leading face axis, key by key.  ``np.stack`` on the HOST (a
        cached jnp.stack would trap tracers) is a pure
        index copy, so ``view["gs"][key][t]`` equals ``ctx.gs6[t][key]``
        EXACTLY (pinned by a round-trip test).  Only keys whose shape is
        identical on all six faces are stacked; the rest are recorded in
        ``"unstacked_keys"`` and stay loop-path-only -- a vmapped kernel
        that needs one fails LOUDLY with a ``KeyError`` naming it, never
        with face-1 values broadcast.  A per-face KEY split raises here
        (fail closed): stacking different metrics into one batch slot
        would be silent.
    ``"da_min6"``, ``"da_min_c6"``
        ``(6,)`` float64 arrays from ``flags6[t].da_min`` /
        ``.da_min_c`` -- the two per-face flag fields, batchable because
        kernels use them arithmetically (:data:`PER_FACE_FLAG_FIELDS`).
    ``"flags"``
        Dict of every OTHER ``GridFlags`` field, single shared value.
        Guarded by the common-mode assert: any of those fields differing
        across the six faces RAISES naming the field, because they are
        static branch selectors and a batched arm can only pass ONE
        value to the kernel.
    ``"unstacked_keys"``
        Tuple of gs keys excluded from ``"gs"`` for shape mismatch.

    CACHED on the context (``ctx._batched_gs``) so the stack runs once
    per context, not once per phase call.  The cache is validated by
    IDENTITY of ``ctx.gs6`` / ``ctx.flags6`` -- a clone that swaps in a
    fresh ``flags6`` tuple (the test helpers do) gets a fresh view, and
    a ctx without the cache slot still gets a correct, merely uncached,
    view.
    """
    cached = getattr(ctx, "_batched_gs", None)
    if cached is not None:
        src_gs, src_flags, view = cached
        if src_gs is ctx.gs6 and src_flags is ctx.flags6:
            return view

    gs6, flags6 = ctx.gs6, ctx.flags6
    # nb = the batch axis: the cube's six faces, or the tiled port's
    # 6*kt*kt sub-face windows (fv3_duo_windows.build_window_ctx)
    nb = batch_size(ctx)
    if len(flags6) != nb:
        raise ValueError(
            f"build_batched_gs: ctx carries {len(gs6)} gridstructs and "
            f"{len(flags6)} flag sets -- one of each per batch member")

    keys0 = set(gs6[0])
    for t in range(1, nb):
        if set(gs6[t]) != keys0:
            missing = sorted(keys0 - set(gs6[t]))
            extra = sorted(set(gs6[t]) - keys0)
            raise KeyError(
                f"build_batched_gs: gs6[{t}] key set differs from "
                f"gs6[0]'s (missing {missing}, extra {extra}); a "
                f"per-face key split would stack a different metric "
                f"into the same batch slot")

    stacked, unstacked = {}, []
    for key in sorted(keys0):
        shapes = {tuple(np.shape(gs6[t][key])) for t in range(nb)}
        if len(shapes) != 1:
            unstacked.append(key)
            continue
        # np.stack, NOT jnp: gs6 holds HOST numpy metric constants, and
        # the view is CACHED on the ctx.  A jnp.stack executed inside a
        # jit trace would cache TRACERS, and the next trace's reuse is
        # an UnexpectedTracerError (measured: the composed step's first
        # call built the cache in-trace, job 9503200).  Host arrays are
        # constants in every trace that closes over them.
        stacked[key] = np.stack([np.asarray(gs6[t][key])
                                 for t in range(nb)], axis=0)

    f0 = flags6[0]
    for name in f0._fields:
        if name in PER_FACE_FLAG_FIELDS:
            continue
        vals = [getattr(flags6[t], name) for t in range(nb)]
        if any(v != vals[0] for v in vals):
            raise ValueError(
                f"build_batched_gs: GridFlags.{name} differs across "
                f"faces ({vals}). It is a STATIC branch selector, so a "
                f"face-batched kernel call can only take one value -- "
                f"broadcasting face 1's would silently run face 1's "
                f"branch on all six faces. Only "
                f"{list(PER_FACE_FLAG_FIELDS)} may vary per face.")

    view = {
        "gs": stacked,
        "unstacked_keys": tuple(unstacked),
        # np, not jnp: same cached-tracer hazard as the gs stack above.
        "da_min6": np.asarray([fl.da_min for fl in flags6],
                              dtype=np.float64),
        "da_min_c6": np.asarray([fl.da_min_c for fl in flags6],
                                dtype=np.float64),
        "flags": {name: getattr(f0, name) for name in f0._fields
                  if name not in PER_FACE_FLAG_FIELDS},
    }
    try:
        ctx._batched_gs = (gs6, flags6, view)
    except (AttributeError, TypeError):
        pass  # a ctx without the cache slot gets a correct uncached view
    return view


# ---------------------------------------------------------------------
# entry gates
# ---------------------------------------------------------------------

def require_uniform_float_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate: every operand of a phase shares ONE floating
    dtype, and it is float32 or float64.

    WAS a strict-float64 gate; RELAXED (2026-08-28) for the coarse
    fv3_duo precision policy (``FV3DuoConfig.storage_dtype``). The
    original guarantee it protected -- "no silent float32 downcast of a
    run that intends fp64" -- now lives at the MODEL BOUNDARY
    (``FV3DuoDynamicsModel`` asserts the IC dtype equals the configured
    storage dtype). What THIS gate now catches is the harder mixed-
    precision failure: an f64 grid metric / workspace leaking into an
    f32 phase (or vice-versa), which JAX would silently promote to f64
    -- defeating the fp32 run and, under ``lax.scan``, raising a carry
    dtype mismatch far from the cause. A single-precision-uniform phase
    is the invariant.

    Reads only ``.dtype`` (static under jit). The certified fp64 deck is
    unchanged: every operand is float64, uniform, so this passes exactly
    where the old gate did.
    """
    seen = None
    for name, a in arrays.items():
        if a is None:
            continue
        _arr = jnp.asarray(a)
        if _arr.ndim == 0 and getattr(_arr, "weak_type", False):
            # Skip ONLY a WEAK-typed 0-dim scalar (a python-float
            # timestep/coeff like dt/kgb): it is weak-promoting and not a
            # field, so it is not part of the field uniformity invariant.
            # A STRONG-f64 0-dim (an f64 constant / damping coeff that
            # "went strong") is NOT skipped -> it still trips this gate
            # against f32 fields, closing the silent-promotion blind spot
            # a wholesale 0-dim skip left (codex+GLM+Claude, increment 2).
            continue
        dt = _arr.dtype
        if dt not in (jnp.float32, jnp.float64):
            raise TypeError(
                f"{fname}: {name} must be float32 or float64 (got {dt}); "
                f"the duo lane runs a single uniform float dtype "
                f"(FV3DuoConfig.storage_dtype).")
        if seen is None:
            seen = dt
        elif dt != seen:
            raise TypeError(
                f"{fname}: MIXED float dtypes -- {name} is {dt} but an "
                f"earlier operand was {seen}. A phase must be single-"
                f"precision-uniform; an f64 metric/workspace leaking into "
                f"an f32 phase would silently promote to f64 (and break a "
                f"lax.scan carry). Downcast the odd operand to the run's "
                f"storage dtype (FV3DuoConfig.storage_dtype).")


# Back-compat alias: the historical name is used at ~40 call sites. Keep it
# pointing at the generalised gate so the certified fp64 path is
# byte-identical and no call site churns.
require_f64_jax = require_uniform_float_jax


def require_bool(fname: str, name: str, value) -> None:
    """A truthy non-bool would silently select a branch.

    ``remap_follows`` in particular is a PROMISE that the vertical remap
    runs later; a string sentinel or a stray ``1`` must not license a
    deformed-``delp`` state.  ``hydrostatic`` chooses which arms of the
    D-grid stages run at all, i.e. what the barriers are handed.
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"{fname}: {name} must be a bool, got "
            f"{type(value).__name__} ({value!r})")


def require_km(fname: str, km) -> int:
    """``km`` is a Python trip count and a shape -- never traced."""
    if isinstance(km, bool) or not isinstance(km, (int, np.integer)):
        raise TypeError(
            f"{fname}: km must be a Python int (it is a loop trip count "
            f"and an array extent, so it cannot be traced), got "
            f"{type(km).__name__} ({km!r})")
    if km < 1:
        raise ValueError(f"{fname}: km must be >= 1, got {km}")
    return int(km)


def require_nord(fname: str, name: str, nord) -> int:
    """A damping ORDER is integral by construction.

    ``int()`` on 2.7 would round to 2 without a word, and both
    ``c_sw`` and ``exchange_post_pgrad_sixface`` branch on ``nord > 0``
    -- so a float here silently selects a different set of divergence
    terms, or whether the ``divgd`` halo is exchanged at all.  Same
    guard, same reason, as the km=1 lane's.
    """
    if isinstance(nord, bool) or nord != int(nord):
        raise ValueError(
            f"{fname}: {name} must be an integral damping order, got "
            f"{nord!r}")
    nord = int(nord)
    if nord < 0:
        raise ValueError(f"{fname}: {name} must be >= 0, got {nord}")
    return nord


def validate_stacked(fname: str, container: dict, ctx, km: int,
                     required: tuple, *, what: str) -> None:
    """Tier-0 gate: every required key present, every shape declared.

    Shapes are ``(6,) + field_shape(name, n, ng, km)`` with
    :data:`CSW_OUT_LIKE` mapping a C-grid output name onto the field
    whose shape it shares.  Checking is not optional politeness: a
    stagger slip between ``(m_a, m_b)`` and ``(m_b, m_a)`` BROADCASTS in
    a later arithmetic op instead of raising, which is precisely the
    failure ``field_shape`` was written to stop in the NumPy lane.
    """
    if not isinstance(container, dict):
        raise TypeError(
            f"{fname}: {what} must be the face-stacked dict of arrays "
            f"(convention C1), got {type(container).__name__}. A list of "
            f"six per-face dicts is the NumPy lane's container -- convert "
            f"it with fv3_cgrid_phase_3d.state_3d_to_jax().")
    missing = [k for k in required if k not in container]
    if missing:
        raise KeyError(
            f"{fname}: {what} is missing {missing}; keys are "
            f"{sorted(container)}")
    n, ng = ctx.n, ctx.ng
    for name in required:
        a = jnp.asarray(container[name])
        want = (batch_size(ctx),) + field_shape(
            CSW_OUT_LIKE.get(name, name), n, ng, km)
        if a.shape != want:
            raise ValueError(
                f"{fname}: {what}[{name!r}] has shape {a.shape}, expected "
                f"{want} for n={n} ng={ng} km={km}")


# ---------------------------------------------------------------------
# assembly helpers
# ---------------------------------------------------------------------

def stack_levels(fname: str, name: str, per_level: list, want2d=None):
    """``km`` per-level outputs -> one array with ``km`` at AXIS 2.

    Axis 2 rather than -1 so an ``allflux`` slab ``(i, j, slot)``
    becomes ``(i, j, km, slot)`` -- the oracle's own
    ``allflux_x(i,j,k,iq)`` layout (``dyn_core.F90:860``) and the NumPy
    spec's ``np.stack(..., axis=2)``.  For a plain 2-D output the two
    conventions coincide.

    ``want2d`` is the DECLARED per-level shape, from a caller that has
    one (``field_shape``).  When it is given every level is checked
    against it, which also catches the case where level 0 is itself
    wrong; when it is omitted level 0's shape is the reference, for the
    stages whose kernels publish no shape table -- re-deriving ten
    Fortran bound expressions at the call site would be exactly the kind
    of restatement that drifts.

    ``jnp.stack`` is a pure index copy -- no ``x*y + z`` for XLA to
    contract into an FMA -- which is why the jit-vs-eager gate on the
    assembly step alone may be BITWISE while the gates on the kernels'
    arithmetic may not.
    """
    if not per_level:
        raise ValueError(
            f"{fname}: {name!r} got an empty per-level list; km >= 1 is "
            f"enforced by require_km, so an empty list here means the "
            f"level loop never ran")
    want = tuple(want2d) if want2d is not None else tuple(per_level[0].shape)
    start = 0 if want2d is not None else 1
    for k, arr in enumerate(per_level[start:], start=start):
        if tuple(arr.shape) != want:
            ref = ("the 3-D container expects" if want2d is not None
                   else "level 0 has")
            raise ValueError(
                f"{fname}: level {k} output {name!r} has shape "
                f"{arr.shape}, {ref} {want}. A stagger or window "
                f"mismatch here would broadcast, not raise.")
    return jnp.stack(per_level, axis=2)


def stack_levels_batched(fname: str, name: str, per_level: list):
    """``km`` FACE-BATCHED per-level stacks -> one array with ``km`` at
    AXIS 3.

    The vmapped arms' twin of :func:`stack_levels`: each element is a
    ``(6, i, j[, slot])`` face-batched plane, so the level axis is
    inserted at position 3 -- the loop path's per-face axis 2 with the
    face axis prepended -- and both arms return the same
    ``(6, i, j, km[, slot])`` layout (the convention the first batched
    arm, ``_csw_phase_3d_batched``, states inline).  Level 0's shape is
    the reference, exactly :func:`stack_levels`'s ``want2d=None`` arm
    and for the same reason: the d_sw kernels publish no shape table,
    and re-deriving their Fortran bound expressions here would be the
    restatement that drifts.
    """
    if not per_level:
        raise ValueError(
            f"{fname}: {name!r} got an empty per-level list; km >= 1 is "
            f"enforced by require_km, so an empty list here means the "
            f"level loop never ran")
    want = tuple(per_level[0].shape)
    for k, arr in enumerate(per_level[1:], start=1):
        if tuple(arr.shape) != want:
            raise ValueError(
                f"{fname}: level {k} output {name!r} has shape "
                f"{arr.shape}, level 0 has {want}. A stagger or window "
                f"mismatch here would broadcast, not raise.")
    return jnp.stack(per_level, axis=3)


def stack_faces(fname: str, per_face: list) -> dict:
    """Six per-face dicts of 3-D arrays -> one dict of ``(6, …)`` stacks.

    All six dicts carry the same keys and, within one key, the same
    per-face shape -- the FACE axis is stackable, the stagger axis is
    not, which is why this stacks per key and never across keys.
    """
    if len(per_face) not in (6,) and (len(per_face) < 6
                                      or len(per_face) % 6):
        raise ValueError(
            f"{fname}: expected 6 per-face dicts (convention C1) or "
            f"6*kt*kt per-window dicts, got {len(per_face)}")
    keys = tuple(per_face[0])
    for t, d in enumerate(per_face):
        if tuple(d) != keys:
            raise KeyError(
                f"{fname}: face {t + 1} produced keys {sorted(d)}, face 1 "
                f"produced {sorted(keys)} -- a per-face lane split")
    return {k: jnp.stack([d[k] for d in per_face], axis=0) for k in keys}
