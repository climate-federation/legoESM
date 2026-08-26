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
    "build_batched_gs",
    "require_f64_jax",
    "require_bool",
    "require_km",
    "require_nord",
    "validate_stacked",
    "stack_levels",
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
        a new leading face axis, key by key.  ``jnp.stack`` is a pure
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
    if len(gs6) != 6 or len(flags6) != 6:
        raise ValueError(
            f"build_batched_gs: ctx carries {len(gs6)} gridstructs and "
            f"{len(flags6)} flag sets, not 6 of each -- the face batch "
            f"axis is the cube's six faces")

    keys0 = set(gs6[0])
    for t in range(1, 6):
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
        shapes = {tuple(jnp.shape(gs6[t][key])) for t in range(6)}
        if len(shapes) != 1:
            unstacked.append(key)
            continue
        stacked[key] = jnp.stack([gs6[t][key] for t in range(6)], axis=0)

    f0 = flags6[0]
    for name in f0._fields:
        if name in PER_FACE_FLAG_FIELDS:
            continue
        vals = [getattr(flags6[t], name) for t in range(6)]
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
        "da_min6": jnp.asarray([fl.da_min for fl in flags6],
                               dtype=jnp.float64),
        "da_min_c6": jnp.asarray([fl.da_min_c for fl in flags6],
                                 dtype=jnp.float64),
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

def require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or, with ``jax_enable_x64``
    disabled, the whole phase would silently run in float32 -- and the
    oracle build is ``-fdefault-real-8``.
    """
    for name, a in arrays.items():
        if a is None:
            continue
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


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
        want = (6,) + field_shape(CSW_OUT_LIKE.get(name, name), n, ng, km)
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


def stack_faces(fname: str, per_face: list) -> dict:
    """Six per-face dicts of 3-D arrays -> one dict of ``(6, …)`` stacks.

    All six dicts carry the same keys and, within one key, the same
    per-face shape -- the FACE axis is stackable, the stagger axis is
    not, which is why this stacks per key and never across keys.
    """
    if len(per_face) != 6:
        raise ValueError(
            f"{fname}: expected 6 per-face dicts (convention C1), got "
            f"{len(per_face)}")
    keys = tuple(per_face[0])
    for t, d in enumerate(per_face):
        if tuple(d) != keys:
            raise KeyError(
                f"{fname}: face {t + 1} produced keys {sorted(d)}, face 1 "
                f"produced {sorted(keys)} -- a per-face lane split")
    return {k: jnp.stack([d[k] for d in per_face], axis=0) for k in keys}
