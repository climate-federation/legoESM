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
}


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
