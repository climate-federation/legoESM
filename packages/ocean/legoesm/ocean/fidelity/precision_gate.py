"""Precision contract for oracle-fidelity comparisons (skill Rule 1c).

NEMO, MITgcm and Veros are **fp64** models.  An oracle comparison run in
float32 measures our own rounding, not our physics: f32 eps = 1.19e-7, so any
"finding" in the 1e-8..1e-5 band may be the dtype rather than a defect.

The trap this exists to close (#1226): ``JAX_ENABLE_X64=1`` *permits* f64
arrays but does NOT change legoESM's precision policy.  Constructors cast to
``get_policy().control``, which **defaults to float32** — so a harness can run
with x64 enabled and still build its GRID in single precision while T/S look
correct.  ``create_z_star_from_thicknesses`` did exactly that to NEMO's f64
``gdept_1d`` (median |rel| 2.555e-8 = 0.21 x f32 eps, the fingerprint of
single precision), which alone accounted for the residuals in ``eos_rab
alpha``, ``bn2`` and 10 of ``zdf_mxl``'s MLD columns.

This is HARNESS glue, not model code: it asserts a comparison precondition and
never changes any answer.  Would a user with a different goal select it?  No —
hence it lives under ``ocean/fidelity/``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.precision import get_policy

__all__ = ["require_fp64", "describe_float_leaves"]

_POLICY_FIELDS = ("storage", "compute", "accumulate", "control")


def describe_float_leaves(obj, prefix: str = "") -> list[tuple[str, str]]:
    """``[(path, dtype)]`` for every inexact (floating) leaf of a pytree.

    Integer/bool leaves (level indices, masks, counts) are skipped — they carry
    no precision.  Used to PRINT dtypes, which Rule 10 requires instead of
    trusting a declaration.
    """
    out: list[tuple[str, str]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(obj)[0]:
        arr = jnp.asarray(leaf)
        if not jnp.issubdtype(arr.dtype, jnp.inexact):
            continue
        out.append((prefix + jax.tree_util.keystr(path), str(arr.dtype)))
    return out


def require_fp64(*objects, context: str = "oracle comparison") -> None:
    """Raise unless the precision policy AND every float leaf are float64.

    Parameters
    ----------
    *objects
        Pytrees to scan (a ``z_coord``, a geometry, a state, raw arrays).
        Pass the GEOMETRY, not just the state: the f32-ladder bug had a
        correct-looking state inside wrongly-sized boxes (skill Rule 2).
    context
        Named in the error, so a failure says which comparison is unsound.

    Raises
    ------
    ValueError
        Listing the offending policy fields and leaves.  Deliberately loud and
        fail-CLOSED: a silently-f32 comparison is worse than no comparison,
        because it produces plausible numbers that are pure rounding.
    """
    policy = get_policy()
    bad_policy = [
        f"{f}={jnp.dtype(getattr(policy, f)).name}"
        for f in _POLICY_FIELDS
        if jnp.dtype(getattr(policy, f)) != jnp.float64
    ]

    bad_leaves: list[tuple[str, str]] = []
    for i, obj in enumerate(objects):
        for path, dt in describe_float_leaves(obj, prefix=f"arg{i}"):
            if dt != "float64":
                bad_leaves.append((path, dt))

    if not bad_policy and not bad_leaves:
        return

    lines = [
        f"{context}: FLOAT64 REQUIRED — the oracle is an fp64 model, so this "
        f"comparison would be measuring our own rounding (f32 eps = 1.19e-7)."
    ]
    if bad_policy:
        lines += [
            "  precision policy is not fp64: " + ", ".join(bad_policy),
            "    fix:  from legoesm.core.precision import PrecisionPolicy, "
            "set_policy",
            "          set_policy(PrecisionPolicy.fp64())",
            "    NOTE: JAX_ENABLE_X64=1 does NOT do this — it only permits f64.",
        ]
    if bad_leaves:
        shown = bad_leaves[:20]
        lines.append(f"  {len(bad_leaves)} non-f64 float leaf/leaves:")
        lines += [f"    {p}: {d}" for p, d in shown]
        if len(bad_leaves) > len(shown):
            lines.append(f"    ... and {len(bad_leaves) - len(shown)} more")
        lines.append(
            "    a float32 GEOMETRY with a float64 state is the easy case to "
            "miss: the values look right, the boxes holding them do not."
        )
    raise ValueError("\n".join(lines))
