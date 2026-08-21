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

__all__ = ["require_fp64", "describe_float_leaves", "require_explicit_e3t_mode"]

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


def require_explicit_e3t_mode(context: str = "oracle comparison") -> str:
    """Raise unless ``LEGOESM_NEMO_E3T`` is set EXPLICITLY; return its value.

    ``bridge_nemo_to_legoesm_topo`` defaults this to ``"off"``, which feeds
    legoESM NEMO's *analytic 1-D* ``e3t_1d`` while NEMO itself runs on the 3-D
    ``e3t_0``.  At and below k=25 (0-based, as everywhere in this file) those two
    NEMO ladders diverge, by up to **70.4 m**
    at the deepest wet level -- which is 12.9% of ``e3t_0`` and 14.8% of
    ``e3t_1d``.  Always quote the denominator here: those two percentages are
    ONE measurement and have already been mistaken for a disagreement between
    two.  NEMO builds its reference ladder in two passes
    (``zgr_lib.F90::zgr_sco_mi96`` re-anchors at
    ``kkconst = argmin(|gdepw - rn_hco|)``, and DINO's ``rn_hco = 1000 m`` puts
    that at ``kkconst = 26`` in Fortran's 1-based indexing, i.e. k=25 here), so
    the default silently puts a 12.9%-of-``e3t_0`` geometry error into the
    deepest third of the column.

    It has now contaminated FOUR measurements.  The most recent cost a full
    false root-cause: the barotropic seed measured 2.31e-2 and was attributed to
    the depth-averaging operator, when at ``e3t=both`` the seed is 2.19e-16
    (exact) and the error actually ACCUMULATES through the substeps -- the
    opposite conclusion.

    The default is NOT changed here, and this gate stays fail-closed.  But its
    STATED REASON no longer holds as written: the instability it cites (gate row
    "STABILITY on NEMO true grid") DID NOT REPRODUCE in 2026-08-21 measurements
    -- four 90-day DINO twin arms from the day-180 restart, differing only in
    this variable, all ran stable to day 90 at 0.633-0.635 m/s peak speed, with
    the two end arms confirmed under an fp64 precision policy (#1455; see
    scripts/validate/ocean_fidelity/dino_1226/d180_step_walk.py and the note in
    nemo_state_bridge.effective_vertical_scale_factors).  That is a
    non-reproduction under one configuration, NOT a refutation, so "a real
    unfixed defect" is downgraded to "an unexplained recorded observation".
    Either way the mode stays a CONSCIOUS choice here, never an inherited silent
    default -- which is the part of this gate that was always load-bearing.
    """
    import os

    mode = os.environ.get("LEGOESM_NEMO_E3T")
    if mode is None:
        raise ValueError(
            f"{context}: LEGOESM_NEMO_E3T is NOT SET, so the bridge would "
            'silently use "off" -- NEMO\'s analytic e3t_1d, which differs from '
            "the e3t_0 NEMO actually runs on by up to 70.4 m below k=25 "
            "(12.9% of e3t_0, 14.8% of e3t_1d). That "
            "default has already contaminated four measurements, most recently "
            "producing a completely wrong root cause for the barotropic seed. "
            'Set it explicitly: "both" (NEMO\'s true 3-D ladder, what a '
            'fidelity comparison wants) or "off" (the 1-D ladder) -- and say '
            "which in the report."
        )
    if mode not in ("off", "e3t_only", "gdept_only", "both"):
        raise ValueError(
            f"{context}: unknown LEGOESM_NEMO_E3T={mode!r}; expected "
            '"off", "e3t_only", "gdept_only" or "both"')
    return mode
