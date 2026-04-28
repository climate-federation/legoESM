"""Dispatch tests for the GM/Redi MPAS skeleton.

These tests guard the not-yet-implemented contract: every public
entry point in ``gm_redi_mpas.py`` must raise NotImplementedError
with a pointer to the implementation plan.  When the body of any
function is filled in, the corresponding test here MUST be replaced
with a real numerical correctness test — the file should not be left
testing only the error path once the implementation lands.

See docs/ocean_experiments/gm_redi_mpas_plan.md for the phased plan.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    compute_isopycnal_slopes_mpas,
    gm_redi_tracer_tendency_centered_mpas,
    gm_redi_tracer_tendency_mpas,
    gm_redi_tracer_tendency_triads_mpas,
)


@pytest.mark.parametrize("fn,name", [
    (compute_isopycnal_slopes_mpas,         "compute_isopycnal_slopes_mpas"),
    (gm_redi_tracer_tendency_centered_mpas, "gm_redi_tracer_tendency_centered_mpas"),
    (gm_redi_tracer_tendency_triads_mpas,   "gm_redi_tracer_tendency_triads_mpas"),
    (gm_redi_tracer_tendency_mpas,          "gm_redi_tracer_tendency_mpas"),
])
def test_skeleton_raises_with_plan_pointer(fn, name):
    """Every public entry point must raise NotImplementedError that
    names the function and points at the plan doc, so callers know
    what's missing and where to look."""
    with pytest.raises(NotImplementedError) as exc_info:
        # Pass placeholder Nones — the body must raise before touching args.
        fn(*[None] * len(_signature_arg_count(fn)))
    msg = str(exc_info.value)
    assert name in msg, f"error must name the called function: got {msg!r}"
    assert "gm_redi_mpas_plan.md" in msg, (
        "error must point at the implementation plan doc")


def _signature_arg_count(fn) -> tuple:
    """Number of positional args fn declares (best-effort)."""
    import inspect
    sig = inspect.signature(fn)
    n_pos = sum(
        1 for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    return tuple(range(n_pos))
