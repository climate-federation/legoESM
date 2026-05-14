"""AIMIP Phase 2 sentinel: every nominally-trainable param must be
wired through ``build_segment_fn`` or explicitly listed as a known
unwired-TODO.

Iters 251-254 added ``albedo_land``, ``Ri_crit``, ``Ck``, ``l_mix_max``
to the constraint lists in :mod:`legoesm.training.trainable_params`
but NONE of them are kwargs on
:func:`legoesm.driver.compiled_segments.build_segment_fn`.  Net result:
the gradient updates those parameters but their values never affect
the rollout, so the training loop optimises a constant.  CLAUDE.md
explicitly bans this kind of half-impl.

This sentinel catches the gap.  As each param is wired end-to-end
(``build_segment_fn`` kwarg → ``step_unified`` kwarg → physics
module), remove its name from ``UNWIRED_TODO``; the test then
enforces that no NEW unwired params land.
"""
from __future__ import annotations

import inspect

from legoesm.driver.compiled_segments import build_segment_fn
from legoesm.training.trainable_params import (
    _COMMON_TRAINABLE,
    _LOUIS_TURBULENCE_TRAINABLE,
    _SBM_TRAINABLE,
)

# Params known to be added to constraint lists but NOT yet threaded as
# kwargs on ``build_segment_fn``.  Each line must cite the iter that
# added it to the trainable list + the AIMIP phase that will land the
# wiring.  As wiring lands, the entry leaves this set.
#
# iter-256: the previous UNWIRED_TODO entries
# (``albedo_land``, ``Ri_crit``, ``Ck``, ``l_mix_max``) were all
# reverted out of the constraint lists rather than left as nominal-
# only trainables.  Surface and turbulence infra (land-fraction
# blend; Louis kwargs through ``step_unified``) must land before they
# are re-added.  Set is intentionally empty so any new addition
# without wiring trips ``test_all_trainable_param_names_are_segment_fn_kwargs_or_todo``.
UNWIRED_TODO: set[str] = set()


def _kwargs_of_build_segment_fn() -> set[str]:
    sig = inspect.signature(build_segment_fn)
    return set(sig.parameters)


def test_all_trainable_param_names_are_segment_fn_kwargs_or_todo():
    """Every name in the trainable lists must either be a kwarg of
    ``build_segment_fn`` (so gradients flow through the rollout) or
    appear in ``UNWIRED_TODO`` with a documented future-iter plan.
    """
    kwargs = _kwargs_of_build_segment_fn()
    all_trainable = (
        [c.name for c in _COMMON_TRAINABLE]
        + [c.name for c in _SBM_TRAINABLE]
        + [c.name for c in _LOUIS_TURBULENCE_TRAINABLE]
    )
    truly_unwired = [n for n in all_trainable if n not in kwargs]
    surprise = [n for n in truly_unwired if n not in UNWIRED_TODO]
    assert not surprise, (
        f"New unwired trainable param(s) detected: {surprise}.  "
        f"Each name in _COMMON_TRAINABLE / _SBM_TRAINABLE / "
        f"_LOUIS_TURBULENCE_TRAINABLE MUST either be a kwarg of "
        f"build_segment_fn (so the gradient affects the rollout) or "
        f"be listed in UNWIRED_TODO with a documented wiring plan.  "
        f"See tests/test_aimip_phase2_trainable_wired.py."
    )


def test_unwired_todo_entries_actually_unwired():
    """``UNWIRED_TODO`` must not contain names that have already been
    wired — otherwise the whitelist is stale and risks silently
    accepting a future regression.
    """
    kwargs = _kwargs_of_build_segment_fn()
    stale = [n for n in UNWIRED_TODO if n in kwargs]
    assert not stale, (
        f"UNWIRED_TODO contains names that ARE kwargs of "
        f"build_segment_fn: {stale}.  Remove them from UNWIRED_TODO "
        f"so the strictness of the sentinel grows monotonically."
    )


def test_wired_trainable_count_matches_expected():
    """Pin the count of trainable params that are ACTUALLY wired so a
    regression that silently drops one is caught.

    Wired = name appears as a kwarg of ``build_segment_fn``.  As of
    iter-255 (this sentinel) the wired-and-trainable set is the
    original 8: ``tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
    C_H, C_E, albedo_ice, albedo_ocean``.
    """
    kwargs = _kwargs_of_build_segment_fn()
    all_trainable = (
        [c.name for c in _COMMON_TRAINABLE]
        + [c.name for c in _SBM_TRAINABLE]
        + [c.name for c in _LOUIS_TURBULENCE_TRAINABLE]
    )
    wired = sorted(n for n in all_trainable if n in kwargs)
    expected = sorted(
        ["tau_equator", "tau_pole", "sbm_tau_c", "sbm_RH_ref",
         "C_H", "C_E", "albedo_ice", "albedo_ocean",
         # iter-257 (AIMIP Phase 2.6): albedo_land wired through
         # build_segment_fn + step_unified + compute_radiation_core
         # with the 3-way land/ice/ocean blend.
         "albedo_land",
         # iter-258 (AIMIP Phase 2.7): l_mix_max wired through
         # build_segment_fn + step_unified + physics_step_no_rad ->
         # louis_turbulence kwarg override.
         "l_mix_max"]
    )
    assert wired == expected, (
        f"Wired-trainable param set changed.  Expected {expected}, "
        f"got {wired}.  If you added wiring for a new param, ALSO "
        f"remove it from UNWIRED_TODO and bump the expected list."
    )
