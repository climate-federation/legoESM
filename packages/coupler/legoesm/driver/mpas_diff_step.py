"""Differentiable production MPAS step / segment (review F9, design D3 subset).

The production AMIP deck runs ``ModelDriver._run_column``: an eager per-step
Python loop that ``jax.grad`` cannot enter.  Its per-step core is already pure
(``MPASPrimitiveEquationModel.step_pure``: RK dynamics, operator-split physics,
dry-mass fixer, positivity), so nothing is re-implemented here.
``ModelDriver.export_mpas_step`` runs the production loop's own setup and hands
back the closures it built (the CAM6 physics variants, the post-step dycore
edits, the cadence) at step 0; :func:`mpas_segment` then advances

    (state, phys_state) -> (state, phys_state)

over ``n_steps`` with exactly the eager loop's per-step sequence for the
ATMOSPHERE: the cadence-selected physics variant, ``step_pure`` with the same
dry-mass anchor, then the shared post-step edits (top sponge, q_v smoother).

WHAT IS HELD FIXED inside a differentiated window (documented subset, not a
silent approximation): the forcing dict other than its per-step clock --
prescribed SST/SIC, the prognostic ice skin, the interactive land's skin
temperature, albedo, surface humidity and fluxes, external forcing.  The eager
loop refreshes those between steps (land every ``land_k`` steps, ice skin every
step), so a window longer than one step differs from production by those
feedbacks; ``land_step`` is exported separately so its own gradient can be
measured.  MPI, the post-step hard-saturation drain and the budget ledger are
refused at export.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["mpas_cadence", "pick_physics_fn", "mpas_step", "mpas_segment"]


def mpas_cadence(step: int, *, subcycle_rad: bool, rad_update_steps: int,
                 hold_phys: bool, phys_update_steps: int) -> tuple[bool, bool]:
    """``(use_rad, is_phys_step)`` for job-local ``step``: radiation is solved
    on step 0 and every ``rad_update_steps``-th step, physics on step 0 and
    every ``phys_update_steps``-th; the one rule the eager loop and the
    segment share."""
    return ((not subcycle_rad) or step % rad_update_steps == 0,
            (not hold_phys) or step % phys_update_steps == 0)


def pick_physics_fn(use_rad: bool, is_phys_step: bool, full, norad, held):
    """The physics variant a step runs: full (with radiation), no-radiation
    (held heating), or the held-physics re-application between physics steps."""
    return (full if use_rad else norad) if is_phys_step else held


def mpas_step(exp: dict, state, phys_state, forcing, step: int,
              target_mass=None):
    """One production atmosphere step: ``(state, phys_state, sfc_diag)``."""
    use_rad, is_phys = mpas_cadence(
        step, subcycle_rad=exp["subcycle_rad"],
        rad_update_steps=exp["rad_update_steps"], hold_phys=exp["hold_phys"],
        phys_update_steps=exp["phys_update_steps"])
    pfn = pick_physics_fn(use_rad, is_phys, exp["physics_fn"],
                          exp["physics_fn_norad"], exp["physics_fn_held"])
    state, phys_state, sfc_diag, _ = exp["model"].step_pure(
        state, exp["dt"], physics_fn=pfn, forcing=forcing,
        phys_state=phys_state, target_mass=target_mass)
    return exp["poststep"](state), phys_state, sfc_diag


def mpas_segment(exp: dict, state, phys_state, n_steps: int, forcing=None,
                 step0: int = 0, checkpoint: bool = False):
    """Advance ``n_steps`` from job-local ``step0``; pure in (state, phys_state,
    forcing).  The forcing clock (``day_of_year`` / ``seconds_of_day``) is
    advanced per step exactly as the eager loop does; every other forcing key
    is held (see the module docstring).  The dry-mass target is
    ``model.mass_target`` as in eager ``step``: None on the MPAS lane (the
    factory never sets ``anchor_mass_to_initial``), so the fixer restores each
    step's own pre-step mass.  ``checkpoint`` rematerialises each step in the
    reverse pass (bounded memory)."""
    forcing = exp["forcing"] if forcing is None else forcing
    target_mass = exp["model"].mass_target(state)
    for k in range(n_steps):
        step = step0 + k
        f = forcing
        if f is not None:     # the eager loop builds no dict without forcing
            doy, sod = exp["calendar"](step)
            f = dict(f, day_of_year=jnp.asarray(doy),
                     seconds_of_day=jnp.asarray(sod))

        def _one(s, p, f_, _step=step):
            return mpas_step(exp, s, p, f_, _step, target_mass)[:2]
        if checkpoint:
            _one = jax.checkpoint(_one)
        state, phys_state = _one(state, phys_state, f)
    return state, phys_state
