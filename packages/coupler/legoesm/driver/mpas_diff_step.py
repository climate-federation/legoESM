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

TWO LOOP FORMS, same per-step sequence, same gradient:

* ``schedule=None`` (default): a Python-unrolled loop; each step's physics
  variant is a trace-time choice.  Reverse-mode memory grows with the window
  (every step's residuals, or with ``checkpoint=True`` every step's carry:
  measured 67 GiB at 24 steps and 506 GiB at 72 on production res6).
* ``schedule`` in ``legoesm.training.checkpoint_schedule.VALID_SCHEDULES``: a
  fixed-length loop whose body is ONE step; the variant is picked on device by
  ``lax.switch`` on the step counter from a trace-time table (so only the
  variant the eager loop would run executes and only its gradient is
  non-zero), and the forcing clock -- ``day_of_year``, ``seconds_of_day`` and
  ``model_seconds`` when the forcing carries it -- is read from trace-time
  tables built by the same calendar/clock the unrolled loop uses
  (:func:`clock_tables`).  The
  loop is driven by :func:`legoesm.training.checkpoint_schedule.checkpointed_loop`
  (``"uniform"``: per-step remat; ``"binomial"``: Griewank-Walther/Treeverse,
  peak memory ~ ``checkpoints`` carries + one step's residuals -- under
  reverse-mode ``lax.switch`` one step's residuals are the union of every
  used variant's residual shapes, zero-filled for the variants not taken; default
  ``checkpoints = ceil(sqrt(n_steps))``; ``storage="host"`` offloads matmul
  residuals).  The ``checkpoint`` flag belongs to the unrolled form and is
  refused here: the schedule IS the remat policy, and nesting a second
  ``jax.checkpoint`` would only recompute each step a third time.

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

__all__ = ["mpas_cadence", "pick_physics_fn", "mpas_branch_index", "mpas_step",
           "mpas_segment"]


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


def mpas_branch_index(step: int, exp: dict) -> int:
    """The variant ``step`` runs as a switch index: 0 full, 1 no-radiation,
    2 held -- :func:`pick_physics_fn` applied to the indices, so the scheduled
    loop and the eager loop share the one cadence rule."""
    use_rad, is_phys = mpas_cadence(
        step, subcycle_rad=exp["subcycle_rad"],
        rad_update_steps=exp["rad_update_steps"], hold_phys=exp["hold_phys"],
        phys_update_steps=exp["phys_update_steps"])
    return pick_physics_fn(use_rad, is_phys, 0, 1, 2)


_VARIANT_KEYS = ("physics_fn", "physics_fn_norad", "physics_fn_held")


def _run_variant(exp: dict, pfn, state, phys_state, forcing, target_mass):
    """One production atmosphere step with physics variant ``pfn``:
    ``(state, phys_state, sfc_diag)``."""
    state, phys_state, sfc_diag, _ = exp["model"].step_pure(
        state, exp["dt"], physics_fn=pfn, forcing=forcing,
        phys_state=phys_state, target_mass=target_mass)
    return exp["poststep"](state), phys_state, sfc_diag


def mpas_step(exp: dict, state, phys_state, forcing, step: int,
              target_mass=None):
    """One production atmosphere step: ``(state, phys_state, sfc_diag)``."""
    pfn = exp[_VARIANT_KEYS[mpas_branch_index(step, exp)]]
    return _run_variant(exp, pfn, state, phys_state, forcing, target_mass)


def _clock(exp: dict, forcing, step: int):
    """The eager loop's per-step forcing: ``forcing`` with its clock advanced
    (no dict is built without forcing, as in the eager loop)."""
    if forcing is None:
        return None
    doy, sod = exp["calendar"](step)
    f = dict(forcing, day_of_year=jnp.asarray(doy),
             seconds_of_day=jnp.asarray(sod))
    if "model_seconds" in f:      # ecrad_ifs seed clock (vote S3a)
        f["model_seconds"] = jnp.asarray(exp["model_seconds"](step))
    return f


def clock_tables(exp: dict, forcing, steps):
    """Per-step clock values of :func:`_clock` for ``steps``, stacked (the
    scheduled loop indexes them with its traced counter).  Same keys, same
    source (``exp["calendar"]``, ``exp["model_seconds"]``) as ``_clock``."""
    if forcing is None:
        return None
    clock = [exp["calendar"](s) for s in steps]
    t = {"day_of_year": jnp.asarray([c[0] for c in clock]),
         "seconds_of_day": jnp.asarray([c[1] for c in clock])}
    if "model_seconds" in forcing:
        t["model_seconds"] = jnp.asarray([exp["model_seconds"](s) for s in steps])
    return t


def mpas_segment(exp: dict, state, phys_state, n_steps: int, forcing=None,
                 step0: int = 0, checkpoint: bool = False,
                 schedule: str | None = None, checkpoints: int | None = None,
                 storage: str = "recompute", tap=None):
    """Advance ``n_steps`` from job-local ``step0``; pure in (state, phys_state,
    forcing).  The forcing clock (``day_of_year`` / ``seconds_of_day``, and
    ``model_seconds`` when the run carries it) is
    advanced per step exactly as the eager loop does; every other forcing key
    is held (see the module docstring).  The dry-mass target is
    ``model.mass_target`` as in eager ``step``: None on the MPAS lane (the
    factory never sets ``anchor_mass_to_initial``), so the fixer restores each
    step's own pre-step mass.

    ``schedule=None``: Python-unrolled loop; ``checkpoint`` rematerialises each
    step in the reverse pass.  Otherwise the fixed-length scheduled loop of the
    module docstring (``checkpoints`` / ``storage`` are its budget and residual
    storage; ``checkpoint`` must stay False).

    ``tap(step, state, phys_state) -> (state, phys_state)``: an identity hook
    applied to the state ENTERING each step of the unrolled loop (a
    ``custom_vjp`` identity whose backward pass records the cotangent is how
    a probe reads the adjoint at every step); unrolled loop only."""
    forcing = exp["forcing"] if forcing is None else forcing
    if schedule is None and (checkpoints is not None or storage != "recompute"):
        raise ValueError("mpas_segment: checkpoints/storage configure a "
                         "schedule; the unrolled loop (schedule=None) takes "
                         "only checkpoint=")
    if schedule == "uniform" and checkpoints is not None:
        raise ValueError("mpas_segment: checkpoints is the binomial budget; "
                         "the uniform schedule remats every step")
    target_mass = exp["model"].mass_target(state)
    if schedule is not None:
        if tap is not None:
            raise ValueError("mpas_segment: tap is the unrolled loop's "
                             "per-step hook; the scheduled loop has no "
                             "static step to hand it")
        return _scheduled_segment(exp, state, phys_state, n_steps, forcing,
                                  step0, target_mass, checkpoint, schedule,
                                  checkpoints, storage)
    for k in range(n_steps):
        step = step0 + k
        f = _clock(exp, forcing, step)

        def _one(s, p, f_, _step=step):
            return mpas_step(exp, s, p, f_, _step, target_mass)[:2]
        if checkpoint:
            _one = jax.checkpoint(_one)
        if tap is not None:
            state, phys_state = tap(step, state, phys_state)
        state, phys_state = _one(state, phys_state, f)
    return state, phys_state


def _scheduled_segment(exp, state, phys_state, n_steps, forcing, step0,
                       target_mass, checkpoint, schedule, checkpoints, storage):
    from legoesm.training.checkpoint_schedule import checkpointed_loop

    if checkpoint:
        raise ValueError(
            "mpas_segment: checkpoint=True is the unrolled loop's per-step "
            "remat; with a schedule the schedule is the remat policy "
            "(use schedule='uniform' for per-step remat).")
    steps = range(step0, step0 + n_steps)
    # Trace-time tables (n_steps is static): which variant each step runs and
    # the eager loop's clock at each step.  Only the variants this window uses
    # are traced into the switch (a None variant -- cadence without that
    # branch -- never enters it).
    branch = [mpas_branch_index(s, exp) for s in steps]
    used = sorted(set(branch))
    idx = jnp.asarray([used.index(b) for b in branch], dtype=jnp.int32)
    tables = clock_tables(exp, forcing, steps)

    def _branch(key):
        pfn = exp[key]

        def run(s, p, f):
            return _run_variant(exp, pfn, s, p, f, target_mass)[:2]
        return run
    branches = tuple(_branch(_VARIANT_KEYS[b]) for b in used)

    def body(carry):
        i, s, p = carry
        f = forcing
        if f is not None:
            f = dict(f, **{k: v[i] for k, v in tables.items()})
        if len(branches) == 1:
            s, p = branches[0](s, p, f)
        else:
            s, p = jax.lax.switch(idx[i], branches, s, p, f)
        return i + 1, s, p

    _, state, phys_state = checkpointed_loop(
        body, (jnp.asarray(0, dtype=jnp.int32), state, phys_state), n_steps,
        schedule=schedule, checkpoints=checkpoints, storage=storage)
    return state, phys_state
