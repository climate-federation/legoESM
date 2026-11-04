"""Conservation gate for column runs with a surface law (PI 2026-11-02 .. 11-03).

A CloudBench column run passes only if, at every step, the column's water
(q_v + q_c + q_r), moist energy (c_pd T + L_v q_v + ½|u|², the one total every
closure is held to) and both momentum components change by exactly what the
surface law, the rain reaching the ground and the SCM forcing supply, to round-off;
and if the column totals rebuilt from the recorded states agree with those supplies
step by step and over the whole run. A run that misses is recorded as failed.

Checks per quantity (``rollout`` records the terms, :func:`~.candidate_eval.
step_budget`):

* step: ``|rate − supply| ≤ STEP_REL_TOL · (|rate| + Σ|supply terms| + Σ gross
  process terms)`` (the gross terms keep a step with ~zero supply from judging
  its round-off against nothing);
* process: turbulence against the surface law, microphysics against the rain
  reaching the ground (water) and zero (energy, momentum), each to ``STEP_REL_TOL``
  of that process's gross terms, so errors in two processes cannot cancel;
* record: the surface record's density and Exner function are the column's
  surface interface values (the conversion both ends use);
* state: ``|ΔX − dt·rate| ≤ STATE_REL_TOL · Σ m|x|`` per step, and the same
  for the whole run's ``X_N − X_0 − dt·Σ supply`` (cancellation against the column
  content is what limits this one);
* boundary: each closure's own surface flux agrees with the surface law to
  ``OWN_BOUNDARY_TOL``.

Any non-finite budget term fails the run. The SCM forcing (subsidence, advection,
nudging, wind relaxation, radiation) is an external source prescribed as in the LES:
its own column change is its declared supply, so the gate certifies that nothing
outside turbulence, microphysics and forcing changes the column, not what the
forcing itself does (e.g. subsidence carrying vapour through the column top).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants

from .candidate_eval import BUDGET_QUANTITIES, RolloutRecord

#: Tolerances, about ten times the float64 floor measured over 12 five-day CloudBench
#: runs (3 cases x 4 benchmarks in energy form, 2026-11-03): worst step 7.1e-14,
#: worst per-step state 1.2e-15, worst whole-run state 8.9e-16.
STEP_REL_TOL = 1.0e-12
STATE_REL_TOL = 1.0e-14
#: A closure's own surface flux may differ from the surface law only by its own
#: heat-flux conversion (the lowest-level vs surface Exner ratio). Measured as the
#: extra transfer it would make, relative to the step's gross turbulent transfer of
#: that quantity; beyond this the closure applies a boundary flux of its own (EDMF's
#: plume) and the run fails.
OWN_BOUNDARY_TOL = 1.0e-2


class ConservationVerdict(NamedTuple):
    passed: bool
    worst_step_rel: dict       # quantity -> max relative residual of the step total
    worst_step_index: dict     # quantity -> step of that maximum
    worst_process_rel: dict    # "process_quantity" -> max relative residual
    worst_state_rel: dict      # quantity -> max relative per-step state residual
    run_state_rel: dict        # quantity -> whole-run relative state residual
    #: flux -> max over steps of the transfer the closure's own surface flux would
    #: have made beyond the surface law's, relative to the step's gross turbulent
    #: transfer of that quantity (see ``OWN_BOUNDARY_TOL``)
    own_boundary_rel: dict
    reason: str


#: Declared supply of each checked process (forcing is its own declared source).
_PROCESS_SUPPLY = {
    "turbulence": lambda b, q: b["sfc_" + q],
    "microphysics": lambda b, q: -b["precip"] if q == "water" else 0.0 * b["precip"],
}
_OWN_FLUX = {"wtheta": ("energy", "w_theta_s"), "wqt": ("water", "w_qt_s"),
             "uw": ("u", "uw_s"), "vw": ("v", "vw_s")}


def _supply(b: dict, q: str):
    out = b[f"sfc_{q}"] + b[f"forcing_{q}"]
    return out - b["precip"] if q == "water" else out


def _supply_scale(b: dict, q: str):
    out = jnp.abs(b[f"sfc_{q}"]) + jnp.abs(b[f"forcing_{q}"])
    return out + jnp.abs(b["precip"]) if q == "water" else out


def _rel(resid, scale):
    """``|resid| / scale``; 0 where both vanish, inf where only the scale does."""
    resid, scale = np.abs(resid), np.asarray(scale)
    return np.where(scale > 0.0, resid / np.where(scale > 0.0, scale, 1.0),
                    np.where(resid == 0.0, 0.0, np.inf))


def column_totals(inputs) -> dict:
    """``(water, energy, u, v)`` column totals and the matching gross content
    ``Σ m|x|`` (the round-off scale), for a ColumnInputs (leading axes kept)."""
    m = inputs.mass
    ke = 0.5 * (inputs.u ** 2 + inputs.v ** 2)
    lv = constants.L_v  # latent-ok: column moist-energy total at the constant L_v
    e = constants.c_pd * inputs.temp + lv * inputs.q_v + ke
    fields = {"water": inputs.q_t, "energy": e, "u": inputs.u, "v": inputs.v}
    return ({k: jnp.sum(m * f, axis=-1) for k, f in fields.items()},
            {k: jnp.sum(m * jnp.abs(f), axis=-1) for k, f in fields.items()})


def check_conservation(rec: RolloutRecord, dt: float, *, step_rel_tol: float = STEP_REL_TOL,
                       state_rel_tol: float = STATE_REL_TOL,
                       own_boundary_tol: float = OWN_BOUNDARY_TOL) -> ConservationVerdict:
    """The gate verdict for one rollout (see module docstring)."""
    if rec.budget is None:
        raise ValueError("this rollout recorded no budget (no surface law on the column)")
    b = {k: np.asarray(v, np.float64) for k, v in rec.budget.items()}
    nsteps = b["rate_water"].shape[0]
    empty = {q: float("nan") for q in BUDGET_QUANTITIES}
    if nsteps == 0:
        return ConservationVerdict(False, empty, {q: -1 for q in empty}, {}, empty, empty,
                                   {}, "no steps: nothing was checked")
    tot_in, scale_in = column_totals(rec.inputs)
    tot_f, scale_f = column_totals(rec.final_inputs)
    worst, where, wproc, wstate, wrun, own, bad = {}, {}, {}, {}, {}, {}, []
    # a NaN or inf scale would read as a zero residual below
    bad += [f"{k}: non-finite budget" for k, v in b.items() if not np.all(np.isfinite(v))]

    def judge(label, rel, tol):
        rel = np.asarray(rel, np.float64)
        if np.any(np.isnan(rel)):
            bad.append(f"{label}: non-finite budget")
            return float("nan")
        m = float(np.max(rel))
        if m > tol:  # inf (a residual against a zero scale) fails too
            bad.append(f"{label}: residual {m:.3e} at step {int(np.argmax(rel))}")
        return m

    for q in BUDGET_QUANTITIES:
        rate, sup = b[f"rate_{q}"], np.asarray(_supply(b, q))
        gross_q = sum(b[f"gross_{p}_{q}"] for p in ("turbulence", "microphysics", "forcing"))
        rel = _rel(rate - sup, np.abs(rate) + np.asarray(_supply_scale(b, q)) + gross_q)
        worst[q] = judge(q, rel, step_rel_tol)
        where[q] = int(np.argmax(np.nan_to_num(rel, nan=np.inf)))
        for proc, supply in _PROCESS_SUPPLY.items():
            r, s_ = b[f"{proc}_{q}"], np.asarray(supply(b, q))
            wproc[f"{proc}_{q}"] = judge(f"{proc} {q}", _rel(r - s_, b[f"gross_{proc}_{q}"]
                                                             + np.abs(s_)), step_rel_tol)
        x = np.concatenate([np.asarray(tot_in[q]), np.asarray(tot_f[q])[None]])
        gross = np.concatenate([np.asarray(scale_in[q]), np.asarray(scale_f[q])[None]])
        wstate[q] = judge(f"{q} state", _rel(np.diff(x) - dt * rate, gross[:-1]), state_rel_tol)
        wrun[q] = judge(f"{q} whole-run state",
                        _rel(np.asarray([x[-1] - x[0] - dt * np.sum(sup)]),
                             np.asarray([np.max(gross)])), state_rel_tol)
    for key in ("record_rho_rel", "record_exner_rel"):
        judge(key.replace("_", " "), b[key], step_rel_tol)
    rh = np.asarray(rec.inputs.rho_half[..., 0], np.float64)
    ex = np.asarray(rec.inputs.exner_half[..., 0], np.float64)
    for f, (q, law) in _OWN_FLUX.items():
        key = "own_sfc_" + f
        if key not in b:
            bad.append(f"{f}: the closure's own surface flux was not recorded")
            continue
        d = rh * (b[key] - np.asarray(getattr(rec.inputs, law), np.float64))
        if q == "energy":
            d = d * constants.c_pd * ex
        own[f] = judge(f"{f}: closure's own surface flux vs the law",
                       _rel(d, b[f"gross_turbulence_{q}"] + np.abs(b[f"sfc_{q}"])),
                       own_boundary_tol)
    return ConservationVerdict(passed=not bad, worst_step_rel=worst, worst_step_index=where,
                               worst_process_rel=wproc, worst_state_rel=wstate,
                               run_state_rel=wrun, own_boundary_rel=own,
                               reason="; ".join(bad))
