"""Tune every SCM turbulence closure against a matched LES reference.

Runs a single-column model on the SAME initial sounding, large-scale forcing
and surface boundary condition as an LES of the same case, once per selectable
turbulence scheme, and scores the resulting mean-state profiles against the
LES time-mean. Then gradient-tunes each scheme's spec'd parameters against that
same reference.

CONTROLLED COMPARISON is the whole point, so the things that make it one are
mechanical rather than trusted:

* Every scheme is handed a byte-identical column and forcing built once by
  :mod:`legoesm.atmosphere.forcing.scm.sam_case_scm`; the ONLY difference
  between arms is ``PhysicsConfig.turbulence``. ``_assert_arms_differ_only_in_turbulence``
  checks that rather than assuming it.
* Every scheme is scored over the SAME trailing analysis window as the LES
  reference, on the same levels, with the same mass weights and the same
  metric. The window is printed next to every number.
* Radiation, convection and microphysics are held identical across arms.

Case matching, which is not negotiable:

* BOMEX and RICO apply NO radiation in the LES (``run_bomex_les.py`` has no
  radiation at all; the deck's ``lsf`` temperature tendency is the entire
  thermal forcing), so an SCM with ``radiation="none"`` is an exact match.
* DYCOMS applies a parameterized Stevens (2005) longwave cooling in the LES
  (``run_dycoms_les.py`` ``make_stevens_lw``) that the SCM has no equivalent
  for. Without cloud-top radiative cooling the SCM stratocumulus is a
  different problem, so DYCOMS is REFUSED here rather than scored into a
  confounded number. Porting that forcing to the SCM side is what unblocks it.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_scm_les_turbulence_tuning.py \\
        --case bomex --les-dir results/les_ref/bomex
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from legoesm.atmosphere.forcing.scm.sam_case_scm import (  # noqa: E402
    SAM_SCM_CASES,
    load_sam_scm_case,
)
from legoesm.atmosphere.physics.combined import PhysicsConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    TurbulenceConfig,
)
from legoesm.ml.training import TrainingConfig, create_optimizer  # noqa: E402
from legoesm.training.les_reference import (  # noqa: E402
    load_les_reference,
)
from legoesm.training.param_collector import (  # noqa: E402
    apply_param_overrides,
    build_registry,
    build_trainable_params,
)
from legoesm.training.scm_rce_metrics import (  # noqa: E402
    normalized_profile_rmse,
    safe_sqrt,
)
from legoesm.training.trainable_params import (  # noqa: E402
    ParamConstraint,
    TrainablePhysicsParams,
    range_to_sigmoid_array,
)

# Every selectable closure in turbulence/integration.py::get_turbulence_fn,
# minus "none" (which is the absence of a closure, not a candidate).
TURBULENCE_SCHEMES: tuple[str, ...] = (
    "smagorinsky", "louis", "tke", "mynn25", "clubb_lite", "clubb",
    "holtslag_boville", "ysu", "edmf",
)

# Cases whose LES applies a radiative forcing the SCM cannot reproduce. Scoring
# these would compare two different problems; see the module docstring.
_RADIATION_MISMATCH = {
    "dycoms": (
        "the DYCOMS LES applies a parameterized Stevens (2005) longwave "
        "cooling (run_dycoms_les.py make_stevens_lw) that the SCM has no "
        "equivalent for. Without cloud-top radiative cooling the SCM "
        "stratocumulus is a different problem and any score would be a "
        "radiation confound, not a turbulence result. Port that forcing to "
        "the SCM side before tuning this case."
    ),
}

DEFAULT_OUTDIR = Path("results/scm_les_turbulence")
DEFAULT_NLEV = 64
DEFAULT_DT_S = 60.0
DEFAULT_ANALYSIS_HOURS = 2.0
DEFAULT_CHUNK_STEPS = 60
DEFAULT_STEPS = 8
DEFAULT_LR = 1.0e-1
GRAD_NONZERO_TOL = 1.0e-14
# Floor on the reference's own spread, so a nearly-uniform profile (u, v in a
# case with weak shear) cannot divide the score by ~0.
PROFILE_FLOOR = 1.0e-8
_LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)


# --------------------------------------------------------------------------
# configuration: identical across arms except turbulence
# --------------------------------------------------------------------------

def build_physics_config(scheme: str, *, prescribed_fluxes: bool) -> PhysicsConfig:
    """PhysicsConfig with ONLY the turbulence scheme varying.

    ``prescribed_fluxes`` zeroes the bulk exchange coefficient for heat on the
    active scheme's surface sub-config. The SCM refuses ``prescribe="fluxes"``
    alongside a live bulk formula (it would double-count the deck's SHF/LHF),
    and the zeroing has to happen identically on every arm or the arms stop
    being comparable.
    """
    if scheme not in TURBULENCE_SCHEMES:
        raise ValueError(
            f"Unknown turbulence scheme {scheme!r}; choose from "
            f"{list(TURBULENCE_SCHEMES)}."
        )
    turb = TurbulenceConfig(scheme=scheme)
    if scheme == "clubb" and getattr(turb, "clubb", None) is None:
        turb = turb._replace(clubb=CLUBBConfig())
    if prescribed_fluxes:
        sub = getattr(turb, scheme)
        surface = sub.surface._replace(Ch_neutral=0.0)
        turb = turb._replace(**{scheme: sub._replace(surface=surface)})
    return PhysicsConfig(
        turbulence=turb,
        radiation=_scheme_none(PhysicsConfig().radiation),
        convection=_scheme_none(PhysicsConfig().convection),
        microphysics=_scheme_none(PhysicsConfig().microphysics),
        gravity_wave_drag=_scheme_none(PhysicsConfig().gravity_wave_drag),
    )


def _scheme_none(component):
    return component._replace(scheme="none")


def _assert_arms_differ_only_in_turbulence(configs: dict[str, PhysicsConfig]) -> None:
    """Every non-turbulence component must be identical across arms."""
    fields = [f for f in PhysicsConfig._fields if f != "turbulence"]
    reference_name = next(iter(configs))
    reference = configs[reference_name]
    for name, cfg in configs.items():
        for field in fields:
            if getattr(cfg, field) != getattr(reference, field):
                raise RuntimeError(
                    f"arm {name!r} differs from {reference_name!r} in "
                    f"PhysicsConfig.{field}; the comparison would not be "
                    "controlled."
                )


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def _create_scm(cfg: PhysicsConfig, case, dt: float):
    return case.create_scm(
        physics_config=cfg, dt=dt, dtype=jnp.float64,
        time_integrator="forward_euler",
    )


def _make_step_once(scm, dt: float):
    """One pure ``(carry, k) -> (carry, out)`` scan body.

    Uses the SCM's OWN tendency closure (``scm._tend_fn``), which applies the
    physics AND the SCMForcing (subsidence, advective tendencies, prescribed
    surface fluxes / skin temperature). Calling ``physics_fn`` directly instead
    would silently drop the entire large-scale forcing, which for these cases
    is most of the problem.
    """
    step_fn = scm._step_fn
    tend_fn = scm._tend_fn
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)

    def step_once(carry, k):
        state, phys_state = carry
        t = k.astype(jnp.float64) * dt_arr
        new_state, new_phys = step_fn(state, phys_state, tend_fn, dt, t)
        out = (
            new_state.T.data[0, 0, 0],
            new_state.tracers["q_v"].data[0, 0, 0],
            new_state.u.data[0, 0, 0],
            new_state.v.data[0, 0, 0],
            new_state.p_s.data[0, 0],
        )
        return (new_state, new_phys), out

    return step_once


def _rollout_means(
    params: TrainablePhysicsParams | None,
    *,
    base_cfg: PhysicsConfig,
    case,
    dt: float,
    hours: float,
    analysis_hours: float,
    chunk_steps: int,
):
    """Integrate the column and return the analysis-window time-mean profiles.

    The SCM is rebuilt here, INSIDE the differentiated region, on purpose: the
    physics closure bakes config scalars in at construction, so building it
    once outside would capture concrete floats and give an identically zero
    gradient.
    """
    cfg = base_cfg if params is None else _apply_trainable_params(base_cfg, params)
    scm = _create_scm(cfg, case, dt)
    step_once = jax.checkpoint(_make_step_once(scm, dt))

    nsteps = max(1, int(round(hours * 3600.0 / dt)))
    analysis_steps = max(1, int(round(analysis_hours * 3600.0 / dt)))
    if analysis_steps > nsteps:
        raise ValueError(
            f"analysis window ({analysis_hours} h) exceeds the run ({hours} h)."
        )
    chunk_steps = max(1, int(chunk_steps))
    nchunks = int(math.ceil(nsteps / chunk_steps))
    total_steps = nchunks * chunk_steps
    nlev = case.nlev

    def masked_step(carry, k):
        zeros = (
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((), dtype=jnp.float64),
        )
        return lax.cond(
            k < nsteps,
            lambda c: step_once(c, k),
            lambda c: (c, zeros),
            carry,
        )

    def chunk_body(carry, chunk_index):
        ks = chunk_index * chunk_steps + jnp.arange(chunk_steps)
        return lax.scan(masked_step, carry, ks)

    chunk_body = jax.checkpoint(chunk_body)
    carry0 = (scm.state, scm.phys_state)
    _final, history = lax.scan(chunk_body, carry0, jnp.arange(nchunks))

    T_h, qv_h, u_h, v_h, ps_h = (
        x.reshape((total_steps,) + x.shape[2:])[:nsteps] for x in history
    )
    window = slice(nsteps - analysis_steps, nsteps)
    means = {
        "T": jnp.mean(T_h[window], axis=0),
        "qv": jnp.mean(qv_h[window], axis=0),
        "u": jnp.mean(u_h[window], axis=0),
        "v": jnp.mean(v_h[window], axis=0),
    }
    return means, ps_h


def _theta_from_T(T_profile, p_full):
    """theta = T / Pi on the SCM's own (time-invariant) pressure levels.

    p_s carries no tendency in these cases, so Exner is a constant of the run;
    ``_assert_surface_pressure_static`` verifies that rather than assuming it.
    """
    from legoesm import constants
    exner = (jnp.asarray(p_full) / constants.p_ref) ** constants.kappa
    return jnp.asarray(T_profile) / exner


def _assert_surface_pressure_static(ps_history, p_s: float, tol_pa: float = 1.0):
    drift = float(jnp.max(jnp.abs(jnp.asarray(ps_history) - p_s)))
    if drift > tol_pa:
        raise RuntimeError(
            f"surface pressure drifted {drift:.3f} Pa (> {tol_pa} Pa) during "
            "the run, so Exner is not a constant of the run and the "
            "theta conversion in _theta_from_T is invalid."
        )
    return drift


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_against_les(means, *, reference, p_full):
    """Per-variable and combined normalized profile score vs the LES.

    Each variable is normalized by the LES profile's own mass-weighted spread
    (:func:`normalized_profile_rmse`), which is what makes K, kg/kg and m/s
    commensurable, then combined in quadrature.
    """
    mask = jnp.asarray(reference.mask)
    weights = jnp.asarray(reference.weights, dtype=jnp.float64)
    predicted = {
        "theta": _theta_from_T(means["T"], p_full),
        "qv": means["qv"],
        "u": means["u"],
        "v": means["v"],
    }
    components = {}
    for name in reference.scored_variables():
        ref_profile = jnp.asarray(
            np.nan_to_num(reference.profiles[name], nan=0.0), dtype=jnp.float64
        )
        pred = jnp.where(mask, predicted[name], 0.0)
        components[name] = normalized_profile_rmse(
            ref_profile, pred, weights, profile_floor=PROFILE_FLOOR,
        )
    stacked = jnp.stack([components[k] for k in sorted(components)])
    combined = safe_sqrt(jnp.mean(stacked ** 2))
    return components, combined


# --------------------------------------------------------------------------
# trainable parameters
# --------------------------------------------------------------------------

def _scheme_keys_for(scheme: str) -> set[str]:
    """Registry scheme keys carrying the tunable leaves of ``scheme``.

    Full CLUBB nests its closure coefficients one level down in ``.params``
    (a ``CLUBBParams``), and it is that nested tuple which carries the
    ``__param_spec__``, so the key is CLUBBParams rather than CLUBBConfig.
    """
    cfg = build_physics_config(scheme, prescribed_fluxes=False)
    sub = getattr(cfg.turbulence, scheme)
    target = sub.params if scheme == "clubb" else sub
    wanted = type(target).__name__
    keys = {m.scheme_key for m in build_registry()
            if m.config_class == wanted}
    if not keys:
        raise RuntimeError(
            f"no __param_spec__ registered for {wanted} (scheme {scheme!r}); "
            "it cannot be tuned."
        )
    return keys


def _apply_trainable_params(base_cfg: PhysicsConfig,
                            params: TrainablePhysicsParams) -> PhysicsConfig:
    """Splice traced leaves into the active turbulence sub-config."""
    scheme = base_cfg.turbulence.scheme
    sub = getattr(base_cfg.turbulence, scheme)
    overrides = params.to_overrides()
    for _scheme_key, field_values in overrides.items():
        if scheme == "clubb":
            sub = sub._replace(
                params=apply_param_overrides(sub.params, field_values)
            )
        else:
            sub = apply_param_overrides(sub, field_values)
    return base_cfg._replace(
        turbulence=base_cfg.turbulence._replace(**{scheme: sub})
    )


def _raw_from_physical(value: float, constraint: ParamConstraint):
    arr = jnp.asarray(value, dtype=jnp.float64)
    if constraint.transform == "sigmoid":
        return range_to_sigmoid_array(
            arr, constraint.min_val, constraint.max_val,
        ).astype(jnp.float64)
    if constraint.transform == "softplus":
        y = jnp.maximum(arr, jnp.asarray(PROFILE_FLOOR, dtype=jnp.float64))
        # overflow-safe inverse of softplus: log(exp(y)-1) == y + log1p(-exp(-y))
        return (y + jnp.log1p(-jnp.exp(-y))).astype(jnp.float64)
    return arr


def _initial_params(scheme: str, tier: str) -> TrainablePhysicsParams:
    return build_trainable_params(
        active_scheme_keys=_scheme_keys_for(scheme),
        tier=tier,
        dtype=jnp.float64,
    )


def _grad_stats(grads: TrainablePhysicsParams, tol: float) -> dict[str, dict]:
    stats = {}
    for constraint in grads.constraints:
        grad = grads.raw_values[constraint.name]
        abs_max = float(jnp.max(jnp.abs(grad)))
        finite = bool(jnp.all(jnp.isfinite(grad)))
        stats[constraint.name] = {
            "abs_max": abs_max,
            "finite": finite,
            "nonzero": bool(finite and abs_max > tol),
        }
    return stats


def _filter_params(params: TrainablePhysicsParams,
                   keep: set[str]) -> TrainablePhysicsParams:
    constraints = [c for c in params.constraints if c.name in keep]
    raw = {c.name: params.raw_values[c.name] for c in constraints}
    return TrainablePhysicsParams(raw_values=raw, constraints=constraints)


def _assert_strict_bounds(params: TrainablePhysicsParams) -> None:
    physical = params.as_dict()
    for constraint in params.constraints:
        value = physical[constraint.name]
        if not bool(jnp.all(jnp.isfinite(value))):
            raise RuntimeError(f"{constraint.name} is non-finite after transform")
        lo = jnp.asarray(constraint.min_val, dtype=value.dtype)
        hi = jnp.asarray(constraint.max_val, dtype=value.dtype)
        if not bool(jnp.all((value > lo) & (value < hi))):
            raise RuntimeError(
                f"{constraint.name}={np.asarray(value)} escaped strict bounds "
                f"({constraint.min_val}, {constraint.max_val})"
            )


# --------------------------------------------------------------------------
# per-scheme evaluation and tuning
# --------------------------------------------------------------------------

@dataclass
class SchemeResult:
    scheme: str
    status: str
    score_default: float | None = None
    score_tuned: float | None = None
    components_default: dict[str, float] | None = None
    components_tuned: dict[str, float] | None = None
    n_trained: int = 0
    frozen: dict[str, str] | None = None
    parameters: list[dict[str, Any]] | None = None
    loss_history: list[float] | None = None
    error: str | None = None
    wall_s: float = 0.0
    ps_drift_pa: float | None = None


def evaluate_scheme(scheme: str, *, case, reference, args) -> tuple:
    cfg = build_physics_config(
        scheme, prescribed_fluxes=(case.forcing.prescribe == "fluxes"),
    )
    means, ps_hist = _rollout_means(
        None, base_cfg=cfg, case=case, dt=args.dt, hours=args.hours,
        analysis_hours=args.analysis_hours, chunk_steps=args.chunk_steps,
    )
    drift = _assert_surface_pressure_static(ps_hist, case.p_s)
    components, combined = score_against_les(
        means, reference=reference, p_full=case.p_full,
    )
    return cfg, means, components, combined, drift


def tune_scheme(scheme: str, *, case, reference, args, base_cfg) -> SchemeResult:
    result = SchemeResult(scheme=scheme, status="tuned")
    t0 = time.time()

    params_all = _initial_params(scheme, args.tier)
    if not params_all.constraints:
        result.status = "no_tunable_params"
        result.wall_s = time.time() - t0
        return result

    def loss_fn(params: TrainablePhysicsParams):
        means, _ps = _rollout_means(
            params, base_cfg=base_cfg, case=case, dt=args.dt, hours=args.hours,
            analysis_hours=args.analysis_hours, chunk_steps=args.chunk_steps,
        )
        _components, combined = score_against_les(
            means, reference=reference, p_full=case.p_full,
        )
        return combined

    # Preflight: permanently freeze any parameter the loss does not depend on,
    # recording why. A dead gradient here is information about the scheme, not
    # a reason to stop.
    preflight_loss, preflight_grads = eqx.filter_value_and_grad(loss_fn)(params_all)
    if not np.isfinite(float(preflight_loss)):
        result.status = "failed"
        result.error = f"preflight loss non-finite: {float(preflight_loss)}"
        result.wall_s = time.time() - t0
        return result
    stats = _grad_stats(preflight_grads, args.grad_nonzero_tol)
    frozen = {
        name: ("non-finite gradient" if not s["finite"]
               else f"|grad| <= {args.grad_nonzero_tol:g} in preflight")
        for name, s in stats.items() if not s["nonzero"]
    }
    keep = {name for name, s in stats.items() if s["nonzero"]}
    result.frozen = frozen
    if not keep:
        result.status = "no_active_gradient"
        result.wall_s = time.time() - t0
        return result

    params = _filter_params(params_all, keep)
    _assert_strict_bounds(params)
    result.n_trained = len(params.constraints)

    optimizer = create_optimizer(TrainingConfig(
        lr=args.lr, warmup_steps=args.warmup_steps,
        total_steps=max(args.steps, 1), grad_clip_norm=args.grad_clip_norm,
        optimizer=args.optimizer,
    ))
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    loss_history = [float(preflight_loss)]

    for step in range(1, args.steps + 1):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        loss_val = float(loss)
        step_stats = _grad_stats(grads, args.grad_nonzero_tol)
        bad = {n: s for n, s in step_stats.items()
               if not s["finite"] or not s["nonzero"]}
        if bad:
            result.error = f"gradient gate failed at step {step}: {bad}"
            result.status = "failed"
            break
        updates, opt_state_candidate = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(params, eqx.is_array),
        )
        accepted = False
        for scale in _LINE_SEARCH_SCALES:
            candidate = eqx.apply_updates(
                params, jax.tree_util.tree_map(lambda x: x * scale, updates),
            )
            _assert_strict_bounds(candidate)
            candidate_loss = float(loss_fn(candidate))
            if np.isfinite(candidate_loss) and candidate_loss < loss_val:
                params, opt_state = candidate, opt_state_candidate
                loss_history.append(candidate_loss)
                accepted = True
                break
        print(f"    [{scheme}] step {step}/{args.steps} loss={loss_val:.6g} "
              f"{'accepted' if accepted else 'no reducing step -> stop'}",
              flush=True)
        if not accepted:
            break

    result.loss_history = loss_history
    tuned_cfg = _apply_trainable_params(base_cfg, params)
    means, ps_hist = _rollout_means(
        None, base_cfg=_materialize_static(tuned_cfg), case=case, dt=args.dt,
        hours=args.hours, analysis_hours=args.analysis_hours,
        chunk_steps=args.chunk_steps,
    )
    result.ps_drift_pa = _assert_surface_pressure_static(ps_hist, case.p_s)
    components, combined = score_against_les(
        means, reference=reference, p_full=case.p_full,
    )
    result.score_tuned = float(combined)
    result.components_tuned = {k: float(v) for k, v in components.items()}

    meta_by_name = {f"{m.scheme_key}.{m.field}": m for m in build_registry()}
    physical = params.as_dict()
    all_physical = params_all.as_dict()
    rows = []
    for constraint in params_all.constraints:
        meta = meta_by_name.get(constraint.name)
        trained = constraint.name in {c.name for c in params.constraints}
        rows.append({
            "name": constraint.name,
            "units": getattr(meta, "units", None),
            "default": float(np.asarray(all_physical[constraint.name])),
            "tuned": (float(np.asarray(physical[constraint.name])) if trained
                      else None),
            "lower": constraint.min_val,
            "upper": constraint.max_val,
            "trained": trained,
            "frozen_reason": frozen.get(constraint.name),
        })
    result.parameters = rows
    result.wall_s = time.time() - t0
    return result


def _materialize_static(cfg: PhysicsConfig) -> PhysicsConfig:
    """Turn traced 0-d arrays back into Python floats for a non-traced rerun."""
    return jax.tree_util.tree_map(
        lambda x: float(x) if isinstance(x, jax.Array) and x.ndim == 0 else x,
        cfg,
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=sorted(SAM_SCM_CASES))
    p.add_argument("--les-dir", required=True, type=Path,
                   help="LES output directory containing profiles/")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--schemes", default="all",
                   help="comma-separated subset, or 'all'")
    p.add_argument("--nlev", type=int, default=DEFAULT_NLEV)
    p.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    p.add_argument("--hours", type=float, default=None,
                   help="SCM run length; defaults to the LES record length")
    p.add_argument("--analysis-hours", type=float,
                   default=DEFAULT_ANALYSIS_HOURS)
    p.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS)
    p.add_argument("--tier", default="extended",
                   choices=("core", "extended", "aggressive"))
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--optimizer", default="muon",
                   choices=("muon", "muon_partitioned", "adam", "adamw"))
    p.add_argument("--grad-nonzero-tol", type=float, default=GRAD_NONZERO_TOL)
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--allow-radiation-mismatch", action="store_true",
                   help="run a case whose LES radiation the SCM cannot match; "
                        "the result is a confound and is labelled as one")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JAX_ENABLE_X64=1 is required for this driver.")

    if args.case in _RADIATION_MISMATCH and not args.allow_radiation_mismatch:
        raise SystemExit(
            f"refusing to tune {args.case!r}: {_RADIATION_MISMATCH[args.case]}\n"
            "Pass --allow-radiation-mismatch to override; the output will be "
            "labelled a confound."
        )

    schemes = (list(TURBULENCE_SCHEMES) if args.schemes == "all"
               else [s.strip() for s in args.schemes.split(",") if s.strip()])
    unknown = [s for s in schemes if s not in TURBULENCE_SCHEMES]
    if unknown:
        raise SystemExit(
            f"unknown scheme(s) {unknown}; choose from {list(TURBULENCE_SCHEMES)}"
        )

    outdir = args.outdir or (DEFAULT_OUTDIR / args.case)
    outdir.mkdir(parents=True, exist_ok=True)

    case = load_sam_scm_case(args.case, nlev=args.nlev, dt=args.dt)
    hours = args.hours
    reference = load_les_reference(
        args.les_dir, case=args.case, z_scm=case.z_full,
        p_half=_half_pressures(case),
        domain_top_m=case.les_domain_top_m,
        analysis_hours=args.analysis_hours,
    )
    if hours is None:
        hours = reference.window_hours[1]
    args.hours = hours

    print(f"case={args.case} nlev={args.nlev} dt={args.dt}s hours={hours} "
          f"analysis={args.analysis_hours}h")
    print(f"LES reference: {reference.source_dir}")
    print(f"  window={reference.window_label} levels_in_domain="
          f"{int(reference.mask.sum())}/{case.nlev}")
    print(f"  scored variables: {list(reference.scored_variables())}")
    print(f"  surface: prescribe={case.forcing.prescribe}")

    configs = {}
    profiles_default: dict[str, dict[str, np.ndarray]] = {}
    results: list[SchemeResult] = []
    for scheme in schemes:
        print(f"\n[eval] {scheme}", flush=True)
        res = SchemeResult(scheme=scheme, status="ok")
        t0 = time.time()
        try:
            cfg, means, components, combined, drift = evaluate_scheme(
                scheme, case=case, reference=reference, args=args,
            )
            configs[scheme] = cfg
            profiles_default[scheme] = {
                "theta": np.asarray(_theta_from_T(means["T"], case.p_full)),
                "qv": np.asarray(means["qv"]),
                "u": np.asarray(means["u"]),
                "v": np.asarray(means["v"]),
            }
            res.score_default = float(combined)
            res.components_default = {k: float(v) for k, v in components.items()}
            res.ps_drift_pa = drift
            print(f"    score={res.score_default:.6g} "
                  + " ".join(f"{k}={v:.4g}"
                             for k, v in res.components_default.items()))
        except Exception as exc:                      # noqa: BLE001
            res.status = "failed"
            res.error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED {res.error}", flush=True)
        res.wall_s = time.time() - t0
        results.append(res)

    if configs:
        _assert_arms_differ_only_in_turbulence(configs)
        print("\n[control] arms verified identical outside "
              "PhysicsConfig.turbulence")

    if not args.skip_tuning:
        for res in results:
            if res.status != "ok":
                continue
            print(f"\n[tune] {res.scheme}", flush=True)
            try:
                tuned = tune_scheme(
                    res.scheme, case=case, reference=reference, args=args,
                    base_cfg=configs[res.scheme],
                )
            except Exception as exc:                  # noqa: BLE001
                res.status = "tune_failed"
                res.error = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED {res.error}", flush=True)
                continue
            res.status = tuned.status
            res.score_tuned = tuned.score_tuned
            res.components_tuned = tuned.components_tuned
            res.n_trained = tuned.n_trained
            res.frozen = tuned.frozen
            res.parameters = tuned.parameters
            res.loss_history = tuned.loss_history
            if tuned.error:
                res.error = tuned.error
            if res.score_tuned is not None:
                print(f"    default={res.score_default:.6g} -> "
                      f"tuned={res.score_tuned:.6g} "
                      f"({res.n_trained} params trained)")

    _write_outputs(outdir, args, case, reference, results)
    _write_profiles(outdir, case, reference, profiles_default)
    print(f"\nwrote {outdir}")
    return 0


def _write_profiles(outdir: Path, case, reference,
                    profiles_default: dict[str, dict]) -> None:
    """Save the LES reference and every arm's mean profiles for plotting.

    Saved on the SCM levels with the LES-domain mask alongside, so a plot
    cannot silently draw the extrapolated region.
    """
    payload = {
        "z_scm": np.asarray(case.z_full),
        "p_full": np.asarray(case.p_full),
        "mask": np.asarray(reference.mask),
        "weights": np.asarray(reference.weights),
        "z_les": np.asarray(reference.z_les),
        "window_hours": np.asarray(reference.window_hours),
    }
    for name, profile in reference.profiles.items():
        payload[f"les_scmlev_{name}"] = np.asarray(profile)
    for name, profile in reference.profiles_les.items():
        payload[f"les_native_{name}"] = np.asarray(profile)
    for scheme, prof in profiles_default.items():
        for name, values in prof.items():
            payload[f"scm_{scheme}_{name}"] = np.asarray(values)
    np.savez(outdir / "profiles.npz", **payload)


def _half_pressures(case) -> np.ndarray:
    """SCM half-level pressures [Pa] from the case's sigma column."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(
        case.nlev, sigma_top=case.sigma_top, dtype=jnp.float64,
    )
    return np.asarray(sigma.sigma_half, dtype=np.float64) * case.p_s


def _write_outputs(outdir: Path, args, case, reference, results) -> None:
    ranked = sorted(
        results,
        key=lambda r: (r.score_tuned if r.score_tuned is not None
                       else r.score_default if r.score_default is not None
                       else float("inf")),
    )
    with (outdir / "ranking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "status", "score_default", "score_tuned",
                    "n_trained", "wall_s", "error"])
        for r in ranked:
            w.writerow([r.scheme, r.status, r.score_default, r.score_tuned,
                        r.n_trained, f"{r.wall_s:.1f}", r.error or ""])

    payload = {
        "case": args.case,
        "protocol": {
            "nlev": args.nlev, "dt_s": args.dt, "hours": args.hours,
            "analysis_hours": args.analysis_hours,
            "tier": args.tier, "optimizer": args.optimizer, "lr": args.lr,
            "steps": args.steps,
            "surface_prescribe": case.forcing.prescribe,
            "radiation": "none", "convection": "none", "microphysics": "none",
            "scored_variables": list(reference.scored_variables()),
            "note": (
                "All arms share one column and forcing built from the same "
                "gSAM deck as the LES; only PhysicsConfig.turbulence differs, "
                "verified mechanically. Turbulent fluxes are not scored "
                "because the SCM does not expose a per-level w'theta' for 8 "
                "of the 9 schemes."
            ),
        },
        "les_reference": {
            "source_dir": reference.source_dir,
            "window_hours": list(reference.window_hours),
            "n_frames": reference.n_frames,
            "levels_in_domain": int(reference.mask.sum()),
        },
        "schemes": [
            {
                "scheme": r.scheme, "status": r.status,
                "score_default": r.score_default, "score_tuned": r.score_tuned,
                "components_default": r.components_default,
                "components_tuned": r.components_tuned,
                "n_trained": r.n_trained, "frozen": r.frozen,
                "parameters": r.parameters, "loss_history": r.loss_history,
                "surface_pressure_drift_pa": r.ps_drift_pa,
                "error": r.error, "wall_s": r.wall_s,
            }
            for r in ranked
        ],
    }
    (outdir / "tuned_parameters.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        f"# SCM turbulence closures vs LES — {args.case}", "",
        f"LES reference: `{reference.source_dir}`, window "
        f"{reference.window_label}, {int(reference.mask.sum())} of "
        f"{case.nlev} SCM levels inside the LES domain.", "",
        f"SCM: nlev={args.nlev}, dt={args.dt} s, {args.hours} h, "
        f"trailing {args.analysis_hours} h averaged (same window as the LES).",
        "",
        "Every arm shares one initial column, one large-scale forcing and one "
        "surface boundary condition, all built from the same gSAM deck the LES "
        "read; only `PhysicsConfig.turbulence` differs, and that is checked "
        "mechanically rather than assumed. Scored on "
        f"{', '.join(reference.scored_variables())} — normalized by the LES "
        "profile's own mass-weighted spread, combined in quadrature. Lower is "
        "better.", "",
        "| scheme | status | score (default) | score (tuned) | params trained |",
        "|---|---|---|---|---|",
    ]
    for r in ranked:
        d = "—" if r.score_default is None else f"{r.score_default:.4f}"
        t = "—" if r.score_tuned is None else f"{r.score_tuned:.4f}"
        lines.append(f"| {r.scheme} | {r.status} | {d} | {t} | {r.n_trained} |")
    failures = [r for r in ranked if r.error]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- **{r.scheme}** ({r.status}): {r.error}" for r in failures]
    lines += [
        "", "## Caveats", "",
        "- Turbulent fluxes are NOT scored. The SCM's hydrostatic turbulence "
        "driver drops `Km`/`Kh`/`shflx`/`lhflx` from `TurbulenceOutput`, and "
        "only prognostic CLUBB carries a per-level `w'theta_l'`, so a flux "
        "score would exist for 1 of 9 schemes.",
        "- Radiation, convection and microphysics are off, identically on "
        "every arm. For BOMEX/RICO that matches the LES exactly: the deck's "
        "`lsf` temperature tendency IS the case's radiative cooling "
        "(-2.0 K/day below 1500 m, tapering to 0 by 2500 m for BOMEX) and both "
        "models read it from the same file. It is why DYCOMS is refused.",
        "- The averaging WINDOW is identical on both sides, but the sampling "
        "inside it is not: the LES mean is over its saved frames (10-minute "
        "cadence), the SCM mean is over every timestep. Same interval, "
        "different estimator variance — not a window confound, but do not "
        "quote a difference smaller than the LES frame-to-frame scatter.",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
