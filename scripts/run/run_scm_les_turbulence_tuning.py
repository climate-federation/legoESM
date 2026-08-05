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
from legoesm.atmosphere.forcing.scm.analytic_scm_case import (  # noqa: E402
    ANALYTIC_SCM_CASES,
    load_analytic_scm_case,
)
from legoesm.atmosphere.forcing.scm.sam_case_scm import (  # noqa: E402
    SAM_SCM_CASES,
    load_sam_scm_case,
)
from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import exner_function  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    TurbulenceConfig,
)
from legoesm.core.bulk_flux import neutral_drag_coefficient  # noqa: E402
from legoesm.ml.training import TrainingConfig, create_optimizer  # noqa: E402
from legoesm.training.les_reference import (  # noqa: E402
    SCORED_VARIABLES,
    load_les_reference,
)
from legoesm.core.param_overrides import apply_param_overrides  # noqa: E402
from legoesm.training.param_collector import (  # noqa: E402
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

ALL_CASES: tuple[str, ...] = tuple(sorted(SAM_SCM_CASES)) + tuple(
    sorted(ANALYTIC_SCM_CASES))


def load_case(name: str, *, nlev: int, dt: float | None):
    """Load a case from whichever registry owns it."""
    if name in SAM_SCM_CASES:
        return load_sam_scm_case(name, nlev=nlev, dt=dt)
    if name in ANALYTIC_SCM_CASES:
        return load_analytic_scm_case(name, nlev=nlev, dt=dt)
    raise ValueError(f"Unknown case {name!r}; choose from {list(ALL_CASES)}")


def case_scored(name: str, override: tuple[str, ...] | None):
    """Variables this CASE is scored on.

    Per-case, not global: a neutral Ekman layer has no theta signal to speak of
    (constant by construction, so normalising by its own spread divides by the
    floor), a dry case has no q_v at all, and CBL has neither rotation nor a
    geostrophic wind so its winds stay ~0.
    """
    if override:
        return tuple(override)
    if name in ANALYTIC_SCM_CASES:
        return tuple(ANALYTIC_SCM_CASES[name].scored)
    return ("theta", "qv")


@dataclass
class CaseArm:
    """One case in a multi-case campaign, with everything it needs."""
    name: str
    case: Any
    reference: Any
    scored: tuple[str, ...]
    hours: float
    analysis_hours: float
    dt: float
    chunk_steps: int
    surface: Any
    prescribed_fluxes: bool
    les_dir: str


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
# Score assigned to a non-finite rollout. Large enough to lose every
# comparison, finite so gradients and the line search still work.
NONFINITE_PENALTY = 1.0e3
# The SCM's window may miss the LES window by at most this fraction of the
# window itself. Not a fraction of the timestep: rounding already bounds that
# residual by half a step, so a step-based test is vacuous. 1% of a 2 h window
# is 72 s, far below the LES frame cadence, while GABLS1's real 0.1 s endpoint
# overshoot is 0.0014% and passes.
_WINDOW_TOL_FRAC = 0.01
_LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)
# Registry namespace for the ATMOSPHERIC turbulence configs. Class names alone
# collide across components (the ocean also registers a TKEConfig).
_ATM_TURB_NAMESPACE = "atm.turb."


# --------------------------------------------------------------------------
# configuration: identical across arms except turbulence
# --------------------------------------------------------------------------

def build_surface_config(case, *, bulk_scheme: str = "constant"):
    """The ONE surface-layer config every arm uses, derived from case physics.

    The SCM default is a fixed ``Cd_neutral = 1.5e-3``, while every one of
    these LES cases drives its momentum stress with a MOST wall model at the
    case roughness. At BOMEX's z0 = 1e-4 m and a ~20 m lowest level the neutral
    log law gives Cd = 1.07e-3 and u* = 0.287 m/s at the 8.75 m/s trade wind --
    which is the LES's u* -- whereas the SCM default gives 0.339 m/s, a 18%
    surface-stress error the closures would otherwise be tuned to compensate
    for.

    So ``Cd_neutral`` is DERIVED as the neutral-MOST drag
    ``kappa^2 / ln^2(z_ref / z0)`` at the case's own z0, evaluated at the SCM's
    lowest full level (the level whose wind ``compute_surface_fluxes`` is
    handed) rather than the 10 m default, which would be inconsistent with
    that wind.

    ``bulk_scheme`` stays "constant" for these cases ON PURPOSE. All three
    prescribe or fix their surface HEAT flux (BOMEX/DYCOMS prescribe SHF/LHF,
    RICO uses fixed van Zanten C_H/C_Q), and the iterative "most" path derives
    heat from MOST scaling while ignoring Ch_neutral -- it would double-count a
    prescribed flux. "constant" with a log-law-derived Cd reproduces the LES
    momentum drag while leaving the heat channel to the case definition.
    """
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    z0 = float(case.spec.les_z0_m)
    z_ref = float(case.z_full[-1])
    if not z0 > 0.0 or not z_ref > z0:
        raise ValueError(
            f"need 0 < z0 ({z0}) < z_ref ({z_ref}) for a log-law drag."
        )
    cd_neutral = float(neutral_drag_coefficient(z_ref, z0))
    prescribed = case.forcing.prescribe == "fluxes"
    if prescribed:
        ch_neutral = 0.0            # heat comes from the prescribed channel
    elif case.spec.bulk_ch is not None:
        ch_neutral = float(case.spec.bulk_ch)
    else:
        ch_neutral = cd_neutral
    # SurfaceLayerConfig has ONE Ch_neutral and its consumer applies it to both
    # the sensible and the latent flux, so a case whose LES uses separate C_H
    # and C_Q cannot be reproduced exactly. Say so rather than carrying
    # bulk_ce as metadata that merely LOOKS applied.
    if (not prescribed and case.spec.bulk_ce is not None
            and case.spec.bulk_ch is not None
            and abs(case.spec.bulk_ce - case.spec.bulk_ch) > 1.0e-12):
        pct = 100.0 * (case.spec.bulk_ce - case.spec.bulk_ch) / case.spec.bulk_ch
        print(f"  WARNING: {case.name} LES uses separate C_H="
              f"{case.spec.bulk_ch:.6f} and C_Q={case.spec.bulk_ce:.6f}, but "
              f"SurfaceLayerConfig has a single Ch_neutral applied to BOTH "
              f"fluxes. The latent flux therefore runs {pct:+.1f}% off the "
              "LES; moisture tuning will absorb part of that.")
    if bulk_scheme not in ("constant", "most"):
        raise ValueError(
            f"surface bulk_scheme={bulk_scheme!r} not supported here; "
            "choose 'constant' or 'most'."
        )
    if bulk_scheme == "most" and prescribed:
        raise ValueError(
            "bulk_scheme='most' cannot be combined with a prescribed surface "
            "heat flux: the MOST path derives the heat flux from its own "
            "scaling and ignores Ch_neutral, so the prescribed flux would be "
            "counted twice."
        )
    return SurfaceLayerConfig(
        z0=z0, z_ref=z_ref,
        Cd_neutral=cd_neutral, Ch_neutral=ch_neutral,
        bulk_scheme=bulk_scheme,
    )


# CLUBBParams entries with NO consumer anywhere in clubb.py, on either the
# diagnostic or the prognostic path. They are registry entries without an
# implementation, so they can never be tuned and their gradient is structurally
# zero: the six C_invrs_tau_* belong to the Guo (2021) invrs_tau reformulation
# that is not ported (compute_tau_family implements only the CAM-default
# simple form), and the rest have no call site at all.
CLUBB_UNIMPLEMENTED_PARAMS: tuple[str, ...] = (
    "C10", "c_K10h", "Lscale_mu_coef", "mult_coef",
    "coef_spread_DG_means_rt", "coef_spread_DG_means_thl",
    "slope_coef_spread_DG_means_w",
    "C_invrs_tau_bkgnd", "C_invrs_tau_sfc", "C_invrs_tau_shear",
    "C_invrs_tau_N2", "C_invrs_tau_N2_wp2", "C_invrs_tau_N2_xp2",
)


def build_physics_config(scheme: str, *, prescribed_fluxes: bool,
                         microphysics: str = "none",
                         bulk_ch: float | None = None,
                         bulk_ce: float | None = None,
                         surface=None,
                         clubb_prognostic: bool = True) -> PhysicsConfig:
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
    if scheme == "clubb":
        # The DIAGNOSTIC path (prognostic=False, the library default) reads only
        # six CLUBBParams fields and just four of them have a live gradient
        # (c_K, gamma_coef, lmin_coef, mu) -- beta cancels algebraically there
        # because Skw is passed as zeros, which pins mixt_frac to 0.5. The
        # prognostic path evolves the full moment set and takes the tunable
        # tier-1+2 count from 4 to 35 of 48. The SCM carries the moments in
        # PhysicsState.clubb_moments, so nothing else has to change.
        clubb_cfg = getattr(turb, "clubb", None) or CLUBBConfig()
        turb = turb._replace(
            clubb=clubb_cfg._replace(prognostic=bool(clubb_prognostic))
        )
    sub = getattr(turb, scheme)
    if surface is None:
        # Fallback for callers with no case in hand (tests): keep the scheme's
        # own default and only honour the prescribed-flux requirement.
        surface = sub.surface
        if prescribed_fluxes:
            surface = surface._replace(Ch_neutral=0.0)
        elif bulk_ch is not None:
            surface = surface._replace(Ch_neutral=float(bulk_ch))
    # The SAME SurfaceLayerConfig object on every arm: the surface boundary
    # must not be a per-scheme degree of freedom.
    turb = turb._replace(**{scheme: sub._replace(surface=surface)})
    base = PhysicsConfig()
    return PhysicsConfig(
        turbulence=turb,
        radiation=_scheme_none(base.radiation),
        convection=_scheme_none(base.convection),
        # Held identical across arms either way. "none" means the SCM has no
        # condensation, so the cloud layer carries supersaturated vapour where
        # the LES (Morrison) would condense -- a real SCM-vs-LES thermodynamic
        # difference in the buoyancy, not an artefact of the comparison.
        microphysics=base.microphysics._replace(scheme=microphysics),
        gravity_wave_drag=_scheme_none(base.gravity_wave_drag),
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
            # p_s.data is (1, 1, 1) (a 2-D field with a singleton level axis),
            # so index all three to get a SCALAR. [0, 0] leaves shape (1,) and
            # mismatches the scalar padding in masked_step's lax.cond.
            new_state.p_s.data[0, 0, 0],
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

    Uses the shared ``exner_function`` rather than an inline
    ``(p/p_ref)**kappa`` so the SCM, the global model and the CRM cannot drift
    apart on the Poisson exponent.
    """
    return jnp.asarray(T_profile) / exner_function(jnp.asarray(p_full))


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

def score_against_les(means, *, reference, p_full, scored):
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
    for name in scored:
        ref_profile = jnp.asarray(
            np.nan_to_num(reference.profiles[name], nan=0.0), dtype=jnp.float64
        )
        pred = jnp.where(mask, predicted[name], 0.0)
        components[name] = normalized_profile_rmse(
            ref_profile, pred, weights, profile_floor=PROFILE_FLOOR,
        )
    stacked = jnp.stack([components[k] for k in sorted(components)])
    combined = safe_sqrt(jnp.mean(stacked ** 2))
    # A non-finite rollout must score WORST, never best. safe_sqrt returns 0
    # for a NaN input (NaN > 0 is False), and 0 is the perfect score, so a
    # scheme that blows up would otherwise rank FIRST -- which is exactly what
    # happened: an mynn25 CBL arm that went non-finite scored 0.000 against
    # louis's 1.938. Map any non-finite component to a large finite penalty
    # (finite so the gradient stays usable and the line search can still
    # reject the step).
    # Check the (nlev,) predictions and the (n_scored,) component scores
    # SEPARATELY -- stacking them together is a shape error.
    pred_bad = jnp.any(jnp.stack([
        jnp.any(~jnp.isfinite(jnp.asarray(predicted[k]))) for k in scored]))
    comp_bad = jnp.any(~jnp.isfinite(stacked))
    any_bad = pred_bad | comp_bad
    combined = jnp.where(any_bad, NONFINITE_PENALTY, combined)
    # Every COMPONENT is penalised too. Otherwise a NaN rollout still reports
    # per-variable zeros -- "perfect" -- while only the aggregate is large.
    components = {k: jnp.where(any_bad, NONFINITE_PENALTY, v)
                  for k, v in components.items()}
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
    # Class name alone is NOT unique across the registry: the ocean's vertical
    # mixing also registers a TKEConfig ('ocean.vm.tke'), so matching on the
    # name collected the ocean's parameters and then tried to splice them into
    # the atmospheric config ("TKEConfig has no field(s) ['Prandtl_tke0',
    # 'lc_coeff', ...]"). Restrict to the atmospheric turbulence namespace.
    keys = {m.scheme_key for m in build_registry()
            if m.config_class == wanted
            and m.scheme_key.startswith(_ATM_TURB_NAMESPACE)}
    if not keys:
        raise RuntimeError(
            f"no __param_spec__ registered under {_ATM_TURB_NAMESPACE!r} for "
            f"{wanted} (scheme {scheme!r}); it cannot be tuned."
        )
    # Every collected field must exist on the config the overrides are spliced
    # into, or apply_param_overrides raises mid-rollout.
    fields = set(type(target)._fields)
    stray = {m.field for m in build_registry()
             if m.scheme_key in keys and m.field not in fields}
    if stray:
        raise RuntimeError(
            f"registry entries {sorted(stray)} for scheme {scheme!r} are not "
            f"fields of {wanted}; the scheme_key lookup matched the wrong "
            "config."
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
    # ok / tuned            -> rankable, score reflects a real optimisation
    # no_reducing_step      -> line search never accepted; score IS the default
    # no_active_gradient    -> every parameter disconnected from the loss
    # no_tunable_params     -> nothing spec'd to tune
    # failed / tune_failed  -> raised
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
    per_case_default: dict[str, float] | None = None
    per_case_tuned: dict[str, float] | None = None


def _arm_config(scheme: str, arm: "CaseArm", args) -> PhysicsConfig:
    return build_physics_config(
        scheme, prescribed_fluxes=arm.prescribed_fluxes,
        microphysics=args.microphysics,
        bulk_ch=arm.case.spec.bulk_ch, bulk_ce=arm.case.spec.bulk_ce,
        surface=arm.surface, clubb_prognostic=args.clubb_prognostic,
    )


def _arm_score(scheme: str, arm: "CaseArm", args, params=None,
               base_cfg=None):
    """Score one arm. ``params`` traced => differentiable."""
    cfg = base_cfg if base_cfg is not None else _arm_config(scheme, arm, args)
    means, ps_hist = _rollout_means(
        params, base_cfg=cfg, case=arm.case, dt=arm.dt, hours=arm.hours,
        analysis_hours=arm.analysis_hours, chunk_steps=arm.chunk_steps,
    )
    # theta = T/Exner uses the case's FIXED p_full, which is only valid while
    # p_s is static. Discarding ps_hist left this check permanently
    # unexecuted while the output still advertised it.
    #
    # ONLY on the untraced path: the assert concretizes with float(), so
    # calling it inside the differentiated loss would raise
    # TracerArrayConversionError. params is None exactly on the evaluation
    # path, which is where a drift would show up anyway -- the tuned
    # parameters cannot change p_s, only the physics can.
    drift = (None if params is not None
             else _assert_surface_pressure_static(ps_hist, arm.case.p_s))
    components, combined = score_against_les(
        means, reference=arm.reference, p_full=arm.case.p_full,
        scored=arm.scored,
    )
    return means, drift, components, combined


def joint_score(scheme: str, arms: list, args, params=None, cfgs=None):
    """Aggregate across cases.

    Each arm's score is already normalised by that case's OWN reference spread,
    so the arms are commensurable and a plain mean weights every regime
    equally. That is the point of the multi-case fit: a parameter set that wins
    on trade cumulus by wrecking the stable boundary layer must not score well.
    """
    per_case, per_components, drifts, means_out = {}, {}, {}, []
    total = None
    for i, arm in enumerate(arms):
        cfg = None if cfgs is None else cfgs[i]
        _m, _drift, comp, combined = _arm_score(
            scheme, arm, args, params=params, base_cfg=cfg)
        means_out.append(_m)
        per_case[arm.name] = combined
        per_components[arm.name] = comp
        if _drift is not None:
            drifts[arm.name] = float(_drift)
        total = combined if total is None else total + combined
    joint = total / float(len(arms))
    joint_score.last_ps_drift_pa = drifts
    joint_score.last_means = means_out
    return joint, per_case, per_components


def tune_scheme_multicase(scheme: str, *, arms, args, cfgs) -> SchemeResult:
    """Tune ONE parameter set per scheme against ALL cases at once."""
    result = SchemeResult(scheme=scheme, status="tuned")
    t0 = time.time()

    params_all = _initial_params(scheme, args.tier)
    if not params_all.constraints:
        result.status = "no_tunable_params"
        result.wall_s = time.time() - t0
        return result

    def loss_fn(params: TrainablePhysicsParams):
        joint, _pc, _comp = joint_score(scheme, arms, args, params=params,
                                        cfgs=cfgs)
        return joint

    preflight_loss, preflight_grads = eqx.filter_value_and_grad(loss_fn)(
        params_all)
    if not np.isfinite(float(preflight_loss)):
        result.status = "failed"
        result.error = f"preflight loss non-finite: {float(preflight_loss)}"
        result.wall_s = time.time() - t0
        return result
    stats = _grad_stats(preflight_grads, args.grad_nonzero_tol)

    def _reason(name, stat):
        if not stat["finite"]:
            return "non-finite gradient"
        field = name.rsplit(".", 1)[-1]
        if scheme == "clubb" and field in CLUBB_UNIMPLEMENTED_PARAMS:
            return ("UNIMPLEMENTED: no consumer anywhere in clubb.py on "
                    "either path")
        return f"|grad| <= {args.grad_nonzero_tol:g} in preflight"

    frozen = {n: _reason(n, st) for n, st in stats.items() if not st["nonzero"]}
    keep = {n for n, st in stats.items() if st["nonzero"]}
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
        optimizer=args.optimizer))
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    loss_history = [float(preflight_loss)]

    for step in range(1, args.steps + 1):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        loss_val = float(loss)
        bad = {n: st for n, st in _grad_stats(
            grads, args.grad_nonzero_tol).items()
            if not st["finite"] or not st["nonzero"]}
        if bad:
            result.error = f"gradient gate failed at step {step}: {bad}"
            result.status = "failed"
            break
        updates, opt_next = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(params, eqx.is_array))
        accepted = False
        for scale in _LINE_SEARCH_SCALES:
            cand = eqx.apply_updates(
                params, jax.tree_util.tree_map(lambda x: x * scale, updates))
            _assert_strict_bounds(cand)
            cand_loss = float(loss_fn(cand))
            if np.isfinite(cand_loss) and cand_loss < loss_val:
                params, opt_state = cand, opt_next
                loss_history.append(cand_loss)
                accepted = True
                break
        print(f"    [{scheme}] step {step}/{args.steps} joint={loss_val:.6g} "
              f"{'accepted' if accepted else 'no reducing step -> stop'}",
              flush=True)
        if not accepted:
            if step == 1:
                result.status = "no_reducing_step"
            break

    result.loss_history = loss_history
    tuned_cfgs = [_materialize_static(_apply_trainable_params(c, params))
                  for c in cfgs]
    joint, per_case, per_comp = joint_score(scheme, arms, args, params=None,
                                            cfgs=tuned_cfgs)
    result.score_tuned = float(joint)
    result.per_case_tuned = {k: float(v) for k, v in per_case.items()}
    result.components_tuned = {
        k: {kk: float(vv) for kk, vv in c.items()}
        for k, c in per_comp.items()}

    meta = {f"{m.scheme_key}.{m.field}": m for m in build_registry()}
    phys, allphys = params.as_dict(), params_all.as_dict()
    trained_names = {c.name for c in params.constraints}
    result.parameters = [{
        "name": c.name, "units": getattr(meta.get(c.name), "units", None),
        "default": float(np.asarray(allphys[c.name])),
        "tuned": (float(np.asarray(phys[c.name]))
                  if c.name in trained_names else None),
        "lower": c.min_val, "upper": c.max_val,
        "trained": c.name in trained_names,
        "frozen_reason": frozen.get(c.name),
    } for c in params_all.constraints]
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
    p.add_argument("--case", choices=list(ALL_CASES),
                   help="single case (shorthand for --cases NAME:DIR)")
    p.add_argument("--les-dir", type=Path,
                   help="LES output directory for --case")
    p.add_argument("--cases", default=None,
                   help="comma-separated NAME:LES_DIR pairs. ONE parameter set "
                        "per scheme is fitted to all of them jointly, so a "
                        "setting that wins on one regime by wrecking another "
                        "cannot score well. Choices: "
                        + ",".join(ALL_CASES))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--schemes", default="all",
                   help="comma-separated subset, or 'all'")
    p.add_argument("--nlev", type=int, default=DEFAULT_NLEV)
    p.add_argument("--dt", type=float, default=None,
                   help="physics timestep, ALL cases. Default: each case's own "
                        "spec value (60 s for the cumulus decks, 10 s for the "
                        "dry PBL cases). A single 60 s step blew mynn25 up on "
                        "CBL, whose convective eddy turnover is ~850 s.")
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
    p.add_argument("--score-variables", default="",
                   dest="score_variables",
                   help="comma-separated scored profiles. Default excludes u "
                        "and v: the SCM uses a constant-Cd surface drag while "
                        "the LES uses a z0 log-law wall model, so the momentum "
                        "profiles carry a surface-drag mismatch that tuning "
                        "would absorb into the turbulence parameters. The heat "
                        "and moisture fluxes ARE matched (both prescribed from "
                        "the deck), so theta and qv are the well-posed target.")
    p.add_argument("--clubb-prognostic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run full CLUBB on its prognostic path. The diagnostic "
                        "default leaves 44 of 48 tunable params with a zero "
                        "gradient; prognostic makes 35 of 48 live.")
    p.add_argument("--surface-bulk-scheme", default="constant",
                   choices=("constant", "most"),
                   help="surface flux law, IDENTICAL on every arm. 'constant' "
                        "uses a Cd derived from the neutral log law at the "
                        "case's own LES z0 (which reproduces the LES u*), and "
                        "is required whenever the case prescribes its surface "
                        "heat flux, since the MOST path would double-count it.")
    p.add_argument("--microphysics", default="none",
                   help="microphysics scheme, held identical across arms. "
                        "'none' means the SCM cannot condense, so its cloud "
                        "layer is supersaturated vapour where the LES "
                        "condenses.")
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--allow-radiation-mismatch", action="store_true",
                   help="run a case whose LES radiation the SCM cannot match; "
                        "the result is a confound and is labelled as one")
    return p.parse_args(argv)


def _build_arms(args, case_names: list[str], les_dirs: dict[str, Path]):
    """Resolve every case into a CaseArm, with its own window and surface.

    Per-case, because the regimes differ: each case has its own LES record
    length, its own analysis window, its own scored variables and its own
    roughness. Only the SCHEME is held constant across arms.
    """
    arms = []
    for name in case_names:
        case = load_case(name, nlev=args.nlev, dt=args.dt)
        dt = float(args.dt) if args.dt is not None else float(case.dt)
        scored = case_scored(name, args.scored_override)
        ref = load_les_reference(
            les_dirs[name], case=name, z_scm=case.z_full,
            p_half=_half_pressures(case),
            domain_top_m=case.les_domain_top_m,
            analysis_hours=args.analysis_hours, scored=scored,
        )
        # The SCM must land on the LES endpoint to within its OWN time
        # resolution. Demanding bit-exactness is wrong: a real LES ends at
        # whatever `while t < T` overshoots to (GABLS1's record is 9.0000278 h,
        # 0.1 s past 9 h, because its dt is 0.1 s), and no dt divides that.
        # Half a step is the finest the SCM can resolve, so a residual below
        # it is not a window mismatch -- it is rounding, and it is reported.
        les_end = float(ref.window_hours[1])
        if args.hours is not None:
            # Honour it, but only where it still lands on the reference: the
            # whole point of the guard below is that both sides average the
            # same window. Silently ignoring the flag was worse than either
            # obeying or refusing it.
            if abs(float(args.hours) - les_end) > 0.5 * dt / 3600.0:
                raise SystemExit(
                    f"{name}: --hours {args.hours} does not match the LES "
                    f"reference end {les_end:.6f} h, so the two sides would "
                    "average different windows. Omit --hours to follow the "
                    "reference.")
            les_end = float(args.hours)
        nsteps = max(1, int(round(les_end * 3600.0 / dt)))
        # Tolerance is a fraction of the WINDOW, not of the step. n =
        # round(T/dt) makes the residual <= dt/2 BY CONSTRUCTION, so a
        # "half a step" test can never fire -- it was vacuous. What matters is
        # whether the two sides average materially different intervals, which
        # is a fraction of the window.
        end_residual_s = abs(nsteps * dt - les_end * 3600.0)
        if end_residual_s > _WINDOW_TOL_FRAC * les_end * 3600.0:
            raise SystemExit(
                f"{name}: --dt {dt} s cannot land on the LES record end "
                f"{les_end:.6f} h: {nsteps} steps miss it by "
                f"{end_residual_s:.3f} s, over {_WINDOW_TOL_FRAC:.0%} of it, so the SCM "
                "would not end where the reference does."
            )
        span = les_end - float(ref.window_hours[0])
        n_an = max(1, int(round(span * 3600.0 / dt)))
        span_residual_s = abs(n_an * dt - span * 3600.0)
        if span_residual_s > _WINDOW_TOL_FRAC * span * 3600.0:
            raise SystemExit(
                f"{name}: --dt {dt} s cannot cover the retained analysis "
                f"window {span:.6f} h: {n_an} steps miss it by "
                f"{span_residual_s:.3f} s, over {_WINDOW_TOL_FRAC:.0%} of it, so the two "
                "sides would average different spans."
            )
        if max(end_residual_s, span_residual_s) > 1.0e-6:
            print(f"  NOTE {name}: SCM lands within "
                  f"{max(end_residual_s, span_residual_s):.3f} s of the LES "
                  f"endpoint/window (dt = {args.dt:g} s); the residual is "
                  "below one SCM step.")
        # Round the window START, not the span, so the endpoint and the start
        # are snapped on the SAME grid. Rounding both endpoint and span
        # independently lets each sit within half a step while their DIFFERENCE
        # moves the window start by nearly a full step.
        n_start = nsteps - n_an
        start_residual_s = abs(
            n_start * dt - float(ref.window_hours[0]) * 3600.0)
        if start_residual_s > _WINDOW_TOL_FRAC * span * 3600.0:
            raise SystemExit(
                f"{name}: with --dt {dt} s the analysis window would "
                f"start {start_residual_s:.3f} s from the LES window start, "
                "more than half a step.")
        les_end = nsteps * dt / 3600.0
        span = (nsteps - n_start) * dt / 3600.0
        surface = build_surface_config(
            case, bulk_scheme=args.surface_bulk_scheme)
        arms.append(CaseArm(
            name=name, case=case, reference=ref, scored=scored,
            hours=les_end, analysis_hours=span, dt=dt,
            chunk_steps=args.chunk_steps, surface=surface,
            prescribed_fluxes=(case.forcing.prescribe == "fluxes"),
            les_dir=str(les_dirs[name]),
        ))
    return arms


def main(argv=None) -> int:
    args = parse_args(argv)
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("JAX_ENABLE_X64=1 is required for this driver.")

    if bool(args.cases) == bool(args.case):
        raise SystemExit("give exactly one of --case/--les-dir or --cases")
    if args.cases:
        case_names, les_dirs = [], {}
        for tok in args.cases.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise SystemExit(f"--cases entry {tok!r} is not NAME:LES_DIR")
            n, d = tok.split(":", 1)
            n = n.strip()
            if n not in ALL_CASES:
                raise SystemExit(
                    f"unknown case {n!r}; choose from {list(ALL_CASES)}")
            if n in les_dirs:
                raise SystemExit(
                    f"--cases lists {n!r} more than once. Duplicates would "
                    "double-weight that case in the joint loss and overwrite "
                    "its per-case entry in the report, so only one directory "
                    "would actually be read.")
            case_names.append(n)
            les_dirs[n] = Path(d.strip())
    else:
        if args.les_dir is None:
            raise SystemExit("--case requires --les-dir")
        case_names, les_dirs = [args.case], {args.case: args.les_dir}
    for n in case_names:
        if n in _RADIATION_MISMATCH and not args.allow_radiation_mismatch:
            raise SystemExit(
                f"refusing to tune {n!r}: {_RADIATION_MISMATCH[n]}\n"
                "Pass --allow-radiation-mismatch to override; the output will "
                "be labelled a confound.")

    if args.score_variables:
        args.scored_override = tuple(
            v.strip() for v in args.score_variables.split(",") if v.strip())
        if not args.scored_override:
            raise SystemExit("--score-variables selected nothing")
    else:
        args.scored_override = None      # "" => per-case defaults
    args.scored = args.scored_override or ("theta", "qv")
    bad_scored = [v for v in args.scored if v not in SCORED_VARIABLES]
    if bad_scored:
        raise SystemExit(
            f"unknown scored variable(s) {bad_scored}; choose from "
            f"{list(SCORED_VARIABLES)}"
        )
    if not args.scored:
        raise SystemExit("--score-variables selected nothing")

    args.radiation_confound = {
        n: _RADIATION_MISMATCH[n] for n in case_names
        if n in _RADIATION_MISMATCH
    } or None

    schemes = (list(TURBULENCE_SCHEMES) if args.schemes == "all"
               else [s.strip() for s in args.schemes.split(",") if s.strip()])
    unknown = [s for s in schemes if s not in TURBULENCE_SCHEMES]
    if unknown:
        raise SystemExit(
            f"unknown scheme(s) {unknown}; choose from {list(TURBULENCE_SCHEMES)}"
        )

    outdir = args.outdir or (DEFAULT_OUTDIR / ("+".join(case_names)
                                                if len(case_names) > 1
                                                else case_names[0]))
    outdir.mkdir(parents=True, exist_ok=True)

    arms = _build_arms(args, case_names, les_dirs)
    print(f"cases: {', '.join(a.name for a in arms)}   "
          f"(one parameter set per scheme, fitted to ALL of them)")
    for a in arms:
        sc = a.surface
        print(f"  {a.name:8s} nlev={a.case.nlev} dt={a.dt:g}s hours={a.hours:g} "
              f"window={a.reference.window_label} "
              f"levels={int(a.reference.mask.sum())}/{a.case.nlev} "
              f"scored={list(a.scored)} prescribe={a.case.forcing.prescribe}")
        print(f"           z0={sc.z0:.2e} z_ref={sc.z_ref:.1f} m "
              f"Cd={sc.Cd_neutral:.4e} Ch={sc.Ch_neutral:.4e}")

    configs: dict[str, list] = {}
    profiles: dict[str, list] = {}
    results: list[SchemeResult] = []
    for scheme in schemes:
        print(f"\n[eval] {scheme}", flush=True)
        res = SchemeResult(scheme=scheme, status="ok")
        t0 = time.time()
        try:
            cfgs = [_arm_config(scheme, a, args) for a in arms]
            joint, per_case, per_comp = joint_score(
                scheme, arms, args, params=None, cfgs=cfgs)
            configs[scheme] = cfgs
            profiles[scheme] = [
                {"theta": np.asarray(_theta_from_T(mm["T"], a.case.p_full)),
                 "qv": np.asarray(mm["qv"]), "u": np.asarray(mm["u"]),
                 "v": np.asarray(mm["v"])}
                for a, mm in zip(arms, joint_score.last_means)
            ]
            res.score_default = float(joint)
            res.per_case_default = {k: float(v) for k, v in per_case.items()}
            res.ps_drift_pa = max(
                getattr(joint_score, "last_ps_drift_pa", {}).values(),
                default=None)
            res.components_default = {
                k: {kk: float(vv) for kk, vv in c.items()}
                for k, c in per_comp.items()}
            print(f"    joint={res.score_default:.6g}   " + "  ".join(
                f"{k}={v:.4g}" for k, v in res.per_case_default.items()))
        except Exception as exc:                      # noqa: BLE001
            res.status = "failed"
            res.error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED {res.error}", flush=True)
        res.wall_s = time.time() - t0
        results.append(res)
        jax.clear_caches()

    if configs:
        # The control is per CASE: within a case every scheme must differ only
        # in PhysicsConfig.turbulence.
        for i, a in enumerate(arms):
            _assert_arms_differ_only_in_turbulence(
                {s: c[i] for s, c in configs.items()})
        print("\n[control] within every case, arms verified identical "
              "outside PhysicsConfig.turbulence")

    if not args.skip_tuning:
        for res in results:
            if res.status != "ok":
                continue
            print(f"\n[tune] {res.scheme}", flush=True)
            try:
                tuned = tune_scheme_multicase(
                    res.scheme, arms=arms, args=args,
                    cfgs=configs[res.scheme])
            except Exception as exc:                  # noqa: BLE001
                res.status = "tune_failed"
                res.error = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED {res.error}", flush=True)
                continue
            for f in ("status", "score_tuned", "components_tuned", "n_trained",
                      "frozen", "parameters", "loss_history",
                      "per_case_tuned"):
                setattr(res, f, getattr(tuned, f))
            if tuned.error:
                res.error = tuned.error
            if res.score_tuned is not None:
                print(f"    joint {res.score_default:.6g} -> "
                      f"{res.score_tuned:.6g} ({res.n_trained} trained)   "
                      + "  ".join(f"{k}={v:.4g}"
                                  for k, v in (res.per_case_tuned or {}).items()))
            jax.clear_caches()

    _write_outputs(outdir, args, arms, results)
    for i, a in enumerate(arms):
        _write_case_profiles(outdir, a, {s_: p_[i]
                                         for s_, p_ in profiles.items()
                                         if p_[i] is not None})
    print(f"\nwrote {outdir}")

    # Per-arm exceptions are caught so one bad scheme cannot destroy the whole
    # campaign, but the EXIT CODE must still report them: a run where every arm
    # failed was previously indistinguishable from a clean sweep, and a wrapper
    # script checking $? would have called it success.
    failed = [r for r in results if r.status in ("failed", "tune_failed")]
    if failed:
        print(f"\n{len(failed)} of {len(results)} arm(s) FAILED: "
              + ", ".join(f"{r.scheme} ({r.status})" for r in failed))
        return 1
    return 0


def _half_pressures(case) -> np.ndarray:
    """SCM half-level pressures [Pa] from the case's sigma column."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(
        case.nlev, sigma_top=case.sigma_top, dtype=jnp.float64,
    )
    return np.asarray(sigma.sigma_half, dtype=np.float64) * case.p_s


def _write_case_profiles(outdir: Path, arm, per_scheme: dict) -> None:
    """One profiles npz PER CASE, so the plot keeps working and a successful
    run does not silently lose its profile-level audit trail."""
    payload = {
        "z_scm": np.asarray(arm.case.z_full),
        "p_full": np.asarray(arm.case.p_full),
        "mask": np.asarray(arm.reference.mask),
        "weights": np.asarray(arm.reference.weights),
        "z_les": np.asarray(arm.reference.z_les),
        "window_hours": np.asarray(arm.reference.window_hours),
        "scored": np.asarray(list(arm.scored)),
    }
    for name, prof in arm.reference.profiles.items():
        payload[f"les_scmlev_{name}"] = np.asarray(prof)
    for name, prof in arm.reference.profiles_les.items():
        payload[f"les_native_{name}"] = np.asarray(prof)
    for scheme, prof in per_scheme.items():
        for name, values in prof.items():
            payload[f"scm_{scheme}_{name}"] = np.asarray(values)
    out = outdir / (f"profiles_{arm.name}.npz" if outdir.name != arm.name
                    else "profiles.npz")
    np.savez(out, **payload)


def _write_outputs(outdir: Path, args, arms, results) -> None:
    # A failed or gradient-dead arm must NOT be ranked: one that fails after a
    # single favourable update would otherwise be reported as the winner.
    # An arm is RANKED only if its score reflects a genuine optimisation.
    # "no_active_gradient" was previously included, which let a closure whose
    # parameters are disconnected from the loss be ranked -- possibly FIRST --
    # on its untouched default score against genuinely tuned arms.
    _RANKABLE = {"ok", "tuned"}

    def _score_of(r):
        if r.score_tuned is not None:
            return r.score_tuned
        if r.score_default is not None:
            return r.score_default
        return float("inf")

    def _penalised(r) -> bool:
        # An arm that hit the non-finite penalty must not be ranked at all --
        # with --skip-tuning there is no gradient gate to catch it, and a
        # penalty of 1000 still "beats" any genuine score above 1000.
        vals = [v for v in (r.score_default, r.score_tuned) if v is not None]
        vals += list((r.per_case_default or {}).values())
        vals += list((r.per_case_tuned or {}).values())
        return any(v >= NONFINITE_PENALTY * 0.99 for v in vals)

    for r in results:
        if r.status in _RANKABLE and _penalised(r):
            r.status = "nonfinite_rollout"
            r.error = (r.error or "") + (
                " non-finite rollout on at least one case; excluded from the "
                "ranking")
    rankable = [r for r in results
                if r.status in _RANKABLE and r.score_default is not None]
    excluded = [r for r in results if r not in rankable]
    ranked = sorted(rankable, key=_score_of) + sorted(
        excluded, key=lambda r: r.scheme)
    with (outdir / "ranking.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "scheme", "status", "score_default",
                    "score_tuned", "n_trained", "n_frozen", "wall_s", "error"])
        for i, r in enumerate(ranked):
            rankable = r.status in _RANKABLE and r.score_default is not None
            w.writerow([(i + 1) if rankable else "EXCLUDED",
                        r.scheme, r.status, r.score_default, r.score_tuned,
                        r.n_trained, len(r.frozen or {}),
                        f"{r.wall_s:.1f}", r.error or ""])

    payload = {
        "case": args.case,
        # Present and non-null ONLY when --allow-radiation-mismatch was used.
        # Without this the JSON claimed a controlled comparison while ranking a
        # case whose LES radiation the SCM cannot reproduce.
        "RADIATION_CONFOUND": args.radiation_confound,
        "protocol": {
            "nlev": args.nlev,
            "dt_s": {a.name: a.dt for a in arms},
            "cases": [a.name for a in arms],
            "tier": args.tier, "optimizer": args.optimizer, "lr": args.lr,
            "steps": args.steps,
                        "radiation": "none", "convection": "none",
            "microphysics": args.microphysics,
            "scored_variables_per_case": {a.name: list(a.scored) for a in arms},
            "note": (
                "All arms share one column and forcing built from the same "
                "gSAM deck as the LES; only PhysicsConfig.turbulence differs, "
                "verified mechanically. This makes the ACROSS-SCHEME "
                "comparison controlled. It does NOT make the SCM a replica of "
                "the LES: see known_scm_les_differences."
            ),
            # Stated, not buried. Each of these is identical across arms, so
            # the ranking stays controlled, but each degrades the absolute
            # LES-match and could be absorbed into a tuned parameter.
            "single_surface_exchange_coefficient": {
                a.name: {
                    "les_C_H": a.case.spec.bulk_ch,
                    "les_C_Q": a.case.spec.bulk_ce,
                    "scm_Ch_neutral_applied_to_both": a.case.spec.bulk_ch,
                    "latent_flux_error_pct": 100.0
                    * (a.case.spec.bulk_ce - a.case.spec.bulk_ch)
                    / a.case.spec.bulk_ch,
                }
                for a in arms
                if a.case.spec.bulk_ce is not None
                and a.case.spec.bulk_ch is not None
                and not a.prescribed_fluxes
            } or None,
            "known_scm_les_differences": [
                "Surface momentum: the SCM uses a constant-Cd bulk drag while "
                "the LES uses a z0 log-law wall model. This is why u and v are "
                "not scored by default.",
                "Microphysics: the LES runs Morrison; the SCM runs "
                f"{args.microphysics!r}. With 'none' the SCM cannot condense, "
                "so its cloud layer holds supersaturated vapour where the LES "
                "forms liquid, and for RICO the LES precipitates.",
                "Resolved vs parameterized: the LES resolves the large eddies "
                "the SCM closure must represent — that difference IS the "
                "quantity being tuned, not an error.",
            ],
        },
        "cases": [
            {
                "case": a.name,
                "les_dir": a.les_dir,
                "window_hours": list(a.reference.window_hours),
                "n_frames": a.reference.n_frames,
                "levels_in_domain": int(a.reference.mask.sum()),
                "nlev": a.case.nlev,
                "scored": list(a.scored),
                "surface_prescribe": a.case.forcing.prescribe,
                "Cd_neutral": float(a.surface.Cd_neutral),
                "Ch_neutral": float(a.surface.Ch_neutral),
                "z0_m": float(a.surface.z0),
                "z_ref_m": float(a.surface.z_ref),
            }
            for a in arms
        ],
        "schemes": [
            {
                "scheme": r.scheme, "status": r.status,
                "score_default": r.score_default, "score_tuned": r.score_tuned,
                "components_default": r.components_default,
                "components_tuned": r.components_tuned,
                "per_case_default": r.per_case_default,
                "per_case_tuned": r.per_case_tuned,
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
    ]
    if args.radiation_confound:
        lines += ["> **THIS IS NOT A TURBULENCE RANKING.** "
                  "`--allow-radiation-mismatch` was used:", ""]
        lines += [f"> - **{k}**: {v}" for k, v in
                  args.radiation_confound.items()] + [""]
    lines += [
        "\n".join(
            f"- **{a.name}**: `{a.les_dir}`, window {a.reference.window_label}, "
            f"{int(a.reference.mask.sum())} of {a.case.nlev} SCM levels inside "
            f"the LES domain, scored on {', '.join(a.scored)}"
            for a in arms),
        "",
        f"SCM: per-case dt, one parameter set per scheme fitted to ALL "
        f"{len(arms)} case(s) jointly; the joint score is the mean of the "
        "per-case normalized scores.", "",
        "Within every case, all arms share one initial column, one forcing and "
        "one surface boundary condition; only `PhysicsConfig.turbulence` "
        "differs, checked mechanically. Lower is better.", "",
        "| rank | scheme | status | joint (default) | joint (tuned) | "
        "trained | frozen |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked):
        d = "—" if r.score_default is None else f"{r.score_default:.4f}"
        t = "—" if r.score_tuned is None else f"{r.score_tuned:.4f}"
        rankable = r.status in _RANKABLE and r.score_default is not None
        pos = str(i + 1) if rankable else "excl."
        lines.append(f"| {pos} | {r.scheme} | {r.status} | {d} | {t} | "
                     f"{r.n_trained} | {len(r.frozen or {})} |")
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
        "- RADIATION matches the LES for BOMEX/RICO: the deck's `lsf` "
        "temperature tendency IS the case's radiative cooling (-2.0 K/day "
        "below 1500 m, tapering to 0 by 2500 m for BOMEX) and both models read "
        "it from the same file. That is why DYCOMS, whose LES adds a Stevens "
        "longwave parameterization, is refused.",
        "- The SCM is NOT otherwise a replica of the LES. It uses a "
        "constant-Cd surface drag against the LES's z0 log-law wall model "
        "(which is why u and v are excluded from the score by default), and "
        "the LES runs Morrison microphysics while the SCM's is set by "
        "`--microphysics`. Each difference is identical across arms, so the "
        "RANKING is controlled, but each also degrades the absolute LES-match "
        "and can be absorbed into a tuned parameter. See "
        "`known_scm_les_differences` in tuned_parameters.json.",
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
