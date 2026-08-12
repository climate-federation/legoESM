#!/usr/bin/env python
"""Can a convection scheme's tunable parameters reach its tendency at all?

WHY.  In the 2026-08 SCM-RCE campaign, `tiedtke` (16 parameters, 192
evaluations) and `kuo` (2 parameters, 48) returned a tuned score BIT-IDENTICAL
to their defaults, while other schemes moved 40-85 %.  Bit-identity across
that many distinct configurations is not weak sensitivity; it is a signature.

WHAT WAS ALREADY RULED OUT, so this probe does not re-test it:

* the parameters exist as fields on the scheme's config NamedTuple;
* the tuner's override chain writes them (fields change, and the run-cache key
  differs, so distinct configs really are evaluated);
* both schemes read the parameters in their bodies.

So the config reaches the scheme and the scheme reads it. What is left is
whether the read VALUE can change the OUTPUT in this regime -- a parameter
multiplied by zero, or inside a clip that saturates, is live code and dead
physics.

WHAT THIS PROBE CANNOT CONCLUDE (codex review, 2026-08-12 — read before
quoting any DEAD verdict):

* It calls the physics with NO horizontal grid, so the bridge sets
  ``moisture_convergence=None``.  Kuo treats that as a zero source and returns
  zero tendencies, so EVERY Kuo parameter reads DEAD here without being
  tested.  Kuo's real status was established from the code
  (``convection/integration.py:435``: Kuo is deliberately OFF on a
  single-column state) and CONFIRMED by scoring a kuo column against a
  ``convection=none`` column — bit-identical, ``max|dT| = 0``.
* It passes ``phys_state=None``, i.e. a zero prognostic mass-flux profile, so
  what it measures is a COLD-START tendency.  A parameter that only bites once
  the carry has spun up — a cap on a mass flux that is still zero, for
  instance — reads DEAD here and is not.
* It reads only ``dT/dt`` and ``dq_v/dt``.  A parameter that moves only
  condensate, precipitation, momentum or the returned carry reads DEAD.

So a DEAD verdict from this probe means "no effect on the thermal or vapour
tendency, at cold start, on these five synthetic columns" — not "cannot affect
the campaign".

WHAT THIS MEASURES.  For each tunable parameter, on each of several column
states, the scheme's tendency is recomputed with that ONE parameter moved and
compared to the default:

    max|d(dT/dt)|,  max|d(dq_v/dt)|,  and whether the change is EXACTLY zero.

A parameter whose tendency change is exactly zero on EVERY state is dead in
those states.  One that is nonzero somewhere is live, and the campaign's
insensitivity then has to be explained at the equilibrium level instead.

TWO DESIGN POINTS, both from the GLM-5.2 review of an earlier draft:

* A SINGLE column cannot support a "dead" verdict.  Convection schemes are
  branch forests behind trigger predicates, and a parameter governing a branch
  the column never enters is untested, not dead.  Several states are used, and
  the verdict is over their union.
* Sweeping a parameter to 25/75 % of its BOUNDS reproduces the tuner's own
  sweep -- the thing that failed.  Bounds are frequently arbitrary relative to
  the value a threshold is compared against, so a threshold can be live while
  both sweep points sit on the same side of it.  Each parameter is therefore
  moved over several fractions of its range AND to a multiple of its own
  default, and the largest response is reported.

Run (CPU, seconds)::

    JAX_ENABLE_X64=1 python scripts/validate/check_convection_param_reachability.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics import (
    ConvectionConfig, GravityWaveDragConfig, MicrophysicsConfig, PhysicsConfig,
    RadiationConfig, TurbulenceConfig, make_physics,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training.param_collector import build_trainable_params
from legoesm.training.trainable_params import TrainablePhysicsParams

from scripts.run import run_scm_rce_campaign as camp

# Fractions of each parameter's declared range, plus multiples of its own
# default.  The tuner used {0.25, 0.75} of the range only; a threshold whose
# realistic value sits outside that window is untestable by it.
RANGE_FRACTIONS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
DEFAULT_MULTIPLES = (0.1, 0.5, 2.0, 10.0)

P_SFC = 101_480.0


def _column(nlev: int, *, T_sfc: float, rh: float, lapse_K_km: float,
            name: str):
    """A single tropical column: linear lapse to a 200 K cap, uniform RH."""
    from legoesm.thermo import saturation_mixing_ratio

    # Built by the SHARED factory, not by hand: SigmaCoordinate is a
    # NamedTuple with seven derived fields (ln_ratio, alpha, dsigma_full, ...)
    # that the physics reads, and constructing it positionally is how this
    # probe first crashed. Same rule as everywhere else here -- read the
    # constructor, do not infer it.
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full, dtype=float)
    p_full = sigma_full * P_SFC
    # height from a dry hydrostatic estimate, only to build a lapse profile
    z = -(constants.R_d * 290.0 / constants.g) * np.log(
        np.maximum(sigma_full, 1e-6))
    T = np.maximum(T_sfc - lapse_K_km * 1e-3 * z, 200.0)
    q_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T),
                                               jnp.asarray(p_full)))
    q_v = np.maximum(rh * q_sat, 1e-11)
    dims4 = ("face", "x", "y", "level")
    dims3 = ("face", "x", "y")
    shape4 = (1, 1, 1, nlev)
    state = HydrostaticState(
        u=Field(data=jnp.full(shape4, 5.0), name="u", dims=dims4),
        v=Field(data=jnp.zeros(shape4), name="v", dims=dims4),
        T=Field(data=jnp.asarray(T).reshape(shape4), name="T", dims=dims4),
        p_s=Field(data=jnp.full((1, 1, 1), P_SFC), name="p_s", dims=dims3),
        phis=Field(data=jnp.zeros((1, 1, 1)), name="phis", dims=dims3),
        tracers={"q_v": Field(data=jnp.asarray(q_v).reshape(shape4),
                              name="q_v", dims=dims4),
                 "q_c": Field(data=jnp.zeros(shape4), name="q_c", dims=dims4),
                 "q_r": Field(data=jnp.zeros(shape4), name="q_r", dims=dims4)},
    )
    return name, state, sigma


def build_states(nlev: int):
    """States spanning the branches a convection scheme can take.

    A parameter dead in ALL of these is dead for the campaign's purposes; a
    parameter dead in one is merely untested there.
    """
    return [
        _column(nlev, T_sfc=300.0, rh=0.95, lapse_K_km=6.5,
                name="rce_saturated"),      # the campaign's own equilibrium
        _column(nlev, T_sfc=300.0, rh=0.80, lapse_K_km=6.5,
                name="rce_subsaturated"),
        _column(nlev, T_sfc=302.0, rh=0.85, lapse_K_km=8.0,
                name="deep_unstable"),      # steep lapse -> large CAPE
        _column(nlev, T_sfc=298.0, rh=0.60, lapse_K_km=5.0,
                name="suppressed_stable"),  # trigger should NOT fire
        _column(nlev, T_sfc=300.0, rh=0.70, lapse_K_km=7.0,
                name="trade_shallow"),
    ]


def convection_only_physics(scheme: str, dt: float):
    return make_physics(
        PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=ConvectionConfig(scheme=scheme),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        ),
        model_type="hydrostatic", dt=dt,
    )


def _tendency(scheme: str, subcfg, state, sigma, dt: float):
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme=scheme, **{scheme: subcfg}),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    fn = make_physics(cfg, model_type="hydrostatic", dt=dt)
    tend, _ = fn(state, None, sigma)
    dT = np.asarray(tend.dT_dt.data, dtype=float)
    dq = (np.asarray(tend.tracer_tendencies["q_v"].data, dtype=float)
          if tend.tracer_tendencies and "q_v" in tend.tracer_tendencies
          else np.zeros_like(dT))
    return dT, dq


def probe_scheme(scheme: str, *, nlev: int, dt: float) -> list[dict]:
    base_cfg = camp.make_physics_config(convection=scheme)
    _c, _s, sub = camp._active_subconfig(base_cfg, "convection")
    key = camp._scheme_key_for_subconfig(sub)
    params = build_trainable_params(active_scheme_keys={key},
                                    tier="extended", dtype=jnp.float64)
    constraints = params.constraints
    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}
    states = build_states(nlev)

    base = {}
    for name, state, sigma in states:
        base[name] = _tendency(scheme, sub, state, sigma, dt)
        active = float(np.max(np.abs(base[name][0])))
        print(f"  [{scheme}] state {name:18s} baseline max|dT/dt| = "
              f"{active:.4e} K/s  ({'ACTIVE' if active > 0 else 'DORMANT'})")

    rows = []
    for c in constraints:
        lo, hi, d0 = float(c.min_val), float(c.max_val), defaults[c.name]
        trials = [lo + f * (hi - lo) for f in RANGE_FRACTIONS]
        trials += [min(max(d0 * k, lo), hi) for k in DEFAULT_MULTIPLES]
        worst_dT = worst_dq = 0.0
        worst_state = "-"
        for name, state, sigma in states:
            b_dT, b_dq = base[name]
            for v in trials:
                if v == d0:
                    continue
                trial_sub = camp._rewrap_tunable_subconfig(
                    sub, camp.apply_param_overrides(
                        camp._tunable_subconfig(sub), {c.field: v}))
                t_dT, t_dq = _tendency(scheme, trial_sub, state, sigma, dt)
                ddT = float(np.max(np.abs(t_dT - b_dT)))
                ddq = float(np.max(np.abs(t_dq - b_dq)))
                # NaN must never read as DEAD: max(0.0, nan) keeps 0.0, so a
                # scheme that blew up on a trial value would be reported as
                # "no effect" (codex review, 2026-08-12).
                if not (np.isfinite(ddT) and np.isfinite(ddq)):
                    raise FloatingPointError(
                        f"{c.field}={v!r} on state {name} produced a non-finite "
                        "tendency difference; a DEAD verdict here would be an "
                        "artifact of NaN handling")
                if ddT > worst_dT or ddq > worst_dq:
                    worst_state = name
                worst_dT = max(worst_dT, ddT)
                worst_dq = max(worst_dq, ddq)
        rows.append({"param": c.field, "default": d0, "lo": lo, "hi": hi,
                     "max_dT": worst_dT, "max_dq": worst_dq,
                     "state": worst_state,
                     "dead": worst_dT == 0.0 and worst_dq == 0.0})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemes", default="kuo,tiedtke,mass_flux")
    ap.add_argument("--nlev", type=int, default=40)
    ap.add_argument("--dt", type=float, default=600.0)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("Convection tunable-parameter reachability: config -> TENDENCY")
    print("=" * 78)
    any_dead = False
    for scheme in args.schemes.split(","):
        scheme = scheme.strip()
        if not scheme:
            continue
        print(f"\n### {scheme}")
        rows = probe_scheme(scheme, nlev=args.nlev, dt=args.dt)
        print(f"  {'parameter':34s}{'default':>12s}{'max|ddT/dt|':>14s}"
              f"{'max|ddq/dt|':>14s}  {'worst state':16s} verdict")
        for r in sorted(rows, key=lambda r: -r["max_dT"]):
            verdict = "DEAD (no effect anywhere)" if r["dead"] else "live"
            any_dead |= r["dead"]
            print(f"  {r['param']:34s}{r['default']:12.4g}{r['max_dT']:14.4e}"
                  f"{r['max_dq']:14.4e}  {r['state']:16s} {verdict}")
        n_dead = sum(r["dead"] for r in rows)
        print(f"  -> {n_dead}/{len(rows)} parameters produce NO tendency change "
              f"on any tested state")
    print("\n" + "-" * 78)
    print("A DEAD parameter cannot move any objective built on this scheme's")
    print("tendency, so tuning it is wasted budget. A LIVE parameter that")
    print("nevertheless left the campaign score bit-identical is a different")
    print("question -- one about the equilibrium, not the tendency -- and this")
    print("probe deliberately does not answer it.")
    return 1 if any_dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
