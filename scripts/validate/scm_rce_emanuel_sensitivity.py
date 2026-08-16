#!/usr/bin/env python
"""Do Emanuel's tunable parameters reach the column at all?

MEASURED on the 2026-08-16 arm: the tuner spent 240 evaluations over emanuel's
25 parameters and NOT ONE beat the defaults, while every other scheme improved
substantially (dca 6.06 -> 2.13 on a single parameter).  Two explanations fit:
the defaults are already optimal, or the parameters do not change what the
column does.  They are distinguished by MEASUREMENT, not by argument.

For each parameter this runs the campaign's own SCM twice — at the 25 % and
75 % points of the parameter's declared range, everything else at defaults —
and reports how far the thermodynamic score, the temperature profile and the
precipitation move.  A parameter that moves NOTHING is unreachable in this
configuration, which is a defect in the scheme's wiring, not a tuning outcome.

The run length is deliberately short: a parameter that is genuinely live
changes the tendencies immediately, and a dead one produces a bit-identical
column however long it is integrated.  A short run cannot prove a parameter is
unimportant at equilibrium — only that it is, or is not, connected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp  # noqa: E402

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402
from legoesm.training.param_collector import build_trainable_params  # noqa: E402
from legoesm.training.trainable_params import (  # noqa: E402
    TrainablePhysicsParams,
)


def _run(cfg, ref, cache, label, *, days, dt):
    return camp.run_cached(
        cache, cfg, ref, label=label, days=days, dt=dt,
        analysis_days=min(1.0, days), require_equilibrium=False,
        require_realism=False, equil_T_tol_K=1.0, equil_qv_tol=1.0,
        equil_qcond_tol=1.0, scm_microphysics_substeps=30,
        scm_convection_substeps=10,
        surface_wind_m_s=camp.DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
        coriolis_s_inv=camp.DEFAULT_SCM_RCE_CORIOLIS_S_INV,
        large_scale_forcing="none", bl_anchor_top_m=-1.0,
        subcloud_top_m=camp.DEFAULT_SUBCLOUD_TOP_M,
        thermo_humidity="logq")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", default="emanuel")
    parser.add_argument("--param-set", default="physical")
    # NOT camp.DEFAULT_REFERENCE_DIR: that points at results/rcemip1_n128_ocean,
    # OUR OWN CRM run, which is the artifact under investigation rather than an
    # oracle (see the campaign's self-reference note).  Default to the external
    # RCEMIP archive run and refuse anything that looks like our own output.
    # ABSOLUTE: a relative default resolves against the WORKTREE, and this
    # script is run from pinned worktrees where results/ does not exist.
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("/burg-archive/glab/users/pg2328/legoESM/results"
                     "/rcemip_ref_sam300"))
    parser.add_argument("--last-reference-files", type=int, default=5)
    parser.add_argument("--days", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if "rcemip1_n128" in str(args.reference_dir) or "rcemip1_postfix" in str(
            args.reference_dir):
        raise SystemExit(
            f"REFUSED: {args.reference_dir} is our own CRM run, not an oracle.")
    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files)
    base_cfg = camp.make_physics_config(
        radiation="rrtmgp", convection=args.scheme, microphysics="morrison",
        hard_saturation_adjustment=True)
    base_cfg, _st = camp.apply_subsidence_solve_override(
        base_cfg, "implicit_flux", category="convection")
    _c, _s, subcfg = camp._active_subconfig(base_cfg, "convection")
    key = camp._scheme_key_for_subconfig(subcfg)
    tier, include_tier0, exclude = camp.resolve_param_selection(
        key, args.param_set)
    params = build_trainable_params(
        active_scheme_keys={key}, tier=tier, include_tier0=include_tier0,
        exclude=exclude, dtype=jnp.float64)
    constraints = params.constraints
    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}

    cache: dict = {}
    base = _run(base_cfg, ref, cache, f"base:{args.scheme}",
                days=args.days, dt=args.dt)
    print(f"{args.scheme}: {len(constraints)} parameters, "
          f"{args.days:g} d, base thermo={base.thermo_score:.6f} "
          f"T_term={base.thermo_T_term:.6f} P={base.precip_mm_day:.4f}")
    base_T = np.asarray(base.T_profile)

    results = []
    for c in constraints:
        row = {"parameter": c.field, "qualified": c.name,
               "default": defaults[c.name],
               "bounds": [float(c.min_val), float(c.max_val)]}
        for tag, frac in (("lo", 0.25), ("hi", 0.75)):
            values = dict(defaults)
            values[c.name] = camp._interp(c, frac)
            trial = TrainablePhysicsParams(
                raw_values={k: camp._raw_from_physical(v, cc)
                            for (k, v), cc in zip(values.items(), constraints)},
                constraints=constraints)
            field_values = trial.to_overrides().get(key, {})
            tuned = camp.apply_param_overrides(
                camp._tunable_subconfig(subcfg), field_values)
            cfg = camp._set_active_subconfig(
                base_cfg, "convection",
                camp._rewrap_tunable_subconfig(subcfg, tuned))
            run = _run(cfg, ref, cache, f"sens:{c.field}:{tag}",
                       days=args.days, dt=args.dt)
            dT = (float(np.max(np.abs(np.asarray(run.T_profile) - base_T)))
                  if run.T_profile else float("nan"))
            row[tag] = {
                "value": values[c.name],
                "d_thermo": run.thermo_score - base.thermo_score,
                "d_T_max_K": dT,
                "d_precip": run.precip_mm_day - base.precip_mm_day,
                "status": run.status,
            }
        moved = max(abs(row["lo"]["d_T_max_K"]), abs(row["hi"]["d_T_max_K"]))
        row["max_dT_K"] = moved
        row["dead"] = bool(moved == 0.0)
        results.append(row)
        flag = "DEAD" if row["dead"] else "live"
        print(f"  {c.field:<34} {flag:4s} max|dT|={moved:.3e} K  "
              f"d_thermo lo={row['lo']['d_thermo']:+.4f} "
              f"hi={row['hi']['d_thermo']:+.4f}")

    dead = [r["parameter"] for r in results if r["dead"]]
    print(f"\n{len(dead)} of {len(results)} parameters are DEAD in this "
          f"configuration (perturbing them changes the column by exactly 0)")
    if dead:
        print("  " + ", ".join(dead))
    payload = {"scheme": args.scheme, "days": args.days,
               "base_thermo": base.thermo_score, "n_dead": len(dead),
               "dead": dead, "parameters": results}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
