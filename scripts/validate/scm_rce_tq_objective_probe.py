#!/usr/bin/env python
"""Measure the two numbers that decide the SCM-RCE T/q tuning objective.

Both are pre-registered in ``docs/atmosphere/scm_rce_tq_tuning_strategy.md``
and neither is safe to assume:

1. **Where does a humidity metric actually look?**  A mass-weighted ABSOLUTE
   ``q_v`` RMSE is suspected of being a boundary-layer metric, because ``q_v``
   spans ~3 decades vertically and the mass weights add another low-level
   factor.  The probe answers it the only way that is not an argument: impose
   the SAME RELATIVE error at every level and report what fraction of the
   metric's sum-of-squares comes from above a stated height, for each candidate
   variable (absolute ``q_v``, ``log q_v``, relative humidity).  A uniform
   relative perturbation is the right control here because it is the null
   hypothesis "the scheme is equally wrong everywhere in a fractional sense";
   under a metric that is not height-biased it would spread its contribution in
   proportion to the mass weights alone.

2. **How many parameters does each convection scheme expose at each tier?**
   The tuning budget is derived from this count, and the campaign doc has been
   wrong about it before (it claimed ``dca`` had 1 extended-tier parameter; the
   live registry says 0).  Read it from the registry, never from prose.

The probe computes NOTHING of its own: reference profiles, the column pressure
and the relative humidity all come from the campaign's own helpers (so the
probe measures the metric the campaign minimises, not a lookalike), and
parameter metadata comes from ``legoesm.training.param_collector``.

Shares are reported twice: over the FULL column, and over the TROPOSPHERIC MASK
the objective actually scores — the two differ most for relative humidity,
which the stratosphere drags toward zero.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp  # noqa: E402

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402
from legoesm.training.param_collector import build_registry  # noqa: E402
from legoesm.training.scm_rce_metrics import (  # noqa: E402
    DEFAULT_THERMO_LOGQ_FLOOR,
    relative_humidity_profile,
)

#: Schemes in the convection intercomparison, in the array order of
#: ``convtune_arms.sbatch`` so the two can be read side by side.
SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)

#: Heights at which the "how much of the metric lives above here" question is
#: asked [km].  5 km is the level above which a tropical column's humidity is
#: set by convective detrainment rather than by surface fluxes, which is the
#: regime the schemes are supposed to differ in.
SPLIT_HEIGHTS_KM = (2.0, 5.0, 10.0)

#: Floor for log(q_v) [kg/kg].  Taken from the objective itself so the probe
#: cannot measure a metric the campaign does not use — a probe that answers a
#: slightly different question is how a justification number goes wrong.
LOG_QV_FLOOR = DEFAULT_THERMO_LOGQ_FLOOR


def _relative_perturbation_shares(
    values: np.ndarray,
    weights: np.ndarray,
    z_km: np.ndarray,
    *,
    transform: str,
    relative_error: float,
) -> dict:
    """Per-level share of a mass-weighted squared metric under a uniform
    relative error, plus the cumulative share above each split height.

    ``transform`` selects what the metric is computed ON:

    * ``"identity"`` — the metric is ``sum_k w_k (dq_k / sigma)^2`` with
      ``dq_k = eps * q_k``, so the share is ``w_k q_k^2``.  ``sigma`` is a
      single constant and cancels out of the SHARE (not out of the value), so
      the answer is independent of the normalisation choice.
    * ``"log"`` — the ACTUAL floored transform the objective uses,
      ``ln(max((1+eps) q, floor)) - ln(max(q, floor))``.  Away from the floor
      this is ``log(1+eps)`` at every level, so the share reduces to ``w_k``
      and the metric looks exactly where the mass is; at a level the floor
      binds it does NOT, which is the case worth being able to see.
    * ``"precomputed"`` — ``values`` already holds the per-level perturbation
      magnitude (used for relative humidity, where a relative ``q_v`` error
      maps to a relative RH error of the same size only because ``q_sat`` is
      unchanged by it — RH = q_v/q_sat is linear in ``q_v``).
    """
    if transform == "identity":
        magnitude = relative_error * values
    elif transform == "log":
        magnitude = (
            np.log(np.maximum((1.0 + relative_error) * values, LOG_QV_FLOOR))
            - np.log(np.maximum(values, LOG_QV_FLOOR))
        )
    elif transform == "precomputed":
        magnitude = values
    else:
        raise ValueError(
            f"_relative_perturbation_shares: unknown transform {transform!r}; "
            "expected 'identity', 'log' or 'precomputed'")
    contrib = weights * magnitude ** 2
    total = float(np.sum(contrib))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"_relative_perturbation_shares: degenerate total {total!r} for "
            f"transform {transform!r} — the share is undefined.")
    share = contrib / total
    above = {
        f"share_above_{h:g}km": float(np.sum(share[z_km >= h]))
        for h in SPLIT_HEIGHTS_KM
    }
    # The mass share above the same heights is the reference point: a metric
    # whose contribution share EQUALS the mass share is not height-biased.
    mass_above = {
        f"mass_share_above_{h:g}km": float(np.sum(weights[z_km >= h]))
        for h in SPLIT_HEIGHTS_KM
    }
    return {**above, **mass_above}


def humidity_metric_weighting(ref, *, relative_error: float) -> dict:
    """Height distribution of each candidate humidity (and T) metric."""
    z_km = np.asarray(ref.z_m, dtype=float) / 1000.0
    w = np.asarray(ref.mass_weights, dtype=float)
    qv = np.asarray(ref.qv_ref, dtype=float)
    T = np.asarray(ref.T_ref, dtype=float)

    # The SAME pressure profile and the SAME (liquid/ice blended) saturation
    # curve the objective uses.  The first version of this probe used
    # liquid-only Tetens on a pressure computed straight from height, so its RH
    # number described a metric the campaign does not minimise.
    p_full = camp.reference_pressure_profile(ref)
    rh = np.asarray(relative_humidity_profile(
        jnp.asarray(T, dtype=jnp.float64),
        jnp.asarray(qv, dtype=jnp.float64),
        jnp.asarray(p_full, dtype=jnp.float64),
    ), dtype=float)
    # RH = q_v / q_sat is LINEAR in q_v at fixed T, so a uniform relative q_v
    # error is a uniform relative RH error; the per-level magnitude is
    # eps * RH_k, which is NOT uniform because RH itself varies.
    rh_magnitude = relative_error * rh

    out = {
        "relative_error": relative_error,
        "n_levels": int(z_km.size),
        "z_km_range": [float(z_km.min()), float(z_km.max())],
        # Index 0 is the TOP of the column and -1 the surface: the reference
        # grid descends in height (build_reference_profiles sets
        # z_half[-1] = 0.0), which is also why the campaign reads the lowest
        # level as ``T[-1]``.  Getting this backwards only mislabels a printed
        # scalar, but a mislabelled scalar is what gets quoted.
        "qv_g_kg_surface": float(qv[-1] * 1000.0),
        "qv_g_kg_top": float(qv[0] * 1000.0),
        "rh_min": float(np.min(rh)),
        "rh_max": float(np.max(rh)),
        "absolute_qv": _relative_perturbation_shares(
            qv, w, z_km, transform="identity", relative_error=relative_error),
        "log_qv": _relative_perturbation_shares(
            qv, w, z_km,
            transform="log", relative_error=relative_error),
        "relative_humidity": _relative_perturbation_shares(
            rh_magnitude, w, z_km,
            transform="precomputed", relative_error=relative_error),
        # T is included as the CONTROL: temperature spans 200-300 K, i.e. well
        # under one decade, so its absolute metric should already be close to
        # the mass share.  If it is not, the probe itself is wrong.
        "absolute_T": _relative_perturbation_shares(
            T, w, z_km, transform="identity", relative_error=relative_error),
    }
    # The SAME shares over the masked domain the objective actually scores.
    w_masked, min_p_Pa = camp.thermo_mask_weights(ref)
    # The mask is derived from the reference's cold point, so REPORT the cold
    # point rather than leaving the reader to infer it from the bound.
    from legoesm.training.scm_rce_metrics import reference_cold_point
    cold_idx, _T_cold, _z_cold = reference_cold_point(
        jnp.asarray(T), jnp.asarray(ref.z_m))
    out["masked"] = {
        "min_p_Pa": float(min_p_Pa),
        "cold_point_p_Pa": float(p_full[cold_idx]),
        "cold_point_z_km": float(z_km[cold_idx]),
        "cold_point_T_K": float(T[cold_idx]),
        "n_levels": int(np.sum(w_masked > 0.0)),
        "absolute_qv": _relative_perturbation_shares(
            qv, w_masked, z_km, transform="identity",
            relative_error=relative_error),
        "log_qv": _relative_perturbation_shares(
            qv, w_masked, z_km, transform="log",
            relative_error=relative_error),
        "relative_humidity": _relative_perturbation_shares(
            rh_magnitude, w_masked, z_km, transform="precomputed",
            relative_error=relative_error),
    }
    return out


def parameter_counts() -> dict:
    """Per-scheme tunable-parameter count at each tier, from the live registry."""
    registry = build_registry()
    by_key: dict[str, list] = {}
    for meta in registry:
        by_key.setdefault(meta.scheme_key, []).append(meta)

    counts = {}
    for scheme in SCHEMES:
        cfg = camp.make_physics_config(convection=scheme)
        _component, _s, sub = camp._active_subconfig(cfg, "convection")
        key = camp._scheme_key_for_subconfig(sub)
        metas = by_key.get(key, []) if key is not None else []
        tiers = {}
        for meta in metas:
            tiers[int(meta.tunable_tier)] = tiers.get(int(meta.tunable_tier), 0) + 1
        counts[scheme] = {
            "scheme_key": key,
            "n_core": sum(1 for m in metas if 1 <= m.tunable_tier <= 1),
            "n_extended": sum(1 for m in metas if 1 <= m.tunable_tier <= 2),
            "n_aggressive": sum(1 for m in metas if 1 <= m.tunable_tier <= 3),
            "n_tier0_fixed": sum(1 for m in metas if m.tunable_tier == 0),
            "tier_histogram": dict(sorted(tiers.items())),
            "tier3_names": sorted(
                m.qualified_name for m in metas if m.tunable_tier == 3),
            "tier0_names": sorted(
                m.qualified_name for m in metas if m.tunable_tier == 0),
        }
    return counts


def _provenance(reference_dir: Path, argv: list[str]) -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    return {"git_sha": sha, "reference_dir": str(reference_dir), "argv": argv}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path,
                        default=camp.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--last-reference-files", type=int, default=5)
    parser.add_argument(
        "--relative-error", type=float, default=0.10,
        help="uniform fractional error imposed at every level (0.10 = 10%%). "
             "The SHARES are independent of its magnitude for the identity and "
             "precomputed transforms; it is exposed so that can be verified.")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the full result as JSON here")
    args = parser.parse_args(argv)

    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files)
    weighting = humidity_metric_weighting(
        ref, relative_error=args.relative_error)
    counts = parameter_counts()

    print("=== CRM reference humidity/temperature metric weighting ===")
    print(f"levels={weighting['n_levels']}  z={weighting['z_km_range'][0]:.2f}"
          f"-{weighting['z_km_range'][1]:.2f} km  "
          f"q_v sfc {weighting['qv_g_kg_surface']:.2f} -> top "
          f"{weighting['qv_g_kg_top']:.2e} g/kg  "
          f"RH {weighting['rh_min']:.3f}-{weighting['rh_max']:.3f}")
    print(f"uniform relative error = {args.relative_error:.1%} at every level")
    header = "metric".ljust(20) + "".join(
        f"{'>' + str(h) + 'km':>12s}" for h in SPLIT_HEIGHTS_KM)
    print(header)
    for name in ("absolute_qv", "log_qv", "relative_humidity", "absolute_T"):
        row = weighting[name]
        cells = "".join(
            f"{row[f'share_above_{h:g}km']:11.4%} " for h in SPLIT_HEIGHTS_KM)
        print(name.ljust(20) + cells)
    mass_row = weighting["absolute_qv"]
    print("(mass share)".ljust(20) + "".join(
        f"{mass_row[f'mass_share_above_{h:g}km']:11.4%} "
        for h in SPLIT_HEIGHTS_KM))
    m = weighting["masked"]
    print(f"\n-- reference cold point: {m['cold_point_T_K']:.1f} K at "
          f"{m['cold_point_z_km']:.2f} km / {m['cold_point_p_Pa']:.0f} Pa --")
    print(f"-- over the SCORED mask only (p >= {m['min_p_Pa']:.0f} Pa, "
          f"{m['n_levels']} levels) --")
    print(header)
    for name in ("absolute_qv", "log_qv", "relative_humidity"):
        row = m[name]
        cells = "".join(
            f"{row[f'share_above_{h:g}km']:11.4%} " for h in SPLIT_HEIGHTS_KM)
        print(name.ljust(20) + cells)
    print("(mass share)".ljust(20) + "".join(
        f"{m['absolute_qv'][f'mass_share_above_{h:g}km']:11.4%} "
        for h in SPLIT_HEIGHTS_KM))

    print("\n=== tunable parameters per convection scheme ===")
    print(f"{'scheme':<18}{'key':<20}{'core':>6}{'extended':>10}"
          f"{'aggressive':>12}{'tier0':>7}")
    for scheme in SCHEMES:
        c = counts[scheme]
        print(f"{scheme:<18}{str(c['scheme_key']):<20}{c['n_core']:>6}"
              f"{c['n_extended']:>10}{c['n_aggressive']:>12}"
              f"{c['n_tier0_fixed']:>7}")
    extra = {s: c["tier3_names"] for s, c in counts.items() if c["tier3_names"]}
    print("\ntier-3 (aggressive-only) parameters:",
          json.dumps(extra, indent=2) if extra else "none")

    payload = {
        "provenance": _provenance(args.reference_dir, list(argv or sys.argv[1:])),
        "humidity_metric_weighting": weighting,
        "parameter_counts": counts,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
