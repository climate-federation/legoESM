#!/usr/bin/env python
"""Is the tuner's candidate-zero the SAME configuration as the a-priori run?

``_candidate_values`` yields the defaults first, so evaluation zero is supposed
to reproduce the a-priori column exactly and hit the same cache entry.  On the
2026-08-16 emanuel arm it did NOT: all 25 tunable fields came back bit-identical
yet the two runs produced different scores (6.9607 vs 6.7207) and different
cache keys, and the tuner recorded that difference as an "improvement" of 0.24
with zero parameters moved.

Either the round trip
``defaults -> raw -> TrainablePhysicsParams -> to_overrides ->
apply_param_overrides -> _rewrap_tunable_subconfig -> _set_active_subconfig``
changes something the tunable-field list does not cover, or it does not and the
two runs differ for another reason.  This probe answers which, by diffing the
two configurations FIELD BY FIELD and comparing the cache keys the campaign
itself would compute.

Reports, per scheme: whether the keys match, and every leaf whose value or type
differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.numpy as jnp  # noqa: E402

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402
from legoesm.training.param_collector import build_trainable_params  # noqa: E402
from legoesm.training.trainable_params import (  # noqa: E402
    TrainablePhysicsParams,
)

DEFAULT_SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)


def _flatten(obj, prefix=""):
    """Leaf path -> (repr, type name) for a nested jsonable structure."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = (repr(obj), type(obj).__name__)
    return out


def candidate_zero_config(base_cfg, scheme_key, subcfg, param_set: str):
    """Rebuild EXACTLY what the tuner evaluates as candidate zero."""
    tier, include_tier0, exclude = camp.resolve_param_selection(
        scheme_key, param_set)
    params = build_trainable_params(
        active_scheme_keys={scheme_key}, tier=tier,
        include_tier0=include_tier0, exclude=exclude, dtype=jnp.float64)
    constraints = params.constraints
    if not constraints:
        return None, {}
    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}
    raw_values = {
        c.name: camp._raw_from_physical(defaults[c.name], c)
        for c in constraints
    }
    trial = TrainablePhysicsParams(raw_values=raw_values,
                                   constraints=constraints)
    field_values = trial.to_overrides().get(scheme_key, {})
    tuned_tunable = camp.apply_param_overrides(
        camp._tunable_subconfig(subcfg), field_values)
    tuned_subcfg = camp._rewrap_tunable_subconfig(subcfg, tuned_tunable)
    trial_cfg = camp._set_active_subconfig(base_cfg, "convection", tuned_subcfg)
    return trial_cfg, {c.field: float(field_values[c.field]) for c in constraints}


def compare(scheme: str, param_set: str, *, days: float, dt: float) -> dict:
    base_cfg = camp.make_physics_config(
        radiation="rrtmgp", convection=scheme, microphysics="morrison",
        hard_saturation_adjustment=True)
    base_cfg, _status = camp.apply_subsidence_solve_override(
        base_cfg, "implicit_flux", category="convection")
    _component, _s, subcfg = camp._active_subconfig(base_cfg, "convection")
    scheme_key = camp._scheme_key_for_subconfig(subcfg)
    if scheme_key is None:
        return {"scheme": scheme, "skipped": "no scheme key"}

    trial_cfg, values = candidate_zero_config(
        base_cfg, scheme_key, subcfg, param_set)
    if trial_cfg is None:
        return {"scheme": scheme, "skipped": "no tunable parameters"}

    key_args = dict(days=days, dt=dt, scm_microphysics_substeps=30,
                    scm_convection_substeps=10, surface_wind_m_s=5.0,
                    coriolis_s_inv=0.0, large_scale_forcing="none",
                    bl_anchor_top_m=-1.0, subcloud_top_m=1000.0,
                    thermo_humidity="logq")
    base_key = camp._config_cache_key(
        base_cfg, key_args["days"], key_args["dt"],
        key_args["scm_microphysics_substeps"],
        key_args["scm_convection_substeps"], key_args["surface_wind_m_s"],
        key_args["coriolis_s_inv"], key_args["large_scale_forcing"],
        key_args["bl_anchor_top_m"], key_args["subcloud_top_m"],
        key_args["thermo_humidity"])
    trial_key = camp._config_cache_key(
        trial_cfg, key_args["days"], key_args["dt"],
        key_args["scm_microphysics_substeps"],
        key_args["scm_convection_substeps"], key_args["surface_wind_m_s"],
        key_args["coriolis_s_inv"], key_args["large_scale_forcing"],
        key_args["bl_anchor_top_m"], key_args["subcloud_top_m"],
        key_args["thermo_humidity"])

    a = _flatten(camp._to_jsonable(base_cfg))
    b = _flatten(camp._to_jsonable(trial_cfg))
    diffs = []
    for path in sorted(set(a) | set(b)):
        va, vb = a.get(path), b.get(path)
        if va != vb:
            diffs.append({"path": path, "apriori": va, "candidate0": vb})
    return {
        "scheme": scheme,
        "scheme_key": scheme_key,
        "n_params": len(values),
        "keys_match": base_key == trial_key,
        "n_leaf_diffs": len(diffs),
        "diffs": diffs[:40],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", default=",".join(DEFAULT_SCHEMES))
    parser.add_argument("--param-set", default="physical")
    parser.add_argument("--days", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = []
    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        r = compare(scheme, args.param_set, days=args.days, dt=args.dt)
        results.append(r)
        if "skipped" in r:
            print(f"{scheme:16s} SKIPPED ({r['skipped']})")
            continue
        flag = "OK  " if r["keys_match"] else "DIFF"
        print(f"{scheme:16s} {flag} keys_match={r['keys_match']} "
              f"leaf_diffs={r['n_leaf_diffs']} n_params={r['n_params']}")
        for d in r["diffs"][:8]:
            print(f"    {d['path']}: a-priori={d['apriori']} "
                  f"candidate0={d['candidate0']}")

    bad = [r for r in results if not r.get("keys_match", True)]
    print(f"\n{len(bad)} of {len(results)} scheme(s) have a candidate-zero "
          "that is NOT the a-priori configuration")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
