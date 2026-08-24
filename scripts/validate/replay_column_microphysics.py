"""Offline single-column Morrison + hard-sat-drain replay on a saved MPAS state.

Purpose (issue #1515): replay ONE physics step of the Morrison microphysics
(and the post-step hard-saturation drain) on selected columns of a saved
MPAS checkpoint / blowup autopsy state, with per-term instrumentation and
single-input perturbation attribution — WITHOUT any GPU rollout.

The replay mirrors the MPAS lane's microphysics adapter
(`legoesm.atmosphere.physics.microphysics.integration._make_hydrostatic_microphysics`)
exactly:

* tracers are clipped to >= 0 on extraction (the adapter's ``_get_tracer``),
* ``p_half/p_full = sigma * p_s`` (pure-sigma; asserts A_half == 0),
* ``rho = compute_rho(T, p_full, q_v)`` (virtual T),
* ``dz = compute_layer_dz(T, p_half, q_v)`` (hypsometric, TOA-first),
* Morrison config resolved through the SAME applier the driver uses
  (`apply_microphysics_experiment_flags`), fed from the run's
  ``run_manifest.json`` resolved_config.

The post-step drain replicates ``model_driver._mpas_hard_saturation_poststep``
via the shared reviewed core ``_warm_rain.hard_saturation_drain`` on
``p_full = p_s * sigma_full`` (the hook's own convention).

NaN policy: every NaN/Inf in inputs or outputs is COUNTED and reported, and
the exit status is non-zero if any output NaN appears — nanmax/nanmean are
never used.

Provenance: the header stamps the git SHA of the legoesm tree ACTUALLY
imported (which may be a pinned tree via PYTHONPATH), the state path, and the
full argv, so A/B runs across trees are self-describing.

Usage
-----
    python scripts/validate/replay_column_microphysics.py \
        --state /path/blowup_state_day_0968.npz --cells 1432 [--perturb] \
        [--manifest /path/run_manifest.json] [--json-out out.json]

``--cells auto`` picks the column with the largest vertical T range.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# NOTE: imports of legoesm happen AFTER argparse so `--help` works without JAX.


def _git_sha_of(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    try:
        out = subprocess.run(
            ["git", "-C", d, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _count_nonfinite(name, arr, ledger):
    n = int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if n:
        ledger.append((name, n))
    return n


def build_morrison_config(resolved: dict):
    """Resolve the run's MorrisonConfig via the driver's own applier."""
    from legoesm.atmosphere.physics.microphysics.config import (
        MicrophysicsConfig,
        apply_microphysics_experiment_flags,
    )
    if resolved.get("microphysics") != "morrison":
        raise SystemExit(
            f"run used microphysics={resolved.get('microphysics')!r}; this "
            "probe replays morrison only")
    sub = MicrophysicsConfig(scheme="morrison").morrison
    scalars = {
        leaf: resolved.get(f"morrison_{leaf}")
        for leaf in ("bergeron_rate", "rime_coeff", "dep_coeff", "agg_coeff",
                     "k_au", "fall_a_i", "ice_snow_d_auto", "hom_ice_nuc_N")
    }
    return apply_microphysics_experiment_flags(
        sub, "morrison",
        nc_from_aerosol=bool(resolved.get("nc_from_aerosol", False)),
        subgrid_autoconversion=bool(resolved.get("subgrid_autoconversion",
                                                 False)),
        hard_saturation_adjustment=bool(
            resolved.get("hard_saturation_adjustment", False)),
        hard_sat_adjust_threshold=resolved.get("hard_sat_adjust_threshold"),
        hard_sat_max_heating_K=resolved.get("hard_sat_max_heating_K"),
        homogeneous_ice_nucleation=bool(
            resolved.get("homogeneous_ice_nucleation", False)),
        morrison_scalars=scalars,
        morrison_flavor=resolved.get("morrison_flavor"),
    )


def replay_cells(state, cells, cfg, dt, *, perturb_key=None,
                 perturb_scale=0.0):
    """One Morrison step + drain on ``cells``; returns a result dict.

    ``perturb_key``: optional tracer name ('N_r','N_i','q_r','q_s','q_g',
    'q_c') multiplied by ``perturb_scale`` before the call (single-input
    perturbation attribution).
    """
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics._shared import (
        compute_layer_dz,
        compute_rho,
    )
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        hard_saturation_drain,
    )
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import (
        HydrometeorState,
    )

    vg = np.asarray(state["meta_vgrid"], dtype=np.float64)
    A_half, B_half = vg[0], vg[1]
    if not np.allclose(A_half, 0.0):
        raise SystemExit("hybrid A_half != 0: this probe handles the pure-"
                         "sigma MPAS lane only (extend for hybrid)")
    sigma_half = B_half
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])

    cells = np.asarray(cells, dtype=int)
    p_s = np.asarray(state["p_s"], dtype=np.float64)[cells]
    T = np.asarray(state["T"], dtype=np.float64)[cells]

    def trc(name):
        arr = np.asarray(state[f"trc_{name}"], dtype=np.float64)[cells]
        clipped = np.maximum(arr, 0.0)     # adapter's _get_tracer clip
        if perturb_key == name:
            clipped = clipped * perturb_scale
        return jnp.asarray(clipped)

    ledger = []
    _count_nonfinite("T", T, ledger)
    _count_nonfinite("p_s", p_s, ledger)

    q_v = trc("q_v")
    T_j = jnp.asarray(T)
    p_full = jnp.asarray(sigma_full)[None, :] * jnp.asarray(p_s)[:, None]
    p_half = jnp.asarray(sigma_half)[None, :] * jnp.asarray(p_s)[:, None]
    rho = compute_rho(T_j, p_full, q_v)
    # Mirror the production bridge: stored N_c/N_r are per MASS [1/kg]; the
    # scheme wants per VOLUME.  A replay that skips this no longer replays.
    from legoesm.atmosphere.physics.microphysics.integration import (
        number_per_mass_to_per_volume,
    )
    hyd = HydrometeorState(
        q_c=trc("q_c"), q_r=trc("q_r"), q_i=trc("q_i"), q_s=trc("q_s"),
        q_g=trc("q_g"),
        N_c=number_per_mass_to_per_volume(trc("N_c"), rho),
        N_r=number_per_mass_to_per_volume(trc("N_r"), rho),
        N_i=trc("N_i"),
    )
    dz = compute_layer_dz(T_j, p_half, q_v)

    out = morrison_microphysics(T_j, q_v, hyd, p_full, p_half, rho, dz,
                                float(dt), cfg)

    res = {"cells": cells.tolist(), "dt": float(dt),
           "perturb": None if perturb_key is None
           else {perturb_key: perturb_scale}}
    deltas = {}
    for fname in ("dT_dt", "dq_v_dt", "dq_c_dt", "dq_r_dt", "dq_i_dt",
                  "dq_s_dt", "dq_g_dt", "dN_c_dt", "dN_r_dt", "dN_i_dt"):
        arr = np.asarray(getattr(out, fname), dtype=np.float64)
        _count_nonfinite(fname, arr, ledger)
        d = arr * float(dt)
        deltas[fname] = d
        res[f"max_abs_{fname}_x_dt"] = float(np.max(np.abs(d)))
    res["precip_kg_m2_s"] = [float(v) for v in np.asarray(out.precipitation)]
    _count_nonfinite("precipitation", np.asarray(out.precipitation), ledger)

    # Post-step hard-saturation drain (driver hook convention: p = sigma*p_s,
    # liquid curve — hard_sat_ice_curve False on this run).
    drain_rate = hard_saturation_drain(
        T_j, q_v, p_full, float(dt),
        float(cfg.hard_sat_adjust_threshold),
        float(cfg.hard_sat_max_heating_K))
    dq_drain = np.asarray(drain_rate, dtype=np.float64) * float(dt)
    _count_nonfinite("drain_dq", dq_drain, ledger)
    res["drain_max_dq_g_kg"] = float(np.max(dq_drain) * 1e3)
    res["drain_max_heating_K"] = float(
        np.max(dq_drain) * constants.L_v / constants.c_pd)

    res["nonfinite"] = [{"field": k, "count": n} for k, n in ledger]
    res["deltas"] = deltas
    res["profile"] = {
        "T": T, "q_v": np.asarray(q_v), "q_c": np.asarray(hyd.q_c),
        "q_r": np.asarray(hyd.q_r), "N_r": np.asarray(hyd.N_r),
        "drain_dq": dq_drain,
    }
    return res


def print_report(res, label):
    print(f"\n=== {label} (cells {res['cells']}, dt={res['dt']} s, "
          f"perturb={res['perturb']}) ===")
    for k in sorted(res):
        if k.startswith("max_abs_"):
            print(f"  {k:28s} {res[k]:.6g}")
    print(f"  drain: max dq {res['drain_max_dq_g_kg']:.4g} g/kg "
          f"(heating {res['drain_max_heating_K']:.3g} K)")
    print(f"  precip {['%.3g' % p for p in res['precip_kg_m2_s']]} kg/m2/s")
    if res["nonfinite"]:
        print(f"  NON-FINITE VALUES: {res['nonfinite']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--cells", default="auto",
                    help="comma-separated cell indices, or 'auto' (max "
                         "vertical T range)")
    ap.add_argument("--manifest", default=None,
                    help="run_manifest.json (default: sibling of --state)")
    ap.add_argument("--dt", type=float, default=None,
                    help="physics dt [s] (default: manifest dycore.dt)")
    ap.add_argument("--perturb", action="store_true",
                    help="single-input perturbation attribution sweep")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    manifest_path = args.manifest or os.path.join(
        os.path.dirname(os.path.abspath(args.state)), "run_manifest.json")
    with open(manifest_path) as fh:
        resolved = json.load(fh)["config"]["resolved_config"]
    dt = args.dt if args.dt is not None else float(resolved["dycore"]["dt"])

    state = np.load(args.state)

    import legoesm.atmosphere  # noqa: F401  (resolve the active tree)
    import legoesm.atmosphere.physics.microphysics.morrison as _mor
    tree_sha = _git_sha_of(_mor.__file__)
    print(f"legoesm tree : {_mor.__file__}")
    print(f"tree git SHA : {tree_sha}")
    print(f"state        : {args.state}")
    print(f"manifest     : {manifest_path}")
    print(f"argv         : {sys.argv}")
    print(f"x64          : {os.environ.get('JAX_ENABLE_X64')}")

    if args.cells == "auto":
        T = np.asarray(state["T"])
        rng = T.max(axis=1) - T.min(axis=1)
        cells = [int(np.argmax(rng))]
        print(f"auto-selected cell {cells[0]} "
              f"(T range {rng[cells[0]]:.1f} K)")
    else:
        cells = [int(c) for c in args.cells.split(",")]

    cfg = build_morrison_config(resolved)
    print(f"morrison flavor={cfg.morrison_flavor} "
          f"hard_adjust={cfg.hard_saturation_adjustment} "
          f"thr={cfg.hard_sat_adjust_threshold} "
          f"cap={cfg.hard_sat_max_heating_K} K "
          f"predict_Nc={getattr(cfg, 'predict_Nc', None)} "
          f"Nc_0={getattr(cfg, 'Nc_0', None)}")

    results = {"tree_sha": tree_sha, "state": args.state,
               "argv": sys.argv, "runs": []}
    base = replay_cells(state, cells, cfg, dt)
    print_report(base, "BASELINE")
    results["runs"].append({"label": "baseline",
                            **{k: v for k, v in base.items()
                               if k not in ("deltas", "profile")}})

    # Worst-cell per-level table for the baseline.
    i_worst = int(np.argmax(np.max(np.abs(base["deltas"]["dT_dt"]), axis=1)))
    prof = base["profile"]
    dT = base["deltas"]["dT_dt"][i_worst]
    dqv = base["deltas"]["dq_v_dt"][i_worst]
    dqr = base["deltas"]["dq_r_dt"][i_worst]
    dqs = base["deltas"]["dq_s_dt"][i_worst]
    print(f"\nper-level (cell {base['cells'][i_worst]}): "
          "k T qv qc qr N_r | dT dqv dqr dqs drain_dq  (x dt)")
    for k in range(dT.shape[0]):
        print(f"  {k:2d} {prof['T'][i_worst, k]:7.1f} "
              f"{prof['q_v'][i_worst, k]:9.5f} "
              f"{prof['q_c'][i_worst, k]:9.5f} "
              f"{prof['q_r'][i_worst, k]:9.5f} "
              f"{prof['N_r'][i_worst, k]:9.3g} | "
              f"{dT[k]:+9.3f} {dqv[k]:+9.5f} {dqr[k]:+9.5f} "
              f"{dqs[k]:+9.5f} {prof['drain_dq'][i_worst, k]:9.5f}")

    if args.perturb:
        for key, scale in (("N_r", 0.0), ("N_r", 0.5), ("N_i", 0.0),
                           ("q_r", 0.0), ("q_s", 0.0), ("q_g", 0.0),
                           ("q_c", 0.0)):
            r = replay_cells(state, cells, cfg, dt,
                             perturb_key=key, perturb_scale=scale)
            print_report(r, f"PERTURB {key} x {scale}")
            results["runs"].append({"label": f"perturb_{key}_x{scale}",
                                    **{k: v for k, v in r.items()
                                       if k not in ("deltas", "profile")}})

    n_nan = sum(e["count"] for r in results["runs"]
                for e in r["nonfinite"])
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.json_out}")
    if n_nan:
        print(f"\nFATAL: {n_nan} non-finite values encountered")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
