#!/usr/bin/env python
"""Out-of-sample generalization: single-SST vs joint-SST tuned parameters.

The joint campaign tuned each scheme against 295/300/305 K at once; the earlier
campaign tuned only at 300 K.  The fair question is not their in-sample scores
(different SST sets) but how each PARAMETER SET performs across ALL three SSTs.
This applies a named preset to a scheme and scores it at every SST, so
single-vs-joint can be compared on the SAME 3-SST protocol.

For each (scheme, preset), reports the per-SST tropospheric thermo score and
their mean.  A joint mean below the single mean is generalization the joint
tuning actually bought; equal or worse means it did not (or the scheme did not
tune).  Uses the campaign's own run_cached at each sst_K, so the protocol is
identical to the tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402


def _score_preset_at_ssts(scheme, preset_dir, ssts, ref_by_sst, cache, *,
                          days, dt, analysis_days):
    base = camp.make_physics_config(
        radiation="rrtmgp", convection=scheme, turbulence="louis",
        microphysics="morrison", hard_saturation_adjustment=True)
    base, _st = camp.apply_subsidence_solve_override(
        base, "implicit_flux", category="convection")
    cfg, n = camp.apply_tuned_preset(base, scheme, preset_dir) if preset_dir \
        else (base, 0)
    per = {}
    for sst in ssts:
        run = camp.run_cached(
            cache, cfg, ref_by_sst[sst], label=f"gen:{scheme}:{sst}",
            sst_K=sst, days=days, dt=dt, analysis_days=analysis_days,
            require_equilibrium=False, require_realism=False,
            equil_T_tol_K=camp.EQUIL_T_TOL_K, equil_qv_tol=camp.EQUIL_QV_TOL,
            equil_qcond_tol=camp.EQUIL_QCOND_TOL,
            scm_microphysics_substeps=30, scm_convection_substeps=10,
            coriolis_s_inv=0.0)
        per[sst] = (run.thermo_score if run.status == "ok" else float("inf"))
    return per, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemes", required=True, help="comma-separated")
    ap.add_argument("--ssts", default="295,300,305")
    ap.add_argument("--single-preset",
                    default=str(_REPO / "config/params/scm_rce_tuned_f0"))
    ap.add_argument("--joint-preset",
                    default=str(_REPO / "config/params/scm_rce_tuned_f0joint"))
    ap.add_argument("--ref-root", type=Path,
                    default=Path("/burg-archive/glab/users/pg2328/legoESM/"
                                 "results"))
    ap.add_argument("--days", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--analysis-days", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    ssts = [float(x) for x in a.ssts.split(",")]
    ref_by_sst = {s: camp.build_reference_profiles(
        a.ref_root / f"rcemip_ref_sam{int(s)}", 5) for s in ssts}
    schemes = [x.strip() for x in a.schemes.split(",") if x.strip()]
    cache: dict = {}
    results = {}
    print("%-14s %-7s %s  mean" % ("scheme", "preset",
                                   " ".join(f"{int(s)}K" for s in ssts)))
    for sch in schemes:
        row = {}
        for name, pdir in (("single", a.single_preset),
                           ("joint", a.joint_preset)):
            per, n = _score_preset_at_ssts(
                sch, pdir, ssts, ref_by_sst, cache,
                days=a.days, dt=a.dt, analysis_days=a.analysis_days)
            mean = float(np.mean([per[s] for s in ssts]))
            row[name] = {"per_sst": {int(s): per[s] for s in ssts},
                         "mean": mean, "n_params": n}
            print("%-14s %-7s %s  %.2f" % (
                sch, name, " ".join(f"{per[s]:.2f}" for s in ssts), mean))
        # Verdict: lower 3-SST mean wins.
        d = row["single"]["mean"] - row["joint"]["mean"]
        row["joint_better_by"] = d
        print("  -> joint %s single by %.2f\n" % (
            "BEATS" if d > 0 else "loses to", abs(d)))
        results[sch] = row

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
