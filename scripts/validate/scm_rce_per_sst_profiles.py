#!/usr/bin/env python
"""Per-RCE (per-SST) score AND profiles for a tuned parameter set.

Runs one convection scheme with a named preset at each SST (295/300/305) and
records, per SST, the tropospheric thermo score plus the equilibrium T and q_v
profiles.  One JSON per scheme; a companion plotter turns these into the
per-RCE ranking and the per-RCE profile panels.

The preset defaults to the JOINT (295/300/305-tuned) set; a scheme that did not
move under the joint budget falls back to its shipped defaults inside
apply_tuned_preset, which is the honest thing to score.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--ssts", default="295,300,305")
    ap.add_argument("--preset",
                    default=str(_REPO / "config/params/scm_rce_tuned_f0joint"))
    ap.add_argument("--ref-root", type=Path,
                    default=Path("/burg-archive/glab/users/pg2328/legoESM/"
                                 "results"))
    ap.add_argument("--days", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--analysis-days", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    ssts = [float(x) for x in a.ssts.split(",")]
    base = camp.make_physics_config(
        radiation="rrtmgp", convection=a.scheme, turbulence="louis",
        microphysics="morrison", hard_saturation_adjustment=True)
    base, _ = camp.apply_subsidence_solve_override(
        base, "implicit_flux", category="convection")
    cfg, n = camp.apply_tuned_preset(base, a.scheme, a.preset)

    cache: dict = {}
    per = {}
    for sst in ssts:
        ref = camp.build_reference_profiles(
            a.ref_root / f"rcemip_ref_sam{int(sst)}", 5)
        run = camp.run_cached(
            cache, cfg, ref, label=f"persst:{a.scheme}:{sst}", sst_K=sst,
            days=a.days, dt=a.dt, analysis_days=a.analysis_days,
            require_equilibrium=False, require_realism=False,
            equil_T_tol_K=camp.EQUIL_T_TOL_K, equil_qv_tol=camp.EQUIL_QV_TOL,
            equil_qcond_tol=camp.EQUIL_QCOND_TOL,
            scm_microphysics_substeps=30, scm_convection_substeps=10,
            coriolis_s_inv=0.0)
        ok = run.status == "ok" and bool(run.T_profile)
        per[int(sst)] = {
            "thermo": run.thermo_score if ok else float("inf"),
            "status": run.status,
            "T_profile": list(map(float, run.T_profile)) if ok else None,
            "qv_profile": list(map(float, run.qv_profile)) if ok else None,
            "z_m": list(map(float, np.asarray(ref.z_m))),
            "T_ref": list(map(float, np.asarray(ref.T_ref))),
            "qv_ref": list(map(float, np.asarray(ref.qv_ref))),
        }
        print(f"{a.scheme} {int(sst)}K: thermo="
              f"{per[int(sst)]['thermo']:.2f} status={run.status}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"scheme": a.scheme, "n_preset_params": n, "per_sst": per}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
