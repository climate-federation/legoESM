"""Accuracy of a plane-CRM run against an fp64 reference run.

Compares ``rce_checkpoint`` files of two ``run_rcemip_plane.py`` runs made
from the same deck (e.g. ``--precision mixed`` vs ``--precision float64``)
at the last common checkpoint step: relative RMS error per field and the
relative change of dry mass and total water (vapour + condensate slots 1-5).

Acceptance (user-proposed, recorded in the commit): theta', rho' <= 1e-4 and
u, w <= 1e-3 relative RMS after 1 h of the RCEMIP1 deck; and (voted 3-0) the
budgets: |total-water difference| <= 4e-8 (the previous mixed result, the
user's bar) and |dry-mass difference| <= 1e-11. ``--run-precision mixed``
evaluates the run's budgets with rho_ref / dz as the mixed model stores them
(fp32), summed in fp64.

    python scripts/validate/crm_mixed_precision_accuracy.py REF_DIR RUN_DIR \
        --nlev 64 --dx 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = {"theta_prime": 1e-4, "rho_prime": 1e-4, "u": 1e-3, "w": 1e-3}
BUDGET_THRESHOLDS = {"water_rel_diff": 4e-8, "dry_mass_rel_diff": 1e-11}


def relative_rms(a: np.ndarray, ref: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    denom = np.sqrt(np.mean(ref ** 2))
    if denom == 0.0:
        raise ValueError("reference field is identically zero")
    return float(np.sqrt(np.mean((a - ref) ** 2)) / denom)


def budgets(ckpt, rho_ref: np.ndarray, dz: np.ndarray) -> dict:
    rho = rho_ref + np.asarray(ckpt["rho_prime"], np.float64)
    tr = np.asarray(ckpt["tracers"], np.float64)
    water = tr[..., 0] + tr[..., 1:6].sum(-1)
    return {"dry_mass": float((rho * dz).sum()),
            "water": float((rho * water * dz).sum())}


def compare(ref, run, rho_ref, dz, run_geometry=None) -> dict:
    """``run_geometry=(rho_ref, dz)`` for the run when it stores its own
    (e.g. fp32 in mixed precision); default: the reference geometry."""
    out = {k: relative_rms(run[k], ref[k]) for k in ("u", "v", "w", "theta_prime", "rho_prime")}
    b_ref = budgets(ref, rho_ref, dz)
    b_run = budgets(run, *(run_geometry or (rho_ref, dz)))
    for k in b_ref:
        out[f"{k}_rel_diff"] = (b_run[k] - b_ref[k]) / b_ref[k]
    finite = all(np.isfinite(np.asarray(run[k], np.float64)).all()
                 for k in ("u", "v", "w", "theta_prime", "rho_prime", "tracers"))
    out["finite"] = bool(finite and all(np.isfinite(x) for x in out.values()))
    out["pass"] = (out["finite"]
                   and all(out[k] <= v for k, v in THRESHOLDS.items())
                   and all(abs(out[k]) <= v for k, v in BUDGET_THRESHOLDS.items()))
    return out


def _last_common(ref_dir: Path, run_dir: Path) -> str:
    names = sorted({p.name for p in (ref_dir / "checkpoints").glob("ckpt_*.npz")}
                   & {p.name for p in (run_dir / "checkpoints").glob("ckpt_*.npz")})
    if not names:
        raise SystemExit("no common checkpoint step between the two runs")
    return names[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref_dir", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--nlev", type=int, required=True)
    p.add_argument("--dx", type=float, required=True)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--run-precision", choices=["float64", "mixed"], default="float64",
                   help="mixed: the run's budgets use rho_ref/dz rounded to fp32, "
                        "as the mixed model stores them (summed in fp64).")
    a = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import run_rcemip_plane as rcp
    rargs = rcp.parse_args(["--nlev", str(a.nlev), "--dx", str(a.dx),
                            "--T-sfc", str(a.T_sfc), "--precision", "float64"])
    hc, _, _ = rcp.build_rcemip_height_coord(rargs, rcp.P_SFC_RCEMIP, np.float64)
    name = _last_common(a.ref_dir, a.run_dir)
    geo = (np.asarray(hc.rho_ref, np.float64), np.asarray(hc.dz, np.float64))
    run_geo = (tuple(g.astype(np.float32).astype(np.float64) for g in geo)
               if a.run_precision == "mixed" else None)
    res = compare(np.load(a.ref_dir / "checkpoints" / name),
                  np.load(a.run_dir / "checkpoints" / name), *geo,
                  run_geometry=run_geo)
    res["checkpoint"] = name
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
