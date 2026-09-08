#!/usr/bin/env python
"""Is the SCM's near-surface humidity bias common to ALL convection schemes?

That single question decides where to look.  A bias shared by ten independent
convection schemes CANNOT be caused by convection; it has to come from something
they all sit on — the surface fluxes, the boundary-layer scheme, the forcing, or
the initial column.  A bias that varies by scheme is the opposite.

Reports, per scheme and against the CRM reference, the water-vapour error at the
surface and at 0.5 / 1 / 2 km, plus the near-surface temperature error and the
bulk-flux state the surface actually sees (relative humidity, SST - T_air).
Then the ACROSS-SCHEME mean and spread, which is the discriminator.

Everything is read from the campaign's own checkpoints and the shared reference
builder; nothing is re-derived.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402

#: Heights at which the humidity error is reported [km].  The surface and the
#: sub-cloud layer are where the reported bias lives; 2 km is above cloud base
#: and is included so a bias confined to the boundary layer is distinguishable
#: from a whole-column one.
REPORT_Z_KM = (0.5, 1.0, 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-dir", type=Path, required=True,
        help="a completed arm directory holding scheme_*.json checkpoints")
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("/burg-archive/glab/users/pg2328/legoESM/results"
                     "/rcemip_ref_sam300"))
    parser.add_argument("--last-reference-files", type=int, default=5)
    parser.add_argument("--condition", default="tuned",
                        choices=("tuned", "prior"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if "rcemip1_n128" in str(args.reference_dir):
        raise SystemExit(
            f"REFUSED: {args.reference_dir} is our own CRM run, not an oracle.")

    ref = camp.build_reference_profiles(
        args.reference_dir, args.last_reference_files)
    z_km = np.asarray(ref.z_m) / 1000.0
    qv_ref = np.asarray(ref.qv_ref) * 1000.0          # g/kg
    T_ref = np.asarray(ref.T_ref)
    idx = {zk: int(np.argmin(np.abs(z_km - zk))) for zk in REPORT_Z_KM}

    print(f"CRM reference: qv surface {qv_ref[-1]:.3f} g/kg, "
          + ", ".join(f"{zk:g} km {qv_ref[idx[zk]]:.3f}" for zk in REPORT_Z_KM)
          + f" | T surface {T_ref[-1]:.2f} K")
    print("CRM sub-cloud bulk state: RH 0.752, SST-T_air 3.06 K, "
          "evaporation 2.73 mm/day (campaign-recorded)")
    print()

    header = ("%-16s %9s " % ("scheme", "dqv_sfc")
              + " ".join("%9s" % f"dqv_{zk:g}km" for zk in REPORT_Z_KM)
              + " %8s %7s %9s %8s" % ("dT_sfc", "RH", "SST-Ta", "E"))
    print(header)

    rows = []
    for path in sorted(glob.glob(str(args.arm_dir / "scheme_*.json"))):
        d = json.loads(Path(path).read_text())
        run = d[args.condition]
        if not run.get("qv_profile"):
            continue
        qv = np.asarray(run["qv_profile"]) * 1000.0
        T = np.asarray(run["T_profile"])
        rec = {
            "scheme": d["scheme"],
            "dqv_sfc": float(qv[-1] - qv_ref[-1]),
            "dT_sfc": float(T[-1] - T_ref[-1]),
            "sfc_rh": float(run.get("sfc_relative_humidity", float("nan"))),
            "sfc_dT": float(run.get("sfc_delta_T_K", float("nan"))),
            "evap": float(run.get("evap_mm_day", float("nan"))),
        }
        for zk in REPORT_Z_KM:
            rec[f"dqv_{zk:g}km"] = float(qv[idx[zk]] - qv_ref[idx[zk]])
        rows.append(rec)
        print("%-16s %9.3f " % (rec["scheme"], rec["dqv_sfc"])
              + " ".join("%9.3f" % rec[f"dqv_{zk:g}km"] for zk in REPORT_Z_KM)
              + " %8.2f %7.3f %9.2f %8.2f"
              % (rec["dT_sfc"], rec["sfc_rh"], rec["sfc_dT"], rec["evap"]))

    if not rows:
        raise SystemExit(f"no checkpoints with profiles in {args.arm_dir}")

    keys = ["dqv_sfc"] + [f"dqv_{zk:g}km" for zk in REPORT_Z_KM] + [
        "dT_sfc", "sfc_rh", "sfc_dT", "evap"]
    stats = {}
    print()
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        stats[k] = {"mean": float(v.mean()), "min": float(v.min()),
                    "max": float(v.max()), "spread": float(v.max() - v.min())}
        print("%-12s mean %+8.3f   range %+8.3f .. %+8.3f   spread %7.3f"
              % (k, v.mean(), v.min(), v.max(), v.max() - v.min()))

    # THE DISCRIMINATOR, stated rather than left to the reader.
    m = stats.get("dqv_sfc", {})
    if m:
        common = abs(m["mean"]) > m["spread"]
        print()
        print("VERDICT: the surface humidity bias is "
              + ("COMMON to all schemes (|mean| %.3f > spread %.3f), so it "
                 "CANNOT originate in convection — look at the surface fluxes, "
                 "the boundary-layer scheme, or the forcing."
                 % (abs(m["mean"]), m["spread"])
                 if common else
                 "SCHEME-DEPENDENT (|mean| %.3f <= spread %.3f), so convection "
                 "is implicated." % (abs(m["mean"]), m["spread"])))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"condition": args.condition, "per_scheme": rows,
             "across_scheme": stats}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
