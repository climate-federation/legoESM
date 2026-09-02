"""Cross-site skill ladder for the P-model EC scoring campaign (PR#1698).

Reads ``run_ec_site.py`` outputs from one directory PER ARM (the
``diagnostics/ec_site_pmodel/<arm>/`` layout written by
``scripts/cluster/ec_site/run_pmodel_xsite*.sbatch``), scores every site with
the SAME skill code as ``plot_ec_site_xsite`` (imported, not re-derived), and
writes:

* ``ladder_skill.csv``  — per (arm, site, flux): n, r, rmse, bias
* ``ladder_summary.md`` — per (arm, flux): cross-site median r / RMSE / |bias|,
  plus the per-flux delta vs the arm's declared parent (one-variable ladder),
  and the count of sites improved/degraded in r.

Only sites present in EVERY arm enter the summary (matched-site protocol —
an arm that failed on a site must not win by dropping it).

Usage::

    python scripts/validate/ec_pmodel_ladder.py \
        --base-dir diagnostics/ec_site_pmodel \
        --arms base_bb base_med cap cap_g1 \
        --parents none base_bb base_med cap
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import xarray as xr

# Single home for the skill math: reuse the xsite collector's helpers.
import importlib.util as _ilu

_HERE = os.path.dirname(os.path.abspath(__file__))
_XSITE = os.path.join(_HERE, "..", "plot", "plot_ec_site_xsite.py")
_spec = _ilu.spec_from_file_location("_xsite", _XSITE)
_xsite = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_xsite)

_FLUXES = [("gpp", "GPP"), ("le", "LE"), ("h", "H")]


def collect_arm(arm_dir: str) -> dict[str, dict[str, dict]]:
    """site -> flux -> skill dict, using the xsite _skill on each output."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(arm_dir, "*_ec_*.nc"))):
        ds = xr.open_dataset(path)
        site = os.path.basename(path).split("_driver_v2")[0]
        out[site] = {label: _xsite._skill(ds, key) for key, label in _FLUXES}
        ds.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="diagnostics/ec_site_pmodel")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--parents", nargs="+", required=True,
                    help="parent arm per arm ('none' for the anchor); same "
                         "length as --arms")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if len(args.parents) != len(args.arms):
        raise SystemExit("--parents must match --arms one-for-one")
    out_dir = args.out_dir or args.base_dir

    arms = {a: collect_arm(os.path.join(args.base_dir, a)) for a in args.arms}
    for a, sites in arms.items():
        print(f"[{a}] {len(sites)} site outputs")
    common = sorted(set.intersection(*(set(s) for s in arms.values())))
    if not common:
        raise SystemExit("no site present in every arm - nothing comparable")
    dropped = {a: sorted(set(arms[a]) - set(common)) for a in args.arms}
    for a, d in dropped.items():
        if d:
            print(f"[{a}] dropped from matched set (missing elsewhere): {d}")
    print(f"matched sites ({len(common)}): {common}")

    rows = []
    for a in args.arms:
        for site in common:
            for _, label in _FLUXES:
                s = arms[a][site][label]
                rows.append((a, site, label, s["n"], s["r"], s["rmse"],
                             s["bias"]))
    csv = os.path.join(out_dir, "ladder_skill.csv")
    with open(csv, "w") as fh:
        fh.write("arm,site,flux,n,r,rmse,bias\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    def med(a, label, key):
        vals = [arms[a][s][label][key] for s in common]
        return float(np.nanmedian(vals))

    lines = ["# P-model EC scoring ladder (matched sites: "
             f"{len(common)})", "",
             "| arm | flux | median r | median RMSE | median bias | "
             "dr vs parent | sites r up/down |",
             "|---|---|---|---|---|---|---|"]
    for a, parent in zip(args.arms, args.parents):
        for _, label in _FLUXES:
            r_m, rm_m, b_m = (med(a, label, "r"), med(a, label, "rmse"),
                              med(a, label, "bias"))
            if parent == "none":
                dr, updown = "-", "-"
            else:
                dr = f"{r_m - med(parent, label, 'r'):+.3f}"
                ups = sum(arms[a][s][label]["r"] > arms[parent][s][label]["r"]
                          for s in common)
                downs = sum(
                    arms[a][s][label]["r"] < arms[parent][s][label]["r"]
                    for s in common)
                updown = f"{ups}/{downs}"
            lines.append(f"| {a} | {label} | {r_m:.3f} | {rm_m:.2f} | "
                         f"{b_m:+.2f} | {dr} | {updown} |")
    md = os.path.join(out_dir, "ladder_summary.md")
    with open(md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {csv}\n-> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
