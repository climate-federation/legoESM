#!/usr/bin/env python
"""Publication figure: tuned T and q_v profiles vs the SAM CRM, with the
across-seed spread as an envelope.

The single-seed profile hides the campaign's central finding — that the tuned
parameters are non-identifiable, so a scheme's profile is one draw from a
distribution, not a line.  This plots, per scheme, the SEED-MEAN tuned profile
with a shaded min-max band across all five independent searches, over the CRM
reference and the prior (untuned) mean.  A wide band is not a plotting artifact;
it is the honest statement that the fit is under-determined.

Two panels (T, q_v), tropospheric only (<=15 km), schemes ordered by their
seed-mean tuned thermo score so the legend reads best-to-worst.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run import run_scm_rce_campaign as camp  # noqa: E402

PLOT_TOP_KM = 15.0


def _seed_dirs(root: Path, seeds: list[str], suffix: str) -> list[Path]:
    return [root / f"arm_thermo_physical_{suffix}_seed{s}" for s in seeds]


def _collect(dirs: list[Path]):
    """Return {scheme: {'T': (nseed,nlev), 'q': ..., 'Tpri':.., 'qpri':..,
    'score':[..]}} keyed over seeds that actually have the scheme."""
    out: dict[str, dict] = {}
    for d in dirs:
        for p in sorted(glob.glob(str(d / "scheme_*.json"))):
            j = json.loads(Path(p).read_text())
            name = j["scheme"]
            t, pr = j["tuned"], j["prior"]
            if not t.get("T_profile"):
                continue
            rec = out.setdefault(name, {k: [] for k in
                                        ("T", "q", "Tpri", "qpri", "score")})
            rec["T"].append(np.asarray(t["T_profile"]))
            rec["q"].append(np.asarray(t["qv_profile"]) * 1000.0)
            rec["Tpri"].append(np.asarray(pr["T_profile"]))
            rec["qpri"].append(np.asarray(pr["qv_profile"]) * 1000.0)
            rec["score"].append(t.get("thermo_score", np.nan))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--arm-suffix", default="f0")
    ap.add_argument("--reference-dir", type=Path, default=Path(
        "/burg-archive/glab/users/pg2328/legoESM/results/rcemip_ref_sam300"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--schemes", nargs="*", default=None,
                    help="subset to plot; default all")
    args = ap.parse_args(argv)

    if "rcemip1_n128" in str(args.reference_dir):
        raise SystemExit("REFUSED: that is our own CRM run, not an oracle.")
    ref = camp.build_reference_profiles(args.reference_dir, 5)
    z = np.asarray(ref.z_m) / 1000.0
    m = z <= PLOT_TOP_KM
    Tref = np.asarray(ref.T_ref)
    qref = np.asarray(ref.qv_ref) * 1000.0

    data = _collect(_seed_dirs(args.arm_dir, args.seeds, args.arm_suffix))
    if args.schemes:
        data = {k: v for k, v in data.items() if k in args.schemes}
    if not data:
        raise SystemExit("no profiles found")
    order = sorted(data, key=lambda k: np.nanmean(data[k]["score"]))

    cmap = plt.get_cmap("turbo")
    colors = {k: cmap(i / max(1, len(order) - 1))
              for i, k in enumerate(order)}

    fig, (axT, axQ) = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    for name in order:
        r = data[name]
        c = colors[name]
        for ax, key in ((axT, "T"), (axQ, "q")):
            arr = np.vstack(r[key])[:, m]
            mean = arr.mean(0)
            lo, hi = arr.min(0), arr.max(0)
            n = len(r[key])
            ax.fill_betweenx(z[m], lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(mean, z[m], color=c, lw=1.8,
                    label=f"{name} (n={n})" if ax is axT else None)

    axT.plot(Tref[m], z[m], "k", lw=2.6, label="SAM CRM", zorder=10)
    axQ.plot(qref[m], z[m], "k", lw=2.6, zorder=10)
    axT.set_xlabel("Temperature [K]")
    axQ.set_xlabel("Water vapour [g/kg]")
    axT.set_ylabel("Height [km]")
    axT.set_ylim(0, PLOT_TOP_KM)
    for ax in (axT, axQ):
        ax.grid(alpha=0.25)
    axT.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
    fig.suptitle("SCM convection tuned to SAM CRM, RCEMIP RCE300 "
                 "(nonrotating protocol)\n"
                 "line = 5-seed mean, band = seed min-max (width = "
                 "parameter non-identifiability)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out} ({len(order)} schemes, {len(args.seeds)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
