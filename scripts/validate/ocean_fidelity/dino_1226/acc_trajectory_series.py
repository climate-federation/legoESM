#!/usr/bin/env python
"""#1455 §D: the ACC ERROR AS A FUNCTION OF TIME over the 90-day twin, all arms.

Nine faithful per-step fixes left the day-90 ACC error unmoved (1.557 -> 1.708
-> 1.740 Sv). If the anomaly is trajectory-statistical, its TIME SHAPE is the
discriminator no single-step probe can supply:
  - early jump then flat  -> handshake / initialisation (integrator memory)
  - ~linear growth        -> an integrated per-step bias below every probe floor
  - flat then late onset  -> a slow mode diverging

Reuses the acceptance gate's own loaders and the recorded ACC metric
(``acc_thermal_wind.acc_full``: full-section zonal transport, e3t_1d weights,
median over lons 2..-2) VERBATIM -- nothing re-derived. NEMO side: the
RUN_90D_TWIN 10-day restarts (kt 5760..8640, step 320); legoESM side: the
--save-3d snapshots at days 0/30/60/90 (fp64-cast). Only matched days are
compared. Days with no snapshot on either side print '--', never interpolated.

Controls: (1) day-0 lego == NEMO for every arm (bit-identical bridge, so the
series MUST start at 0.000); (2) day-90 must reproduce the gate's recorded
1.557/1.708/1.740 for arms 1/2/3 -- otherwise this instrument is not the gate's.
"""
import glob, os, sys
import numpy as np
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR); sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A                       # noqa: E402
import acceptance_gate_90d as G                    # noqa: E402
from rebuild_nemo_restart import rebuild           # noqa: E402

RECORDED_D90 = {"arm1_pre": 1.557, "arm2_fixes": 1.708, "arm3_bn2": 1.740}
LEGO_DAYS = (0, 30, 60, 90)

def nemo_acc(day):
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    files = glob.glob(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart_*.nc")
    if not files: return None
    un = np.moveaxis(rebuild(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart_*.nc", ["un"])["un"], 0, -1)
    return A.acc_full(un, A.umask)

def lego_acc(npz, day):
    try:    return A.acc_full(G.load_candidate(npz, day)["u"], A.umask)
    except SystemExit: return None

def main(arms):
    days = list(range(0, 91, 10))
    nemo = {d: nemo_acc(d) for d in days}
    print("NEMO ACC [Sv] by day:", {d: (f"{v:.3f}" if v is not None else "--") for d, v in nemo.items()})
    print()
    hdr = "day   NEMO   " + "   ".join(f"{os.path.basename(a).replace('.npz',''):>12s}" for a in arms)
    print(hdr); print("-" * len(hdr))
    series = {a: {} for a in arms}
    for d in days:
        row = f"{d:3d}  {nemo[d]:6.2f}  " if nemo[d] is not None else f"{d:3d}    --   "
        for a in arms:
            la = lego_acc(a, d) if d in LEGO_DAYS else None
            if la is None or nemo[d] is None: row += f"   {'--':>12s}"
            else:
                diff = la - nemo[d]; series[a][d] = diff
                row += f"   {diff:+12.3f}"
        print(row)
    print("\n(lego - NEMO, Sv; sign kept -- '+' means lego ACC ABOVE NEMO)")
    # controls
    print("\n=== CONTROLS ===")
    for a in arms:
        name = os.path.basename(a).replace(".npz", "")
        d0, d90 = series[a].get(0), series[a].get(90)
        c1 = "PASS" if d0 is not None and abs(d0) < 1e-6 else f"FAIL ({d0})"
        rec = RECORDED_D90.get(name)
        c2 = ("PASS" if rec is not None and d90 is not None and abs(abs(d90) - rec) < 2e-3
              else f"CHECK (|d90|={abs(d90) if d90 is not None else None} vs recorded {rec})")
        print(f"  {name:12s} day-0 == 0: {c1:22s} day-90 reproduces gate: {c2}")
    # shape
    print("\n=== SHAPE (fraction of the day-90 error present at each snapshot) ===")
    for a in arms:
        name = os.path.basename(a).replace(".npz", ""); s = series[a]
        if 90 in s and s[90] != 0:
            print(f"  {name:12s} " + "  ".join(f"d{d}:{s[d]/s[90]:+.2f}" for d in LEGO_DAYS if d in s))

if __name__ == "__main__":
    main(sys.argv[1:])
