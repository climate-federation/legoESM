"""#1492: NEMO's OWN 90-day noise floor -- the other half of the comparison
``floor90_ensemble.py`` (legoESM side) started.

WHY.  ``floor90_ensemble.py`` measured legoESM's 90-day reproducibility floor
by perturbing the bridged NEMO restart and re-running the legoESM twin three
times.  That is only half a two-model comparison: it says nothing about how
much NEMO ITSELF wobbles run-to-run over the same 90 days.  This probe
measures that other half, on the SAME gate metrics, with the SAME spread
arithmetic -- imported from ``floor90_ensemble.py``, not re-derived -- so the
two floors can be combined honestly.

MEMBERS (n=4, matching the legoESM ensemble's own N_MEM and seed convention
exactly -- ``floor90_ensemble.SEEDS = (None, 1, 2, 3)``), all four on the
SAME namelist, SAME source restart, and the SAME certified NEMO binary
(``RUN_90D_TWIN`` was NOT reused as the control: it ran on 2026-07-23 on a
``./nemo`` symlink that has since been overwritten by two rebuilds, so its
exact binary is unrecoverable and not provably identical to the certified
one -- a same-binary control is cheap, so one was built instead of trusting
that):
  * m0_control : ``RUN_FLOOR90_M0`` -- unperturbed, same restart, certified binary.
  * m1/m2/m3   : ``RUN_FLOOR90_M{1,2,3}`` -- the SAME namelist and restart,
    with ``tn`` perturbed by seed 1/2/3 via ``perturb_nemo_tn_90d.py``
    (committed sibling tool; read its module docstring for exactly how this
    matches, and where it necessarily diverges from, the legoESM-side
    perturbation).

METRICS: the five gate metrics (``acceptance_gate_90d.metrics``) plus the two
channel-band reductions (``floor90_ensemble.band_transport`` /
``band_transport_campaign``) -- identical KEYS, LABELS, and ``spread()``
arithmetic to the legoESM-side probe, imported.

WHAT IT IS NOT.  Prints tables, never a verdict -- same discipline as
``floor90_ensemble.py``.

Usage
-----
  floor90_nemo_ensemble.py                    # score members already on disk
  floor90_nemo_ensemble.py --self-check       # arithmetic self-check only
"""
import argparse
import glob
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A            # noqa: E402
import acceptance_gate_90d as G         # noqa: E402
import floor90_ensemble as F            # noqa: E402  (spread(), KEYS, LABELS -- imported, not re-derived)

DINO = A.DINO
MEMBER_SEEDS = (None, 1, 2, 3)           # member 0 = unperturbed, certified-binary control
N_MEM = len(MEMBER_SEEDS)
KEYS = F.KEYS
LABELS = F.LABELS

# The legoESM-side 90-day floor, as PRINTED by floor90_ensemble.py scoring
# /tmp/dino_floor90 at this branch's HEAD (re-run live, not copied from a
# commit message -- prose is a pointer, not a citable number). (max-pairwise,
# std), keyed identically to KEYS.
LEGO_FLOOR_90D = {
    "acc":    (1.1465e-05, 5.4014e-06),
    "up":     (4.9987e-08, 2.5627e-08),
    "deep":   (2.9036e-09, 1.3301e-09),
    "smax":   (1.5207e-07, 7.6034e-08),
    "smean":  (1.5324e-07, 7.4587e-08),
    "band":   (2.1800e-05, 1.0167e-05),
    "band_c": (1.7782e-05, 7.6863e-06),
}


def member_dir(i):
    # RUN_FLOOR90_M0: unperturbed, SAME namelist/restart as m1-m3, run on the
    # SAME certified binary (review finding: RUN_90D_TWIN's own './nemo' was
    # overwritten by rebuilds since it ran on 2026-07-23 -- its exact binary
    # is unrecoverable and not provably identical to the certified one m1-m3
    # use, which is itself a confound this dedicated control removes).
    return f"{DINO}/RUN_FLOOR90_M{i}"


def member_name(i):
    return "m0_control" if MEMBER_SEEDS[i] is None else f"m{i}_seed{MEMBER_SEEDS[i]}"


# --------------------------------------------------------------- completion ---
def member_completed(run_dir):
    """Mirrors the repo's own rule: an exit code is not evidence, read the
    harness's own success line. NEMO's success line is 'STOP 0' in the
    captured stdout/stderr log; a blow-up prints 'E R R O R' in ocean.output
    (nemogcm.F90 ctl_stop convention) instead."""
    ts_path = f"{run_dir}/time.step"
    if not os.path.exists(ts_path):
        return False, "no time.step"
    with open(ts_path) as fh:
        ts = int(fh.read().strip())
    if ts != G.KT_DAY90:
        return False, f"time.step={ts}, expected {G.KT_DAY90}"
    oo_path = f"{run_dir}/ocean.output"
    if os.path.exists(oo_path):
        with open(oo_path, errors="replace") as fh:
            if "E R R O R" in fh.read():
                return False, "E R R O R present in ocean.output"
    logs = glob.glob(f"{run_dir}/run_floor90_m*.log") + glob.glob(f"{run_dir}/run_90d_twin.log")
    if not logs:
        return False, "no run log found"
    with open(logs[0], errors="replace") as fh:
        tail = fh.read()[-200:]
    if "STOP 0" not in tail:
        return False, f"no 'STOP 0' in {logs[0]} tail: {tail!r}"
    return True, "OK"


# ------------------------------------------------------------------- growth ---
def growth_table():
    """Same purpose as floor90_ensemble.growth_table(): confirm the 1e-14 kick
    actually reaches the integrated state rather than being an artifact of a
    perturbation that never propagated. NEMO dumps intermediate restarts every
    10 days (nn_stock=320 -> kt 6080,6400,...,8640); reused here rather than
    adding any new NEMO output."""
    print("\n--- GROWTH CONTROL: max|dT| of each perturbed member vs the "
          "control's OWN restart dump, by kt (day) ---")
    days = (10, 30, 60, 90)
    kts = {d: G.KT_RESTART + d * G.STEPS_PER_DAY for d in days}
    print(f"{'member':<14}" + "".join(f"{'day ' + str(d):>16}" for d in days))
    from rebuild_nemo_restart import rebuild
    for i in range(1, N_MEM):
        cells = []
        for d in days:
            kt = kts[d]
            ctrl_pat = f"{member_dir(0)}/DINO_{kt:08d}_restart_*.nc"
            mem_pat = f"{member_dir(i)}/DINO_{kt:08d}_restart_*.nc"
            if not (glob.glob(ctrl_pat) and glob.glob(mem_pat)):
                cells.append(f"{'(absent)':>16}")
                continue
            a = np.asarray(rebuild(mem_pat, ["tn"])["tn"], dtype=np.float64)
            b = np.asarray(rebuild(ctrl_pat, ["tn"])["tn"], dtype=np.float64)
            v = np.abs(a - b)
            if not np.isfinite(v).all():
                raise SystemExit(f"non-finite tn at kt={kt} in {mem_pat} or "
                                 f"{ctrl_pat} -- a blown member is a finding")
            cells.append(f"{float(v.max()):>16.4e}")
        print(f"{member_name(i):<14}" + "".join(cells))


def _self_check():
    return F._self_check()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    if args.self_check:
        return _self_check()

    print("=" * 104)
    print(f"NEMO's OWN 90-DAY ENSEMBLE NOISE FLOOR -- {N_MEM} members "
          "(1 unperturbed certified-binary control + 3 x 1e-14 "
          "relative tn perturbation)")
    print("=" * 104)

    # CONTROL: every member finished, judged by NEMO's own success line, not
    # an exit code.
    dirs = {}
    for i in range(N_MEM):
        d = member_dir(i)
        ok, why = member_completed(d)
        print(f"[{member_name(i)}] {d}: {'COMPLETE' if ok else 'INCOMPLETE'} ({why})")
        if not ok:
            raise SystemExit(f"{member_name(i)} did not complete: {why}. "
                             f"A member that did not finish is a FINDING, "
                             f"not a member to drop -- report it and stop.")
        dirs[i] = d

    # CONTROL: same certified binary for every member (the control's own
    # './nemo' predates this probe; the perturbed members' symlinks are
    # checked against the certified path named in the mission).
    cert = f"{DINO}/BLD/bin/nemo.exe.certified_d3cf9242"
    for i in range(N_MEM):
        link = f"{dirs[i]}/nemo"
        real = os.path.realpath(link)
        if real != os.path.realpath(cert):
            raise SystemExit(f"{member_name(i)}'s nemo symlink resolves to "
                             f"{real}, not the certified binary {cert}")
    print(f"[control] all {N_MEM} members (control included) ran the "
          f"certified binary {cert}: OK")

    _self_check()
    print()
    wet = A.tmask   # NEMO's own full mask, both sides -- acceptance_gate_90d.self_test() convention
    G.instrument_self_checks(wet)

    rows = {}
    for i in range(N_MEM):
        st = G.load_nemo_day90(run_dir=dirs[i], kt=G.KT_DAY90)
        m = G.metrics(st, wet)
        m["band"] = F.band_transport(st["u"], A.umask)
        m["band_c"] = F.band_transport_campaign(st["u"], A.umask)
        rows[member_name(i)] = m

    # CONTROL: at least two of four members separate on every metric --
    # identical values across the whole ensemble would mean the perturbation
    # never reached the integrator (floor exactly zero by construction).
    for k in KEYS:
        vals = [rows[member_name(i)][k] for i in range(N_MEM)]
        if not np.all(np.isfinite(vals)):
            raise SystemExit(f"non-finite member value for {k}: {vals}")
        if len(set(vals)) < 2:
            raise SystemExit(
                f"metric {k!r} is IDENTICAL across ALL members ({vals}) -- "
                f"the perturbation did not reach it")
    print("[control] every metric separates at least two members: OK")

    growth_table()

    print("\n--- member values ---")
    print(f"{'metric':<36}" + "".join(f"{member_name(i):>18}" for i in range(N_MEM)))
    for k in KEYS:
        print(f"{LABELS[k]:<36}"
              + "".join(f"{rows[member_name(i)][k]:>18.9f}" for i in range(N_MEM)))

    print("\n--- TWO-SIDED FLOOR TABLE (single-run floor at 90 days) ---")
    print(f"{'metric':<36}{'lego maxpair':>14}{'lego std':>13}"
          f"{'NEMO maxpair':>14}{'NEMO std':>13}{'ratio(max)':>12}{'ratio(std)':>12}"
          f"{'RSS maxpair':>13}{'RSS std':>12}")
    combined = {}
    for k in KEYS:
        vals = [rows[member_name(i)][k] for i in range(N_MEM)]
        nmx, nsd = F.spread(vals)
        lmx, lsd = LEGO_FLOOR_90D[k]
        rmx = "n/a" if lmx == 0 else f"{nmx / lmx:.2f}"
        rsd = "n/a" if lsd == 0 else f"{nsd / lsd:.2f}"
        rss_mx = float(np.sqrt(lmx ** 2 + nmx ** 2))
        rss_sd = float(np.sqrt(lsd ** 2 + nsd ** 2))
        combined[k] = (rss_mx, rss_sd)
        print(f"{LABELS[k]:<36}{lmx:>14.4e}{lsd:>13.4e}{nmx:>14.4e}{nsd:>13.4e}"
              f"{rmx:>12}{rsd:>12}{rss_mx:>13.4e}{rss_sd:>12.4e}")

    print("\n(no verdict is printed here by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
