"""#1492/#1455: the TRUE 90-DAY NOISE FLOOR of the acceptance-gate metrics.

Pre-registration: ``PREREG_floor90_ensemble.md`` (committed before any member
ran; read it before citing any number this prints).

WHY.  ``acceptance_gate_90d.py`` scores five metrics against floor constants
measured on TEN-YEAR branches (#1492 item 2.1) and applies them to NINETY-DAY
twins.  The transfer across horizons was never validated.  This runner measures
the floor at the horizon the gate is actually applied at, by running the SHIPPED
card four times -- once unperturbed and three times with a 1e-14-relative
temperature perturbation -- and reporting the ensemble spread of every gate
metric plus the channel-band transport.

WHAT IT IS NOT.  It does not edit ``FLOORS``; those constants are
owner-controlled.  It prints tables and never a verdict (skill Rule: a probe
that prints its own conclusion gets that conclusion echoed back as evidence).

ARITHMETIC DISCIPLINE, applied in the printed table itself so it cannot be
dropped downstream:

  * the spread measured here is a SINGLE-RUN floor.  A legoESM-minus-NEMO gap
    is a DIFFERENCE of two runs, so the column that scores gaps divides by
    ``sqrt(2) x floor``, and the column header says so.
  * the 10-year constants are labelled "10-yr xfer" wherever they appear next
    to a 90-day number.
  * n=4 gives a ~41% relative standard error on a standard deviation
    (1/sqrt(2(n-1))); the banner repeats it and both spread statistics
    (max-pairwise AND std) are always printed together.

METRICS.  The five gate metrics are IMPORTED from ``acceptance_gate_90d``
(``metrics()``), not re-derived, so the floor is measured on exactly the
quantity the gate scores.  The sixth is
``acc_thermal_wind.acc_band(u, wet, e3=e3t_1d)`` reduced by the median over
longitudes 2..-2 -- ``acc_full``'s own weighting and reduction, restricted to
the re-entrant channel band.

Usage
-----
  # run the four members (skips any whose npz already exists), then score:
  floor90_ensemble.py --dir /tmp/dino_floor90 --run

  # score members already on disk:
  floor90_ensemble.py --dir /tmp/dino_floor90

  # also score an extra artifact WITHOUT mixing it into the spread
  # (e.g. the promotion lane's arm-D twin, produced at an older SHA):
  floor90_ensemble.py --dir /tmp/dino_floor90 --cross-check /tmp/dino_valid/q_D.npz

  # runnable self-check of the spread arithmetic (no twins needed):
  floor90_ensemble.py --self-check

Exit status is 0 on a completed measurement and non-zero only on a harness
failure -- there is no pass/fail here to encode.
"""
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A            # noqa: E402
import acceptance_gate_90d as G         # noqa: E402

SEEDS = (None, 1, 2, 3)                 # member 0 is the unperturbed control
KEYS = G.KEYS + ("band",)
LABELS = dict(G.LABELS, band="channel-band transport [Sv]")
# The 10-year constants the gate currently applies at 90 days.  Imported, not
# copied, so this table cannot drift from the gate.  There is no 10-yr constant
# for the channel band -- it has never had a floor, which is why it is here.
FLOORS_10Y = dict(G.FLOORS, band=None)
SQRT2 = float(np.sqrt(2.0))


def member_name(i):
    return "m0_control" if SEEDS[i] is None else f"m{i}_seed{SEEDS[i]}"


# --------------------------------------------------------------------- run ---
def run_member(i, out_dir, gpu="0"):
    """Launch one 90-day twin on the SHIPPED card.  No option flags: the card's
    own defaults are the thing being measured, and passing a flag that happens
    to equal the default still changes what the log records was asked for."""
    npz = f"{out_dir}/{member_name(i)}.npz"
    log = f"{out_dir}/{member_name(i)}.log"
    if os.path.exists(npz):
        print(f"[skip] {npz} exists", flush=True)
        return npz
    cmd = [sys.executable, f"{_DIR}/run_fp64.py", f"{_DIR}/kamm_twin_90d.py",
           "nemo_dino_kamm_mlf", npz, "--days", "90", "--bridge-before",
           "--save-3d"]
    if SEEDS[i] is not None:
        cmd += ["--perturb-seed", str(SEEDS[i])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, JAX_ENABLE_X64="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    print("RUN:", " ".join(cmd), f"[CUDA_VISIBLE_DEVICES={gpu}]", flush=True)
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return npz


def log_stamps(path):
    """The lines that say WHAT was run.  Reported, never used to filter: an
    unexpected stamp is a finding to print, not a member to silently drop."""
    want = ("PROVENANCE:", "vertical ladder", "PRECISION:", "seasonal clock",
            "PERTURB seed=", "ABLATION:", "DONE nsteps=")
    if not os.path.exists(path):
        return ["(no log)"]
    with open(path) as fh:
        return [ln.rstrip() for ln in fh if ln.startswith(want)
                or ln.lstrip().startswith(want)]


# ----------------------------------------------------------------- metrics ---
def band_transport(u, wet_u):
    """Channel-band zonal transport [Sv], e3t_1d weighting, median over
    longitudes 2..-2 -- acc_full's weighting and reduction, band-restricted."""
    return float(np.median(A.acc_band(u, wet_u, A.e3t1d)[2:-2]))


# ------------------------------------------------------------------ spread ---
def spread(values):
    """(max pairwise |diff|, sample std) over an ensemble of scalars."""
    v = np.asarray(values, dtype=np.float64)
    mx = max(abs(a - b) for a, b in itertools.combinations(v, 2))
    return float(mx), float(np.std(v, ddof=1))


def _self_check():
    """Runnable check of the spread arithmetic and of the difference-floor
    factor -- the two pieces of this probe that are not imported from a
    recorded harness.  Fails loudly if either is wrong."""
    mx, sd = spread([1.0, 2.0, 4.0, 8.0])
    assert abs(mx - 7.0) < 1e-12, mx
    assert abs(sd - float(np.std([1.0, 2.0, 4.0, 8.0], ddof=1))) < 1e-12, sd
    # A constant ensemble has zero spread, and the pairwise max must not be
    # confused with the range of a sorted list by an off-by-one.
    assert spread([3.0] * 4) == (0.0, 0.0)
    mx2, _ = spread([-5.0, 0.0, 0.0, 5.0])
    assert abs(mx2 - 10.0) < 1e-12, mx2
    # The difference floor is sqrt(2) x the single-run floor, not 2x and not 1x.
    assert abs(SQRT2 - 1.41421356) < 1e-6
    # n=4 relative standard error of a std estimate: 1/sqrt(2(n-1)) ~ 0.41.
    assert abs(1.0 / np.sqrt(2 * (4 - 1)) - 0.4082) < 1e-3
    print("SELF-CHECK OK: spread(max-pairwise, std), sqrt(2) difference "
          "factor, n=4 relative standard error 40.8%")
    return 0


# ------------------------------------------------------------------- print ---
def banner():
    print("=" * 100)
    print("90-DAY ENSEMBLE NOISE FLOOR -- 4 members (1 control + 3 x 1e-14 "
          "relative T perturbation)")
    print("n=4: the relative standard error of a std estimate is 1/sqrt(2(n-1)) "
          "= 41%.  Every floor below is")
    print("a FACTOR-OF-TWO estimate quoted to 3 digits because that is what the "
          "arithmetic produces.")
    print("=" * 100)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default="/tmp/dino_floor90",
                   help="directory for member npz/log artifacts")
    p.add_argument("--run", action="store_true",
                   help="launch any member whose npz is missing (sequential)")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for --run")
    p.add_argument("--cross-check", default=None,
                   help="extra artifact scored ALONGSIDE and never mixed into "
                        "the spread (e.g. a twin produced at an older SHA)")
    p.add_argument("--self-check", action="store_true",
                   help="arithmetic self-check only; no twins needed")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()

    os.makedirs(args.dir, exist_ok=True)
    paths = []
    for i in range(len(SEEDS)):
        paths.append(run_member(i, args.dir, args.gpu) if args.run
                     else f"{args.dir}/{member_name(i)}.npz")
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        raise SystemExit("missing members (run with --run): " + ", ".join(missing))

    _self_check()
    print()
    # The gate's own two fatal controls, on the first member's mask.
    first = G.load_candidate(paths[0])
    wet0 = A.tmask & (first["land_mask"] > 0.5)[:, :, None]
    G.instrument_self_checks(wet0)
    nemo = G.load_nemo_day90()
    nemo_m = G.metrics(nemo, wet0)
    nemo_m["band"] = band_transport(nemo["u"], A.umask)

    banner()
    print("\n--- provenance of every member (printed, never used to filter) ---")
    rows = {}
    for i, q in enumerate(paths):
        name = member_name(i)
        print(f"\n[{name}]  {q}")
        for ln in log_stamps(f"{args.dir}/{name}.log"):
            print("   ", ln)
        cand = G.load_candidate(q)
        wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]
        m = G.metrics(cand, wet)
        m["band"] = band_transport(cand["u"], A.umask)
        rows[name] = m

    print("\n--- member values ---")
    print(f"{'metric':<36}" + "".join(f"{member_name(i):>18}" for i in range(len(SEEDS)))
          + f"{'NEMO d90':>18}")
    for k in KEYS:
        print(f"{LABELS[k]:<36}"
              + "".join(f"{rows[member_name(i)][k]:>18.9f}" for i in range(len(SEEDS)))
              + f"{nemo_m[k]:>18.9f}")

    print("\n--- FLOOR TABLE (single-run floor at 90 days) ---")
    print(f"{'metric':<36}{'10-yr xfer floor':>20}{'90d max-pairwise':>20}"
          f"{'90d std (n=4)':>18}{'maxpair/10yr':>14}")
    floors = {}
    for k in KEYS:
        vals = [rows[member_name(i)][k] for i in range(len(SEEDS))]
        mx, sd = spread(vals)
        floors[k] = (mx, sd)
        ten = FLOORS_10Y[k]
        ratio = "n/a" if ten in (None, 0) else f"{mx / ten:.3f}"
        tenstr = "none" if ten is None else f"{ten:.4g}"
        print(f"{LABELS[k]:<36}{tenstr:>20}{mx:>20.4e}{sd:>18.4e}{ratio:>14}")

    print("\n--- GAPS vs NEMO day 90, scored against BOTH floors ---")
    print("difference-floor rule: a gap is a difference of two runs, so it is "
          "divided by sqrt(2) x floor.")
    print(f"{'metric':<36}{'|gap|':>14}{'gap/(v2*10yr)':>16}"
          f"{'gap/(v2*maxpair)':>18}{'gap/(v2*std)':>15}")
    for k in KEYS:
        gap = abs(rows["m0_control"][k] - nemo_m[k])
        mx, sd = floors[k]
        ten = FLOORS_10Y[k]
        c1 = "n/a" if ten in (None, 0) else f"{gap / (SQRT2 * ten):.2f}"
        c2 = "at floor" if mx == 0 or gap / (SQRT2 * mx) < 1.0 else f"{gap / (SQRT2 * mx):.2f}"
        c3 = "at floor" if sd == 0 or gap / (SQRT2 * sd) < 1.0 else f"{gap / (SQRT2 * sd):.2f}"
        print(f"{LABELS[k]:<36}{gap:>14.6e}{c1:>16}{c2:>18}{c3:>15}")

    if args.cross_check:
        print(f"\n--- CROSS-CHECK, scored alongside and NOT in the spread: "
              f"{args.cross_check} ---")
        cand = G.load_candidate(args.cross_check)
        wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]
        m = G.metrics(cand, wet)
        m["band"] = band_transport(cand["u"], A.umask)
        print(f"{'metric':<36}{'cross-check':>18}{'m0_control':>18}"
              f"{'|diff|':>14}{'diff/maxpair':>14}")
        for k in KEYS:
            d = abs(m[k] - rows["m0_control"][k])
            mx = floors[k][0]
            r = "n/a" if mx == 0 else f"{d / mx:.2f}"
            print(f"{LABELS[k]:<36}{m[k]:>18.9f}{rows['m0_control'][k]:>18.9f}"
                  f"{d:>14.4e}{r:>14}")

    print("\n(no verdict is printed here by design -- see "
          "PREREG_floor90_ensemble.md for the pre-registered decision rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
