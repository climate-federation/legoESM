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
N_MEM = len(SEEDS)
RSE_STD = float(1.0 / np.sqrt(2 * (N_MEM - 1)))   # rel. standard error of a std estimate
KEYS = G.KEYS + ("band", "band_c")
LABELS = dict(G.LABELS,
              band="channel band [Sv] e3t_1d,median",
              band_c="channel band [Sv] e3t_0,mean")
# The 10-year constants the gate currently applies at 90 days.  Imported, not
# copied, so this table cannot drift from the gate.  There is no 10-yr constant
# for the channel band -- it has never had a floor, which is why it is here.
FLOORS_10Y = dict(G.FLOORS, band=None, band_c=None)
# docs/ocean/fidelity/dino_1226_state.md, the #1492 item-2.1 record: the gate's
# constants are the MAX OVER BRANCH-YEARS 1-10 of a 3-member RANGE, and that
# table also prints the matching std maxima.  Recorded here so the std column
# is compared against a std and the range column against a range -- comparing
# across the two statistics is biased ~2x for no physical reason.  That
# ensemble was NEMO PERTURBING NEMO on a 10-year branch; this lane measures
# legoESM on a 90-day branch.  Different model, different horizon.
STD_10Y = {"acc": 0.050, "up": 5.7e-5, "deep": 2.3e-5, "smax": 5.3e-5,
           "smean": 5.3e-5, "band": None, "band_c": None}
SQRT2 = float(np.sqrt(2.0))
# The shipped card's recorded day-90 ACC gap (PHASE2_R6_alignment_and_prereg.md
# arm D, /tmp/dino_valid/q_D_gate.log).  Used ONLY as a fatal control on the
# unperturbed member, never as a target.
RECORDED_ACC_GAP = 0.4107
RECORDED_ACC_TOL = 1e-3


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
        # A reused artifact gets the SAME provenance gate the subprocess would
        # have applied.  kamm_twin_90d.provenance_gate() exists because two
        # runs at byte-identical committed source once differed by 2.5 Sv and
        # nothing had stamped the tree state (#1455 a009c6812); skipping on a
        # bare os.path.exists re-opens that hole from the other side.
        want = head_sha()
        got = member_sha(log)
        if got != want:
            raise SystemExit(
                f"{npz} exists but its log records HEAD={got}, not the current "
                f"HEAD={want}. An ensemble whose members sit at different "
                f"source SHAs measures the SHA as well as the noise. Delete it "
                f"to regenerate, or score it as an explicit --cross-check.")
        print(f"[skip] {npz} exists at the current HEAD", flush=True)
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


def head_sha():
    return subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(_DIR)))), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()


def member_sha(log_path):
    """The HEAD this member's log says it ran at, or None."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as fh:
        for ln in fh:
            if ln.startswith("PROVENANCE: HEAD="):
                return ln.split("HEAD=", 1)[1].split()[0]
    return None


def member_completed(log_path):
    """True only if the harness printed its OWN success line.  An exit code is
    not evidence (the repo's tool-status rule): read the tail."""
    if not os.path.exists(log_path):
        return False
    with open(log_path) as fh:
        return any(ln.startswith("DONE nsteps=") and "STABLE=True" in ln
                   for ln in fh)


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
    longitudes 2..-2 -- acc_full's weighting and reduction, band-restricted.
    This is the PRE-REGISTERED reduction."""
    return float(np.median(A.acc_band(u, wet_u, A.e3t1d)[2:-2]))


def band_transport_campaign(u, wet_u):
    """The reduction the CAMPAIGN has been quoting: real partial-cell e3t_0
    weighting, MEAN over longitudes 2..-2.  Identified by reproducing the
    recorded arm-A band gap of +0.2874 Sv exactly (the pre-registered
    e3t_1d+median reduction gives +0.2791 Sv on the same artifact -- a
    different reduction of the same state, not a discrepancy).  Both are
    reported so a floor measured here can be quoted next to either."""
    return float(np.mean(A.acc_band(u, wet_u)[2:-2]))


# ------------------------------------------------------------------ spread ---
def spread(values):
    """(max pairwise |diff|, sample std) over an ensemble of scalars.

    NaN PROPAGATES in both slots.  The obvious spelling --
    ``max(abs(a - b) for a, b in itertools.combinations(v, 2))`` -- does NOT:
    Python's ``max`` compares with ``>``, every comparison against NaN is
    False, so a NaN anywhere but first is silently discarded and a blown-up
    member yields a plausible finite "floor".  ``np.max`` propagates
    regardless of position.  (Code review of 4e652b101; the repo's own
    "nanmax hides failures" rule, in its subtlest form.)

    Note for the reader, not the code: over n samples the max over all pairs
    is identically the sample RANGE (max - min).  For normal samples at n=4
    E[range] ~ 2.06 sigma, so this statistic runs about twice the std BY
    CONSTRUCTION -- not because it "captures more".
    """
    v = np.asarray(values, dtype=np.float64)
    mx = float(np.max(np.abs(v[:, None] - v[None, :])))
    return mx, float(np.std(v, ddof=1))


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
    # n relative standard error of a std estimate: 1/sqrt(2(n-1)) ~ 0.41 at n=4.
    assert abs(RSE_STD - 0.4082) < 1e-3, RSE_STD
    # NaN MUST propagate into BOTH slots, from ANY position -- the builtin
    # max() spelling this replaced silently returned 3.0 for the first case.
    for vals in ([1.0, 2.0, 4.0, float("nan")], [float("nan"), 1.0, 2.0, 4.0]):
        a, b = spread(vals)
        assert np.isnan(a) and np.isnan(b), (vals, a, b)
    # max-pairwise IS the sample range, on a case where they could differ.
    r = [2.0, -7.0, 0.5, 3.25]
    assert abs(spread(r)[0] - (max(r) - min(r))) < 1e-12
    print("SELF-CHECK OK: spread(max-pairwise == range, std), NaN propagates "
          "from any position, sqrt(2) difference factor, "
          f"n={N_MEM} relative standard error {100 * RSE_STD:.1f}%")
    return 0


# ------------------------------------------------------------------- print ---
def banner():
    print("=" * 104)
    print(f"90-DAY ENSEMBLE NOISE FLOOR -- {N_MEM} members (1 unperturbed "
          "control + 3 x 1e-14 relative T perturbation)")
    print(f"n={N_MEM}: the relative standard error of a std estimate is "
          f"1/sqrt(2(n-1)) = {100 * RSE_STD:.0f}%.  Every floor below is a")
    print("FACTOR-OF-TWO estimate, quoted to 3 digits because that is what the "
          "arithmetic produces.")
    print("The perturbation touches the now-level TEMPERATURE ONLY -- one "
          "direction of a many-dimensional")
    print("tangent space -- so this spread is a LOWER BOUND on the true "
          "single-run floor.  The growth table")
    print("below is the control that says whether the kick reached the metrics "
          "at all.")
    print("=" * 104)


def growth_table(paths):
    """Did the 1e-14 kick actually propagate, or is the tiny spread an artifact
    of a perturbation that never reached the integrated metrics?  This is the
    measurement that discriminates the two, and it must be read before the
    floor table is interpreted.

    PRECISION BOUND, stated because it bounds what may be claimed: the 3-D
    snapshots are stored float32, so on a ~20 K temperature the quantum is
    ~2e-06 K.  The day-0 kick is ~1e-12 K and is therefore INVISIBLE here --
    a 0.0 at day 0 is the storage precision, not the absence of a
    perturbation (the harness's own PERTURB line records the real size).
    Only differences above ~2e-06 K are resolved."""
    print("\n--- GROWTH CONTROL: max|dT| of each perturbed member vs the "
          "control, by snapshot day [K] ---")
    print("    (float32 snapshots: quantum ~2e-06 K on a ~20 K field; day-0 "
          "reads 0 by storage, not by physics)")
    ctrl = np.load(paths[0])
    print(f"{'member':<14}" + "".join(f"{'day ' + str(d):>16}"
                                      for d in (0, 30, 60, 90)))
    for i in range(1, N_MEM):
        d = np.load(paths[i])
        cells = []
        for day in (0, 30, 60, 90):
            key = f"T3d_day{day}"
            if key not in d.files or key not in ctrl.files:
                cells.append(f"{'(absent)':>16}")
                continue
            a = np.asarray(d[key], dtype=np.float64)
            b = np.asarray(ctrl[key], dtype=np.float64)
            v = np.abs(a - b)
            if not np.isfinite(v).all():
                raise SystemExit(f"non-finite T3d_day{day} in {paths[i]} or "
                                 f"the control -- a blown member is a finding")
            cells.append(f"{float(v.max()):>16.4e}")
        print(f"{member_name(i):<14}" + "".join(cells))


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

    # CONTROL 3, mechanized: every member ran to day 90 and said so ITSELF.
    # An exit code is not evidence; the harness's own success line is.
    for i in range(N_MEM):
        lg = f"{args.dir}/{member_name(i)}.log"
        if not member_completed(lg):
            raise SystemExit(
                f"{member_name(i)} has no 'DONE nsteps=... STABLE=True' line in "
                f"{lg}. A member that did not finish is a FINDING, not a member "
                f"to drop -- report it and stop.")
    # CONTROL 3b: one SHA across the whole ensemble.
    shas = {member_name(i): member_sha(f"{args.dir}/{member_name(i)}.log")
            for i in range(N_MEM)}
    if len(set(shas.values())) != 1 or None in shas.values():
        raise SystemExit(f"members sit at different source SHAs: {shas}")

    _self_check()
    print()
    # The gate's own two fatal controls, on the first member's mask.
    first = G.load_candidate(paths[0])
    wet0 = A.tmask & (first["land_mask"] > 0.5)[:, :, None]
    G.instrument_self_checks(wet0)
    nemo = G.load_nemo_day90()
    nemo_m = G.metrics(nemo, wet0)
    nemo_m["band"] = band_transport(nemo["u"], A.umask)
    nemo_m["band_c"] = band_transport_campaign(nemo["u"], A.umask)

    banner()
    print("\n--- provenance of every member (printed, never used to filter) ---")
    rows, masks = {}, {}
    for i, q in enumerate(paths):
        name = member_name(i)
        print(f"\n[{name}]  {q}")
        for ln in log_stamps(f"{args.dir}/{name}.log"):
            print("   ", ln)
        cand = G.load_candidate(q)
        wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]
        m = G.metrics(cand, wet)
        m["band"] = band_transport(cand["u"], A.umask)
        m["band_c"] = band_transport_campaign(cand["u"], A.umask)
        rows[name] = m
        masks[name] = cand["land_mask"]

    # CONTROL: identical wet domain across members.  A silently diverging land
    # mask would not raise -- it would compare members on different oceans.
    ref = masks[member_name(0)]
    for i in range(1, N_MEM):
        if not np.array_equal(masks[member_name(i)], ref):
            raise SystemExit(f"{member_name(i)} has a different land mask from "
                             f"the control -- the members are not comparable")
    print(f"\n[control] land mask identical across all {N_MEM} members: OK")

    # CONTROL 4, mechanized: three DIFFERENT seeds must produce three DIFFERENT
    # day-90 states.  Identical values would mean the perturbation never
    # reached the integrator, and the "floor" would be exactly zero by
    # construction -- the most flattering possible artifact.
    for k in KEYS:
        vals = [rows[member_name(i)][k] for i in range(N_MEM)]
        if not np.all(np.isfinite(vals)):
            raise SystemExit(f"non-finite member value for {k}: {vals}")
        # >= 2 distinct values, NOT N_MEM distinct values.  The strict form
        # was wrong and this control caught the reason: the twin's 3-D
        # snapshots are stored FLOAT32, so a max-type metric (the southern
        # sigma MAX is one cell's value) can read IDENTICAL on two members
        # whose underlying states differ below the storage quantum.  Two
        # members coinciding is quantization; ALL members coinciding would be
        # a perturbation that never reached the integrator.  The growth table
        # is the real control on that, and it runs above.
        if len(set(vals)) < 2:
            raise SystemExit(
                f"metric {k!r} is IDENTICAL across ALL members ({vals}) -- the "
                f"perturbation did not reach it, so its spread is zero by "
                f"construction, not by measurement")
        n_tied = N_MEM - len(set(vals))
        if n_tied:
            print(f"[control] NOTE {k!r}: {n_tied} member value(s) tie at the "
                  f"float32 storage quantum -- its floor is an UPPER bound")
    print(f"[control] every metric separates at least two members: OK")

    # STORAGE-PRECISION BOUND on the whole measurement, printed next to the
    # numbers it bounds.  The 3-D snapshots this scores are float32.  For the
    # transport metrics the accumulated rounding of the stored u field is the
    # same order as the spread being measured, so the floors below are UPPER
    # BOUNDS on the true run-to-run spread -- they cannot be smaller than what
    # the storage can represent.  (Every conclusion in the pre-registration is
    # about the floor being FAR BELOW the gate's constant, and an upper bound
    # is the right side of that inequality -- but the numbers are not to be
    # read as resolved values.)
    _u = np.abs(G.load_candidate(paths[0])["u"])
    _q = float(np.max(_u)) * float(np.finfo(np.float32).eps)
    print(f"[control] float32 storage quantum on u: ~{_q:.2e} m/s "
          f"(max|u|={float(np.max(_u)):.3f}); the ACC floor below is an UPPER "
          f"bound set partly by this, not a resolved value")

    growth_table(paths)

    print("\n--- member values ---")
    print(f"{'metric':<36}" + "".join(f"{member_name(i):>18}" for i in range(len(SEEDS)))
          + f"{'NEMO d90':>18}")
    for k in KEYS:
        print(f"{LABELS[k]:<36}"
              + "".join(f"{rows[member_name(i)][k]:>18.9f}" for i in range(len(SEEDS)))
              + f"{nemo_m[k]:>18.9f}")

    print("\n--- FLOOR TABLE (single-run floor at 90 days) ---")
    print("the gate's constants are RANGE statistics (max over branch-years "
          "1-10 of a 3-member range),")
    print("measured on NEMO perturbing NEMO over TEN YEARS.  Range is compared "
          "against range and std against")
    print("std; comparing across the two is biased ~2x for no physical reason.")
    print(f"{'metric':<36}{'10yr xfer range':>17}{'10yr xfer std':>15}"
          f"{'90d max-pairwise':>18}{'90d std':>13}"
          f"{'range ratio':>13}{'std ratio':>12}")
    floors = {}
    for k in KEYS:
        vals = [rows[member_name(i)][k] for i in range(len(SEEDS))]
        mx, sd = spread(vals)
        floors[k] = (mx, sd)
        ten, tsd = FLOORS_10Y[k], STD_10Y[k]
        rr = "n/a" if ten in (None, 0) else f"{mx / ten:.3e}"
        sr = "n/a" if tsd in (None, 0) else f"{sd / tsd:.3e}"
        print(f"{LABELS[k]:<36}"
              f"{('none' if ten is None else f'{ten:.4g}'):>17}"
              f"{('none' if tsd is None else f'{tsd:.4g}'):>15}"
              f"{mx:>18.4e}{sd:>13.4e}{rr:>13}{sr:>12}")

    # CONTROL 2, mechanized: the unperturbed control must reproduce the
    # recorded shipped-card ACC gap.  If it does not, the intervening commits
    # moved the trajectory and THAT is the finding.
    acc_gap0 = abs(rows["m0_control"]["acc"] - nemo_m["acc"])
    print(f"\n[control] m0 ACC gap {acc_gap0:.6f} Sv vs recorded shipped-card "
          f"{RECORDED_ACC_GAP:.6f} Sv "
          f"(|delta| {abs(acc_gap0 - RECORDED_ACC_GAP):.2e})")
    if abs(acc_gap0 - RECORDED_ACC_GAP) > RECORDED_ACC_TOL:
        raise SystemExit(
            f"the unperturbed control does NOT reproduce the recorded shipped-"
            f"card ACC gap ({acc_gap0:.6f} vs {RECORDED_ACC_GAP:.6f}). The "
            f"trajectory moved; measure THAT before measuring a floor.")

    print("\n--- GAPS vs NEMO day 90, scored against BOTH floors ---")
    print("difference-floor rule: a gap is a difference of two runs.  sqrt(2) x "
          "floor is the special case")
    print("where both runs have the SAME variance; the honest combination is "
          "sqrt(floor_lego^2 + floor_nemo^2),")
    print("and floor_nemo has only ever been measured at TEN YEARS -- so every "
          "ratio below inherits a transfer.")
    print(f"{'metric':<36}{'|gap|':>14}{'gap/(v2*10yr)':>16}"
          f"{'gap/(v2*maxpair)':>18}{'gap/(v2*std)':>15}")
    _fmt_ten = lambda r: "at floor" if r < 2.0 else f"{r:.2f}"   # noqa: E731
    for k in KEYS:
        gap = abs(rows["m0_control"][k] - nemo_m[k])
        mx, sd = floors[k]
        ten = FLOORS_10Y[k]
        c1 = "n/a" if ten in (None, 0) else _fmt_ten(gap / (SQRT2 * ten))
        # "at floor" covers the prereg's "within a factor of ~1" band, not a
        # hard < 1.0 -- a ratio of 1.05 is not a resolved difference.
        _fmt = lambda r: "at floor" if r < 2.0 else f"{r:.3e}"   # noqa: E731
        c2 = "at floor" if mx == 0 else _fmt(gap / (SQRT2 * mx))
        c3 = "at floor" if sd == 0 else _fmt(gap / (SQRT2 * sd))
        print(f"{LABELS[k]:<36}{gap:>14.6e}{c1:>16}{c2:>18}{c3:>15}")

    if args.cross_check:
        print(f"\n--- CROSS-CHECK, scored alongside and NOT in the spread: "
              f"{args.cross_check} ---")
        cand = G.load_candidate(args.cross_check)
        wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]
        m = G.metrics(cand, wet)
        m["band"] = band_transport(cand["u"], A.umask)
        m["band_c"] = band_transport_campaign(cand["u"], A.umask)
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
