"""The barotropic loop's STATE-CONSTANT end-wall bias: what it looks like, and
whether the wind-stress placement difference can be its owner.

WHY THIS EXISTS.  `baro_deposit_wall_nyquist.py` established that ~95 % of the
end-wall barotropic residual is a field that is the SAME AT EVERY STATE, and
that the two-step projection annihilates such a field exactly.  That probe could
therefore say only what it could not see.  This one looks at the constant field
DIRECTLY -- no projector anywhere in it -- from the five consecutive deposit
maps already on disk, and then asks whether the one unstarted candidate on the
card (the wind-stress placement difference) has the right shape and size to own
it.

WHAT IT MEASURES, and the arm it measures it in.  The deposit maps carry two
arms per component:

    dU_avg / dV_avg   legoESM's own forcing   (total residual)
    dU_sub / dV_sub   NEMO's frozen zu_frc/zv_frc/ssh_frc substituted into
                      legoESM's own loop, from a bit-identical entry state

The SECOND is the subject here.  In that arm both models integrate the same
forcing from the same state over the same 68 substeps with the same weights, so
whatever survives is the LOOP's, and -- this is the load-bearing consequence for
section 4 -- every difference in how legoESM ASSEMBLES its slow forcing has been
substituted away, wind stress included.

THE CONTROLS, each of which exists because a review found the failure it
prevents (see the retraction ledger in `dino_wall_deposit_nyquist.md`):

  * RIGHT COMPONENT AT THE WALL.  The scored walls are ZONAL walls, so the
    wall-NORMAL component is the MERIDIONAL one.  Every statistic here is
    computed and labelled per component; an earlier probe measured the
    tangential component and called it an exclusion.
  * AMPLITUDE AGREEMENT IS NOT A MECHANISM.  Section 3 computes, for every row
    pair, both the RMS-amplitude ratio AND the per-CELL regression slope and
    correlation of the bias on the local velocity, and FLAGS any row where the
    amplitudes agree while the cells do not.  This control fired on its first
    run: the two wall rows' amplitude ratios (11.73 vs 11.67) agree to 0.5 %
    while the per-cell correlation at the northern wall is 0.05.  Without it a
    proportionality would have been reported as a finding.
  * TWO DELIBERATE RATIOS-OF-RATIOS, BOTH NAMED.  Every reported quantity is
    bare, with its denominator named, except (a) the amplitude-coincidence
    trigger in `mirrored_pairs`, which asks whether the bias's north/south
    amplitude ratio equals the velocity's, and (b) the per-row
    measured-over-predicted ratio in `vface_shape_test`.  Both are
    dimensionless proportionality TESTS rather than reported results, and both
    print their operands bare alongside.  (The header claimed only ONE until
    round-2 review pointed out section 6 had quietly added the second -- the
    same class of scope word this file has now got wrong twice.)  What is
    banned, and absent, is a HEADLINE double ratio: a previous version's
    factorised into a wall ratio over an interior ratio and moved 1.7x on a
    basin-wide constant with nothing to do with walls.
  * NO BAND SUMS.  Row profiles are per row over the full grid; the wall bands
    are reported alongside the interior they are being contrasted with, never
    as a lone aggregate.
  * THE ESTIMATOR CANNOT CANCEL A FIXED FIELD, because there is no estimator --
    the state mean IS the quantity.  Section 1 nonetheless proves the constancy
    rather than assuming it, per state.

Reads only data already on disk.  Runs no model.

Artifact: results/dino_1455/baro_fixed_bias_wall_map.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

CONSECUTIVE_KTS = (5760, 5761, 5762, 5763, 5764)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "..", ".."))
DEFAULT_MAP_DIR = os.path.join(REPO, "results", "dino_1455", "maps_consecutive")
DEFAULT_SEQDUMP = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_SEQDUMP_D180_1R")

# NEMO's haloless dump frame for this configuration: (jpj, jpi).
JPJ, JPI = 199, 52

# The loop's own reply to a steady forcing perturbation, and the weighted-mean
# substep index, both MEASURED by substep_traj_compare.py on this same state and
# quoted here only to convert a forcing into a deposit for section 4's size
# bound.  They are CLI-overridable and printed with their provenance, and
# section 4's verdict is deliberately built so that it does NOT depend on them:
# the bound is reported at the measured response AND at the assumption-free
# response of 1.0, and the geometry test that decides the section uses neither.
MEASURED_RESPONSE = 0.1484        # substep_d5760.log, "R_deposit"
MEASURED_DT_S = 117.391304        # substep_d5760.log, "dt_s used inside loop"


def provenance() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=REPO, text=True).strip()
    except Exception:
        sha, dirty = "unknown", ""
    return (f"[provenance baro_fixed_bias_wall_map] git={sha}"
            f"{'+dirty' if dirty else ''} maps={CONSECUTIVE_KTS}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_maps(map_dir: str, kts=CONSECUTIVE_KTS) -> dict:
    """Load the consecutive deposit maps, refusing any hole in the series.

    A missing state is fatal rather than a dropped sample: the state MEAN is
    the quantity this probe reports, so a gap silently re-weights it.
    """
    keys = ("dU_avg", "dU_sub", "wetu", "nemo_Ubar_avg",
            "dV_avg", "dV_sub", "wetv", "nemo_Vbar_avg", "wgt_primary")
    meta = ("seqdump", "provenance")
    fields, stamps = {}, []
    for kt in kts:
        path = os.path.join(map_dir, f"deposit_map_kt{kt}.npz")
        if not os.path.exists(path):
            raise SystemExit(
                f"FATAL: no deposit map at {path}. Produce the series with "
                "baro_deposit_time_walk.py --consecutive --deposit-map-dir.")
        z = np.load(path, allow_pickle=False)
        got = int(z["ic_step"])
        if got != kt:
            raise SystemExit(f"FATAL: {path} carries ic_step={got}, want {kt}")
        for key in keys + meta:
            if key not in z:
                raise SystemExit(f"FATAL: {path} has no {key!r}")
        # meta keys are STRINGS -- presence-checked above, carried on the
        # stamp, never stacked into the numeric arrays.
        for key in keys:
            fields.setdefault(key, []).append(np.asarray(z[key], float))
        stamps.append({"kt": kt, "path": path,
                       "seqdump": str(z["seqdump"]),
                       "provenance": str(z["provenance"])})
    out = {k: np.stack(v, axis=0) for k, v in fields.items()}
    for key in ("wetu", "wetv", "wgt_primary"):
        a = out[key]
        if not np.array_equal(a, np.broadcast_to(a[0], a.shape)):
            raise SystemExit(
                f"FATAL: {key} differs across states; the state mean would "
                "average different cells at different times.")
    out["_stamps"] = stamps
    return out


# WHICH KNOBS ACTUALLY DECIDE THE PHYSICS OF A DEPOSIT MAP.
#
# Round-3 review demolished the first version of this list.  It checked
# DINO_HU_WIND / DINO_ZUFRC_WIND / DINO_SEAM_WIND -- which are INERT for this
# producer.  They are continuity controls belonging to OTHER probes
# (hu_avg_perface_diff, zu_frc_assembly_table, seq_seam_walk) and land in this
# stamp only because the provenance helper records one shared environment list
# for the whole directory.  `substep_traj_compare`, which writes these maps,
# never reads them.  Meanwhile the one stamped knob it DOES read that changes
# the wind -- the seasonal clock, on a card whose wind has an annual cycle --
# was omitted entirely.  So the gate was checking three switches that cannot
# matter and missing the one that can.
_PRODUCER_KNOBS = ("DINO_1226_T_SECONDS", "DINO_1226_IC_STEP",
                   "LEGOESM_NEMO_E3T", "DINO_1455_SUB_VFACE",
                   "DINO_1455_SUB_CORIOLIS")
# Recorded for the audit trail, explicitly NOT used as a wind-on witness.
_INERT_KNOBS = ("DINO_HU_WIND", "DINO_ZUFRC_WIND", "DINO_SEAM_WIND",
                "DINO_RECONCILE")


def parse_stamp(prov: str) -> dict:
    """Pull the producer's own knobs, and the inert ones, out of a stamp."""
    import re
    out = {}
    for key in _PRODUCER_KNOBS + _INERT_KNOBS:
        m = re.search(key + r"=('[^']*'|\S+)", prov)
        out[key] = m.group(1).strip("'") if m else None
    return out


#: The v-face metric arms a map directory can hold.  "none" is the plain
#: frozen-forcing arm (a map produced before the arm existed carries no stamp
#: at all and normalises to this).
_VFACE_ARMS = ("none", "nemo", "stagger")

#: The vertex-Coriolis arms, added for #1455 next-action 1.  Same three states
#: and the same reason: a map produced with NEMO's own ff_f substituted into
#: the loop is a well-formed array on an identical wet mask, so which arm made
#: it can only travel with it as a stamp.
_CORIOLIS_ARMS = ("none", "nemo", "stagger")


def vface_arm_of(parsed: dict) -> str:
    """Normalise one stamp's v-face arm to a member of ``_VFACE_ARMS``."""
    got = parsed.get("DINO_1455_SUB_VFACE")
    return "none" if got in (None, "None", "") else got


def coriolis_arm_of(parsed: dict) -> str:
    """Normalise one stamp's vertex-Coriolis arm to ``_CORIOLIS_ARMS``."""
    got = parsed.get("DINO_1455_SUB_CORIOLIS")
    return "none" if got in (None, "None", "") else got


def assert_map_provenance(stamps: list, cli_seqdump: str,
                          vface_arm: str = "none",
                          coriolis_arm: str = "none") -> dict:
    """Refuse to run unless the maps agree AND are in the expected ON state.

    AGREEMENT IS NOT ENOUGH, which is what round-3 review caught: five
    identical wind-OFF maps agree perfectly with each other and sailed straight
    through the first version of this gate.  So each knob the producer actually
    consumes is checked against its expected value, not merely against its
    neighbours.

    What is checked, and why each one can change the answer:

      * ``LEGOESM_NEMO_E3T`` must be ``'both'``.  Anything else silently swaps
        the vertical ladder -- the default that this card's history records as
        having contaminated four separate measurements before it was gated.
      * ``DINO_1226_T_SECONDS`` must be UNSET, so the seasonal clock is derived
        from the step rather than overridden.  DINO's wind has an annual cycle,
        so an overridden clock is a different wind field at the same step.
      * the stamped ``DINO_1226_IC_STEP`` must equal the map's own ``kt``.
        Together with the clause above this pins the clock: derived, and
        derived from the right step.

    THE RESIDUAL GAP, named rather than papered over.  The ORACLE side's
    wind-on state IS witnessed, by `wind_candidate`'s frame guard: NEMO's wind
    increment is nonzero on every one of the wet u faces, which a wind-off
    oracle run could not produce.  legoESM's OWN wind-on state is not directly
    stamped in these maps at all; the clock checks above are the strongest
    proxy available from the saved artifacts, and they are a proxy.  Closing
    that properly means stamping the resolved stress amplitude in the producer.
    """
    if vface_arm not in _VFACE_ARMS:
        raise SystemExit(
            f"vface_arm={vface_arm!r}: expected one of {_VFACE_ARMS}")
    if coriolis_arm not in _CORIOLIS_ARMS:
        raise SystemExit(
            f"coriolis_arm={coriolis_arm!r}: expected one of {_CORIOLIS_ARMS}")
    parsed = [parse_stamp(st["provenance"]) for st in stamps]
    first = parsed[0]

    # THE CROSS-MAP AGREEMENT CLAUSE IS GONE, deliberately.  It checked that
    # the five maps matched each other; every knob it covered now has a
    # PER-MAP on-state check below, which is strictly stronger (it also catches
    # all five being wrong together, which agreement cannot).  With both
    # present neither could be tested: removing either one left the suite green
    # because the other silently covered for it -- two mutations, both green,
    # which is precisely the "guard that cannot fail" this file exists to
    # refuse.  One real guard beats two that alibi each other.
    #
    # If a knob is ever added here WITHOUT an on-state check, restore the
    # agreement clause for that knob and give it a test that fails when the
    # clause is removed.

    # the ON-STATE itself, per knob and PER MAP, not merely consistency.
    # The ladder check used to sit outside this loop and inspect only the first
    # map, so a series with map 0 right and maps 1-4 wrong walked straight
    # through (confirmation pass 2026-08-26).  Every on-state check is now
    # inside this loop, and this loop is the ONLY guard -- the cross-map
    # agreement clause was deleted for the reason given above, so there is no
    # second line of defence and none is wanted: two guards that alibi each
    # other cannot be tested apart.  Each check below therefore has to hold on
    # its own, and each has a test that fails when it is moved outside.
    for st, got in zip(stamps, parsed):
        if got["LEGOESM_NEMO_E3T"] != "both":
            raise SystemExit(
                f"FATAL: {st['path']} was produced with LEGOESM_NEMO_E3T="
                f"{got['LEGOESM_NEMO_E3T']!r}, not 'both'. That is a different "
                "vertical ladder, and this card's history records that default "
                "contaminating four measurements before it was gated.")
        if got["DINO_1226_T_SECONDS"] not in (None, "None"):
            raise SystemExit(
                f"FATAL: {st['path']} carries an OVERRIDDEN seasonal clock "
                f"(DINO_1226_T_SECONDS={got['DINO_1226_T_SECONDS']!r}). DINO's "
                "wind has an annual cycle, so an overridden clock is a "
                "different wind field at the same step.")
        if got["DINO_1226_IC_STEP"] not in (None, str(st["kt"])):
            raise SystemExit(
                f"FATAL: {st['path']} is map kt={st['kt']} but its stamp says "
                f"IC_STEP={got['DINO_1226_IC_STEP']}. The seasonal clock is "
                "derived from that step, so the wind would be from another "
                "time.")
        # THE V-FACE METRIC ARM, per map.  A map produced with NEMO's own e1v
        # substituted into the loop is a well-formed array on an identical wet
        # mask, so no numerical guard can tell it from a baseline one -- the
        # same failure mode as the wind-off maps this gate was built for.  The
        # caller must SAY which arm it is scoring and every map must be it,
        # because a mixed directory would be averaged into a five-state mean
        # that is neither arm.
        if vface_arm_of(got) != vface_arm:
            raise SystemExit(
                f"FATAL: {st['path']} was produced with the v-face metric arm "
                f"{vface_arm_of(got)!r}, but this run is scoring "
                f"{vface_arm!r}. Mixing arms in one directory averages two "
                "different experiments into one five-state mean.")
        # THE VERTEX-CORIOLIS ARM, per map, for exactly the same reason.  The
        # #1455 pair is scored on a JOINT arm, so a directory can legitimately
        # carry BOTH substitutions -- which is precisely why each has to be
        # named and checked separately: "the joint arm" and "the e1v arm" are
        # indistinguishable in the arrays and differ only here.
        if coriolis_arm_of(got) != coriolis_arm:
            raise SystemExit(
                f"FATAL: {st['path']} was produced with the vertex-Coriolis "
                f"arm {coriolis_arm_of(got)!r}, but this run is scoring "
                f"{coriolis_arm!r}. Mixing arms in one directory averages two "
                "different experiments into one five-state mean.")

    # (3) the CLI oracle directory must be one the maps came from
    seqdumps = [st["seqdump"] for st in stamps]
    cli = os.path.normpath(cli_seqdump)
    if cli not in [os.path.normpath(x) for x in seqdumps]:
        raise SystemExit(
            f"FATAL: --seqdump {cli} is not any of the oracle directories the "
            f"maps were produced from ({sorted(set(seqdumps))}). The wind "
            "increment and the bias would come from different runs.")
    return {"producer_knobs": {k: first[k] for k in _PRODUCER_KNOBS},
            "vface_arm": vface_arm,
            "coriolis_arm": coriolis_arm,
            "inert_knobs_recorded_not_checked": {k: first[k]
                                                 for k in _INERT_KNOBS},
            "seqdumps": seqdumps,
            "cli_seqdump_matches_map": True,
            "lego_wind_on_directly_stamped": False}


def wind_step_invariance(stamps: list) -> dict:
    """Is the wind increment the same in every state's own oracle dump?

    The wind dump is the one file in an oracle directory with no step in its
    name, so the frame guard cannot tell one state's from another's.  Rather
    than assume it is step-invariant, load it from EVERY state's own directory
    and report the spread.  If this is large the single-directory comparison in
    `wind_candidate` is invalid and says so.
    """
    fields, missing = [], []
    for st in stamps:
        path = os.path.join(st["seqdump"], "wnd_dump_zu_frc_inc.bin")
        if not os.path.exists(path):
            missing.append(path)
            continue
        fields.append(load_nemo_2d(st["seqdump"], "wnd_dump_zu_frc_inc.bin"))
    if len(fields) < 2:
        return {"n_dumps": len(fields), "missing": missing,
                "max_relative_spread": None,
                "note": "fewer than two dumps available; invariance UNTESTED"}
    a = np.stack(fields, axis=0)
    ref = np.abs(a[0]).max()
    spread = float(np.abs(a - a[0]).max() / max(ref, 1e-300))
    return {"n_dumps": len(fields), "missing": missing,
            "max_relative_spread": spread,
            "STEP_INVARIANT": bool(spread < 1e-3)}


def load_nemo_2d(seqdump: str, name: str) -> np.ndarray:
    path = os.path.join(seqdump, name)
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: no NEMO dump at {path}")
    a = np.fromfile(path, dtype="<f8")
    if a.size != JPJ * JPI:
        raise SystemExit(
            f"FATAL: {path} has {a.size} elements, expected {JPJ * JPI} "
            f"({JPJ}x{JPI} haloless). The loader convention is wrong.")
    return a.reshape(JPJ, JPI)


def coriolis_at_row(seqdump: str, j: int) -> float:
    """NEMO's OWN Coriolis parameter, wet-cell mean over row ``j``.

    Read from the oracle's `domain_cfg_out.nc` (`ff_t`) rather than recomputed
    from a latitude, so the rotated bound in `wind_candidate` cannot disagree
    with the model whose response it is bounding.
    """
    path = os.path.join(seqdump, "domain_cfg_out.nc")
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: no domain_cfg_out.nc at {path}; the rotated "
                         "meridional bound cannot be computed without NEMO's "
                         "own Coriolis parameter.")
    import netCDF4 as nc
    d = nc.Dataset(path)
    ff = np.array(d["ff_t"][0])
    wet = np.array(d["bottom_level"][0])[j] > 0
    if not wet.any():
        raise SystemExit(f"FATAL: row {j} carries no wet cell in ff_t")
    return float(ff[j][wet].mean())


# --------------------------------------------------------------------------
# section 1 -- constancy, proved per state rather than assumed
# --------------------------------------------------------------------------
def constancy(series: np.ndarray, wet: np.ndarray) -> dict:
    """How nearly is this residual the SAME FIELD at every state?

    Reported as the per-state RMS, the RMS of the state mean, and the RMS of
    each state's DEPARTURE from that mean.  The departure is the number that
    matters: it bounds everything a state-to-state estimator could ever see.
    """
    x = series[:, wet]
    mean = x.mean(axis=0)
    dev = x - mean
    per_state = [float(np.sqrt((x[s] ** 2).mean())) for s in range(x.shape[0])]
    dev_rms = [float(np.sqrt((dev[s] ** 2).mean())) for s in range(x.shape[0])]
    mean_rms = float(np.sqrt((mean ** 2).mean()))
    cors = []
    for a in range(x.shape[0]):
        for b in range(a + 1, x.shape[0]):
            if x[a].std() > 0 and x[b].std() > 0:
                cors.append(float(np.corrcoef(x[a], x[b])[0, 1]))
    return {"per_state_rms": per_state,
            "state_mean_rms": mean_rms,
            "departure_rms": dev_rms,
            "max_departure_share": (max(dev_rms) / mean_rms
                                    if mean_rms > 0 else float("nan")),
            # SIGNED: an anti-correlated pair is an alternating mode and must
            # never be allowed to read as state-constant.
            "cross_state_corr_min": min(cors) if cors else float("nan")}


def constancy_selftest() -> dict:
    """The constancy statistic on fields with KNOWN answers, before use.

    A constant field must report zero departure; a state-ALTERNATING field must
    report a departure comparable to its own amplitude and a NEGATIVE minimum
    cross-state correlation.  Without the second case a sign-blind statistic
    would call the two-step mode itself 'constant'.
    """
    rng = np.random.default_rng(0)
    wet = np.ones((4, 4), dtype=bool)
    base = rng.standard_normal((4, 4))
    const = np.stack([base] * 5, axis=0)
    alt = np.stack([base * ((-1.0) ** s) for s in range(5)], axis=0)
    c_const, c_alt = constancy(const, wet), constancy(alt, wet)
    return {"constant_departure_share": c_const["max_departure_share"],
            "constant_corr_min": c_const["cross_state_corr_min"],
            "alternating_departure_share": c_alt["max_departure_share"],
            "alternating_corr_min": c_alt["cross_state_corr_min"]}


# --------------------------------------------------------------------------
# section 2/3 -- the spatial map, per row, per cell
# --------------------------------------------------------------------------
def row_profile(field: np.ndarray, wet: np.ndarray) -> list:
    """Per-row RMS, signed mean and COHERENCE of a 2-D field.

    Coherence = |signed mean| / RMS over the row's wet cells: 1 means a
    zonally uniform offset, 0 means a zero-mean pattern in longitude.  It is
    what separates 'a bias' from 'a wiggle of the same amplitude'.
    """
    rows = []
    for j in range(field.shape[0]):
        m = wet[j]
        if not m.any():
            rows.append({"j": j, "n_wet": 0, "rms": None, "signed": None,
                         "coherence": None})
            continue
        v = field[j][m]
        rms = float(np.sqrt((v ** 2).mean()))
        signed = float(v.mean())
        rows.append({"j": j, "n_wet": int(m.sum()), "rms": rms,
                     "signed": signed,
                     "coherence": (abs(signed) / rms) if rms > 0 else 0.0})
    return rows


def two_dx_share(field: np.ndarray, wet: np.ndarray, j: int) -> float | None:
    """How much of a row's variance is a TWO-GRIDPOINT zigzag in longitude.

    A row with near-zero coherence is a zero-mean pattern, but that alone does
    not say WHICH pattern.  This separates "a grid-scale checkerboard", which
    would point at a numerical null mode, from "a smooth large-scale pattern
    that happens to average to zero", which would not.  Reported as the share
    of the row's RMS carried by the alternating basis vector, after removing
    that vector's own mean so it is orthogonal to a constant offset.
    """
    m = wet[j]
    if m.sum() < 8:
        return None
    v = field[j][m]
    rms = float(np.sqrt((v ** 2).mean()))
    if rms <= 0.0:
        return None
    alt = (-1.0) ** np.arange(v.size)
    alt = alt - alt.mean()
    proj = float((v @ alt) / (alt @ alt))
    return float(abs(proj) * np.sqrt((alt ** 2).mean()) / rms)


def cell_regression(bias: np.ndarray, vel: np.ndarray, wet: np.ndarray) -> dict:
    """Per-CELL least-squares slope and correlation of the bias on the velocity.

    THE CONTROL THAT MAKES SECTION 3 HONEST.  A row's amplitude ratio can match
    the velocity's amplitude ratio by coincidence; only the per-cell
    correlation says whether the bias is actually the velocity times a number.
    Both are always reported together.
    """
    b, a = bias[wet], vel[wet]
    if b.size < 8 or float((a ** 2).sum()) == 0.0:
        return {"slope": None, "corr": None, "n": int(b.size)}
    slope = float((b @ a) / (a @ a))
    corr = (float(np.corrcoef(b, a)[0, 1])
            if b.std() > 0 and a.std() > 0 else None)
    return {"slope": slope, "corr": corr, "n": int(b.size)}


def wall_and_interior(bias: np.ndarray, wet: np.ndarray,
                      exclude: int = 5) -> dict:
    """The headline numbers: each wall row's amplitude, and the interior median.

    Extracted from the report loop so it can be TESTED.  Review 2026-08-26
    mutated the wall reduction from an RMS to a SIGNED MEAN and the whole suite
    stayed green -- a cancelling reduction on the one number every claim in
    section 2 quotes.  The wall statistic is an RMS *deliberately*: the
    meridional bias at these rows is near zero-mean in longitude, so a signed
    mean would report ~0 for a field of amplitude 4e-07.
    """
    j_s, j_n = wall_rows(wet)
    interior = []
    for j in range(j_s + exclude, j_n - exclude + 1):
        if wet[j].any():
            interior.append(float(np.sqrt((bias[j][wet[j]] ** 2).mean())))
    return {"wall_rows": [j_s, j_n],
            "south": float(np.sqrt((bias[j_s][wet[j_s]] ** 2).mean())),
            "north": float(np.sqrt((bias[j_n][wet[j_n]] ** 2).mean())),
            "interior_median": (float(np.median(interior)) if interior
                                else float("nan")),
            "n_interior_rows": len(interior),
            "exclude_rows_each_side": exclude}


def wall_rows(wet: np.ndarray) -> tuple:
    """The southernmost and northernmost rows carrying a wet face, from the MASK.

    Never hardcoded: the two components name different northern rows because a
    meridional face sits between two tracer rows, and that staggering is a
    property of the data, not of this file.
    """
    live = [j for j in range(wet.shape[0]) if wet[j].any()]
    if not live:
        raise SystemExit("FATAL: mask has no wet row")
    return live[0], live[-1]


def mirrored_pairs(bias: np.ndarray, vel: np.ndarray, wet: np.ndarray,
                   depth: int = 14) -> list:
    """Rows at equal distance from the two end walls, side by side.

    The wall GEOMETRY of this configuration is NEARLY mirror-symmetric but not
    exactly (section 5: same latitude and same zonal spacing, but the northern
    column is 14.7 % deeper, which raises that section's asymmetry flag).  A
    bias generated by boundary geometry alone would therefore come out within
    ~1.15x here, not exactly equal.  Each pair carries its amplitude ratio AND both per-cell
    correlations, and a flag when the two disagree.
    """
    j_s, j_n = wall_rows(wet)
    out = []
    for d in range(depth):
        js, jn = j_s + d, j_n - d
        if js >= jn or not wet[js].any() or not wet[jn].any():
            continue
        bs = float(np.sqrt((bias[js][wet[js]] ** 2).mean()))
        bn = float(np.sqrt((bias[jn][wet[jn]] ** 2).mean()))
        vs = float(np.sqrt((vel[js][wet[js]] ** 2).mean()))
        vn = float(np.sqrt((vel[jn][wet[jn]] ** 2).mean()))
        rs = cell_regression(bias[js], vel[js], wet[js])
        rn = cell_regression(bias[jn], vel[jn], wet[jn])
        amp_ratio = (bn / bs) if bs > 0 else float("nan")
        vel_ratio = (vn / vs) if vs > 0 else float("nan")
        # The flag: amplitudes track the velocity while the cells do not.
        agree_amp = (np.isfinite(amp_ratio) and np.isfinite(vel_ratio)
                     and abs(amp_ratio / vel_ratio - 1.0) < 0.25)
        # ANY end failing the per-cell test is enough to void the reading:
        # proportionality is a claim about BOTH ends, so one end whose cells do
        # not track the velocity already means the matching amplitudes are a
        # coincidence.  An earlier draft used `all` here and the flag stayed
        # silent on exactly the row pair that motivated it.
        weak_cells = any((r["corr"] is None or abs(r["corr"]) < 0.3)
                         for r in (rs, rn))
        out.append({"d": d, "j_south": js, "j_north": jn,
                    "bias_south": bs, "bias_north": bn,
                    "vel_south": vs, "vel_north": vn,
                    "bias_amp_ratio_n_over_s": amp_ratio,
                    "vel_amp_ratio_n_over_s": vel_ratio,
                    "cell_fit_south": rs, "cell_fit_north": rn,
                    "AMPLITUDE_COINCIDENCE_FLAG": bool(agree_amp
                                                       and weak_cells)})
    return out


# --------------------------------------------------------------------------
# section 4 -- the wind-stress placement candidate
# --------------------------------------------------------------------------
def wind_candidate(seqdump: str, bias_u: np.ndarray, bias_v: np.ndarray,
                   wetu: np.ndarray, wetv: np.ndarray, wgt: np.ndarray,
                   response: float, dt_s: float) -> dict:
    """Does the wind-stress placement difference have the shape to own the bias?

    NEMO dumps the wind's OWN additive increment to the barotropic slow forcing
    (`wnd_dump_z{u,v}_frc_inc.bin`, the term at dynspg_ts.F90:443-444), so the
    candidate's footprint is read rather than reconstructed.

    WHAT DECIDES THIS SECTION, and what does NOT.  The decider is STRUCTURAL:
    this bias is measured in the arm where NEMO's own zu_frc/zv_frc are
    substituted into legoESM's loop, so every difference in how legoESM
    ASSEMBLES its slow forcing -- the wind term included -- has already been
    removed.  THAT ARGUMENT HAS FOUR PREMISES, not one, and only two of them
    are verified by source (the loop carries no stress term of its own, so
    there is no second route in; and the substitution replaces the forcing
    only).  The third -- that these maps ARE the wind-on arm -- is now asserted
    from the maps' own provenance rather than assumed (`assert_map_provenance`,
    after review 2026-08-26 found nothing checked it).  The fourth -- whether
    the loop re-scales the substituted forcing by legoESM's OWN face depth --
    is UNVERIFIED and named as such here and in the accompanying document.  So
    this is a strong argument, not a closed one.  The quantities below are INDEPENDENT SUPPORTING
    CHECKS, each of which is partial, and each is reported with the part of the
    question it cannot answer:

      1. the meridional increment is EXACTLY zero over the whole domain, because
         DINO's wind stress is zonal-only on both sides.  *** THIS IS A FACT
         ABOUT THE FORCING, NOT ABOUT THE RESPONSE, AND IT DOES NOT EXCLUDE A
         WALL-NORMAL BIAS. ***  A zonal forcing integrated inside a ROTATING
         barotropic loop deposits meridional velocity through Coriolis, so the
         rotated response is computed below rather than waved away.  An earlier
         version of this probe wrote "the candidate has no wall-normal component
         at all" from the zero forcing; adversarial review 2026-08-26 caught it,
         and the rotated bound it asked for turns out to EXCEED the measured
         meridional bias at the southern wall.  Confusing a null forcing with a
         null response is exactly the class of error this campaign keeps paying
         for, so the refutation is kept in the code;
      2. the zonal increment's row profile against the bias's row profile;
      3. the increment at the wall rows against its own basin maximum.

    Both size bounds -- the direct zonal one and the rotated meridional one --
    are reported at the MEASURED loop response and at an assumption-free
    response of 1.0, and at 100 % of the wind term rather than at the small
    measured placement offset.  That is deliberately generous to the candidate:
    a bound that still fails to exclude under those terms is reported as NOT
    EXCLUDING, and this one does fail at the southern wall.
    """
    wu = load_nemo_2d(seqdump, "wnd_dump_zu_frc_inc.bin")
    wv = load_nemo_2d(seqdump, "wnd_dump_zv_frc_inc.bin")

    # ALIGNMENT, established from the data rather than assumed: the wind
    # increment must be nonzero on exactly the wet u-face set the deposit map
    # carries.  If the two disagree the arrays are on different frames and no
    # comparison below means anything.
    nz = (wu != 0.0)
    if not np.array_equal(nz, wetu):
        raise SystemExit(
            "FATAL: the wind increment's nonzero set (%d cells) is not the "
            "deposit map's wet u mask (%d cells); the two arrays are on "
            "different frames and the comparison is void."
            % (int(nz.sum()), int(wetu.sum())))

    # PROVENANCE, because this is the one dump in the seqdump directory that
    # carries no kt in its filename (every other one is `..._kt00005761.bin`).
    # Its nonzero set is step-invariant, so the frame guard above cannot tell
    # one step's file from another's; stamping the source and mtime is what
    # makes the artifact re-checkable.  Measured exposure: across the seqdump
    # directories the increment varies by <=1.7e-4 relative, so the ambiguity
    # is harmless here -- but that is a measurement, not a guarantee.
    _wu_path = os.path.join(seqdump, "wnd_dump_zu_frc_inc.bin")
    v_is_zero = bool((wv == 0.0).all())
    j_s_u, j_n_u = wall_rows(wetu)

    prof_w = [r["rms"] for r in row_profile(wu, wetu)]
    prof_b = [r["rms"] for r in row_profile(np.abs(bias_u), wetu)]
    live = [j for j in range(JPJ) if prof_w[j] is not None]
    pw = np.array([prof_w[j] for j in live])
    pb = np.array([prof_b[j] for j in live])
    # WHOLE-DOMAIN context, not a wall statistic: it compares two 197-row
    # profiles and is printed beside two wall-local facts, so it is labelled.
    # A degenerate (constant) profile has no correlation -- report None rather
    # than let a NaN propagate into the artifact and read as 0.
    shape_corr = (float(np.corrcoef(pw, pb)[0, 1])
                  if pw.std() > 0 and pb.std() > 0 else None)

    peak_j = int(live[int(np.argmax(pw))])
    peak = float(pw.max())
    at_south = float(prof_w[j_s_u])
    at_north = float(prof_w[j_n_u])

    def _bias_rms(field, wet, j):
        return float(np.sqrt((field[j][wet[j]] ** 2).mean()))

    # forcing [m/s^2] -> deposit [m/s] over the averaging window
    jbar = float((wgt * np.arange(1, wgt.size + 1)).sum())
    out = {"wind_dump_path": _wu_path,
           "wind_dump_mtime": os.path.getmtime(_wu_path),
           "wind_dump_is_step_stamped": False,
           "v_increment_exactly_zero": v_is_zero,
           "v_increment_max_abs": float(np.abs(wv).max()),
           "u_increment_rms_wet": float(np.sqrt((wu[wetu] ** 2).mean())),
           "u_increment_peak": peak, "u_increment_peak_row": peak_j,
           "u_increment_at_south_wall": at_south,
           "u_increment_at_north_wall": at_north,
           "wall_vs_peak_suppression_south": (peak / at_south
                                              if at_south > 0 else None),
           "wall_vs_peak_suppression_north": (peak / at_north
                                              if at_north > 0 else None),
           "row_profile_shape_corr_with_bias": shape_corr,
           "jbar_substeps": jbar,
           "bias_u_at_south_wall": _bias_rms(bias_u, wetu, j_s_u),
           "bias_u_at_north_wall": _bias_rms(bias_u, wetu, j_n_u),
           "bounds": {}}
    # The ROTATED meridional response to a purely zonal forcing.  Inside the
    # loop, dv/dt = -f*u while u itself grows as F_u*t under a forcing held
    # constant across substeps, so v(k) ~ -f*F_u*(k*dt)^2/2 and the boxcar
    # average over the window is -f*F_u*dt^2*sum(w_k * k^2)/2.  This is an
    # UPPER bound and is labelled as one: it carries no bottom drag, no lateral
    # friction, and none of the sea-surface-gradient response that opposes the
    # build-up.  It exists so that "the meridional forcing is zero" can never
    # again be read as "the meridional response is zero".
    k = np.arange(1, wgt.size + 1, dtype=float)
    rot_kernel = float((wgt * k ** 2).sum()) / 2.0
    j_s_v, j_n_v = wall_rows(wetv)

    # *** CO-LOCATION, and it flipped a verdict (review round 3). ***
    # The rotated bound evaluates a MERIDIONAL response, so it lives on the v
    # grid's wall rows -- but the forcing driving it is ZONAL and lives on the u
    # grid.  The first version took the forcing from the u grid's OWN wall rows.
    # In the south those coincide (both row 1); in the NORTH they do not (u wall
    # 197, v wall 196), and the wind increment climbs steeply away from the
    # wall -- 2.6x one row in, 25x by ten rows.  Reading the forcing a row too
    # far out UNDERSTATED it by 2.57x at the north and turned a 1.5x
    # non-exclusion into a 3.9x "exclusion".
    #
    # A v face at row j is flanked by the u faces at rows j and j+1, so the
    # forcing is taken over exactly those, and the MAX is used because this is
    # an upper bound and the max is the generous direction for the candidate.
    # Which rows each side came from is REPORTED, not left implicit.
    def _wind_amp_adjacent(j_v):
        rows, amps = [], []
        for j in (j_v, j_v + 1):
            if 0 <= j < wu.shape[0] and wetu[j].any():
                rows.append(int(j))
                amps.append(float(np.sqrt((wu[j][wetu[j]] ** 2).mean())))
        if not amps:
            raise SystemExit(
                f"FATAL: v wall row {j_v} has no wet u face at rows "
                f"{j_v} or {j_v + 1}; the rotated bound has no forcing to "
                "co-locate with and would silently read zero.")
        i = int(np.argmax(amps))
        return amps[i], rows, rows[i]

    f_wall, rot_forcing = {}, {}
    for tag, j in (("south", j_s_v), ("north", j_n_v)):
        f_wall[tag] = float(coriolis_at_row(seqdump, j))
        amp, rows, picked = _wind_amp_adjacent(j)
        rot_forcing[tag] = {"amp": amp, "v_response_row": int(j),
                            "u_forcing_rows_considered": rows,
                            "u_forcing_row_used": picked}
    out["rotated_bound_kernel_sum_w_k2_over_2"] = rot_kernel
    out["coriolis_at_v_wall_rows"] = f_wall
    out["rotated_forcing_colocation"] = rot_forcing
    # HONESTY NOTE (round 3, verified independently): the small-angle step used
    # to build this kernel is applied OUTSIDE its stated regime -- the rotation
    # angle |f|*n*dt reaches ~1.09 rad by window end, not a small angle.
    #
    # The overstatement is DERIVED here from the arrays already in scope, not
    # pasted as a constant.  For du/dt = f*v + F_u, dv/dt = -f*u from rest the
    # exact meridional response is (F_u/f)*(1 - cos(f*t)); the small-angle
    # step keeps only F_u*f*t^2/2.  Averaging both over this window's own
    # weights gives the ratio directly.  (The constant it replaces was correct,
    # but this file has spent four review rounds on numbers that looked
    # rigorous because they were printed -- including one of the reviewer's --
    # so a derivable quantity gets derived.)
    out["rotation_angle_rad_at_window_end"] = {
        tag: abs(f_wall[tag]) * float(wgt.size) * dt_s for tag in f_wall}
    out["small_angle_overstatement_frac"] = {
        tag: small_angle_overstatement(abs(f_wall[tag]), dt_s, wgt)
        for tag in f_wall}
    out["bias_v_at_south_wall"] = _bias_rms(bias_v, wetv, j_s_v)
    out["bias_v_at_north_wall"] = _bias_rms(bias_v, wetv, j_n_v)

    for label, R in (("measured_response", response), ("response_1.0", 1.0)):
        factor = R * jbar * dt_s
        rot = {}
        for tag in ("south", "north"):
            wind_amp = rot_forcing[tag]["amp"]
            dv = abs(f_wall[tag]) * wind_amp * (dt_s ** 2) * rot_kernel * R
            bias_v_amp = out[f"bias_v_at_{tag}_wall"]
            rot[tag] = {
                "rotated_v_deposit_from_FULL_wind_term": dv,
                "measured_v_bias": bias_v_amp,
                "ratio_bias_over_rotated_wind": bias_v_amp / max(dv, 1e-300),
                # The honest field: does this bound actually exclude?
                "EXCLUDES": bool(bias_v_amp > 3.0 * dv)}
        out["bounds"][label] = {
            "response": R,
            "deposit_from_FULL_wind_term_south": at_south * factor,
            "deposit_from_FULL_wind_term_north": at_north * factor,
            "ratio_bias_over_full_wind_south":
                out["bias_u_at_south_wall"] / max(at_south * factor, 1e-300),
            "ratio_bias_over_full_wind_north":
                out["bias_u_at_north_wall"] / max(at_north * factor, 1e-300),
            "u_EXCLUDES_south":
                bool(out["bias_u_at_south_wall"] > 3.0 * at_south * factor),
            "u_EXCLUDES_north":
                bool(out["bias_u_at_north_wall"] > 3.0 * at_north * factor),
            "rotated_meridional": rot}
    return out


# --------------------------------------------------------------------------
# section 5 -- the wall geometry both walls actually have
# --------------------------------------------------------------------------
def wall_geometry(seqdump: str) -> dict:
    """Latitude, column depth and meridional grid spacing at the two end walls.

    The point of the section: the two walls are NEARLY mirror images -- same
    latitude, same zonal spacing -- but NOT exactly, the northern column being
    14.7 % deeper.  That residual asymmetry bounds what boundary geometry alone
    could produce (~1.15x) and is reported next to the measured contrast
    (11.7x) rather than rounded off to the word "symmetric".
    """
    path = os.path.join(seqdump, "domain_cfg_out.nc")
    if not os.path.exists(path):
        return {"available": False, "reason": f"no domain_cfg_out.nc at {path}"}
    try:
        import netCDF4 as nc
    except ImportError:
        return {"available": False, "reason": "netCDF4 not importable"}
    d = nc.Dataset(path)
    lat = np.array(d["gphit"][0])
    e2t = np.array(d["e2t"][0])
    bl = np.array(d["bottom_level"][0])
    e3t0 = np.array(d["e3t_0"][0])
    live = [j for j in range(bl.shape[0]) if (bl[j] > 0).any()]
    out = {"available": True, "grid": "TRACER (T) points -- gphit/e2t/"
                                      "bottom_level", "rows": {}}
    for tag, j in (("south_wall_T", live[0]), ("north_wall_T", live[-1])):
        wet = bl[j] > 0
        # WET-CELL MEANS, not a single mid-basin column: a lone column is one
        # sample of a row that the bias statistics average over in full.
        col = np.array([e3t0[:int(bl[j, i]), j, i].sum()
                        for i in range(bl.shape[1])])
        out["rows"][tag] = {"j": int(j),
                            "lat_deg_mean_wet": float(lat[j][wet].mean()),
                            "n_wet": int(wet.sum()),
                            "mean_depth_m": float(col[wet].mean()),
                            "e2t_m_mean_wet": float(e2t[j][wet].mean())}
    out["symmetry"] = symmetry_summary(out["rows"]["south_wall_T"],
                                       out["rows"]["north_wall_T"])
    return out


def symmetry_summary(a: dict, b: dict) -> dict:
    """Is the two-wall geometry symmetric, and by how much is it not?

    Split out of `wall_geometry` so it can be tested without a netCDF file --
    review 2026-08-26 mutated the latitude difference to a hard 0.0 and the
    suite stayed green, because nothing exercised this arithmetic at all.
    """
    depth_ratio = b["mean_depth_m"] / a["mean_depth_m"]
    e2t_ratio = b["e2t_m_mean_wet"] / a["e2t_m_mean_wet"]
    lat_diff = abs(abs(a["lat_deg_mean_wet"]) - abs(b["lat_deg_mean_wet"]))
    # THE FLAG.  "Symmetric" is a claim, and column depth -- the one geometric
    # variable a barotropic velocity most plausibly scales with -- is NOT
    # symmetric here (1.147).  Review 2026-08-26: nothing fired even at a
    # hypothetical 3.0.  The flag names the largest departure and states what
    # asymmetry that departure could itself produce, so the reader compares it
    # against the measured 11.7x rather than against the word "symmetric".
    # Latitude enters the flag too: two walls at different |latitude| are not
    # mirror images however well their depths match.
    worst = max(abs(depth_ratio - 1.0), abs(e2t_ratio - 1.0),
                lat_diff / 90.0)
    return {
        "abs_lat_diff_deg": lat_diff,
        "depth_ratio_n_over_s": depth_ratio,
        "e2t_ratio_n_over_s": e2t_ratio,
        "worst_fractional_asymmetry": worst,
        "GEOMETRY_ASYMMETRY_FLAG": bool(worst > 0.05),
        "note": ("the wall rows of the U and V grids are NOT these T rows -- "
                 "the staggering puts them at different indices; these are "
                 "T-point geometry and are labelled as such")}


# --------------------------------------------------------------------------
# section 6 -- the v-face zonal metric, the leading untested candidate
# --------------------------------------------------------------------------
def small_angle_overstatement(f_abs: float, dt_s: float,
                              wgt: np.ndarray) -> float:
    """By how much does the small-angle rotated bound exceed the exact one?

    Exact inertial response to a steady zonal forcing from rest:
    ``v(t) = -(F_u/f)*(1 - cos(f*t))``.  The small-angle step keeps only the
    leading term ``-F_u*f*t^2/2``.  Both are linear in ``F_u``, so it cancels
    and the ratio depends on the rotation angle alone.  Returns
    ``1 - exact/approx``: POSITIVE means the bound overstates, which is the
    conservative direction for an exclusion.
    """
    k = np.arange(1, wgt.size + 1, dtype=float)
    theta = f_abs * dt_s * k
    approx = float((wgt * theta ** 2).sum()) / 2.0
    if approx <= 0.0:
        return 0.0
    # 1 - cos(theta) via the HALF-ANGLE identity, not literally.  Written
    # directly it cancels catastrophically: at theta ~ 1e-12, cos(theta)
    # rounds to exactly 1.0 in float64, the numerator becomes 0, and this
    # function would report a 100% overstatement in the regime where the
    # approximation is PERFECT.  Caught by its own small-angle-limit test.
    exact = float((wgt * 2.0 * np.sin(0.5 * theta) ** 2).sum())
    return float(1.0 - exact / approx)


def vface_gap_from_latitudes(gphit: np.ndarray,
                             gphiv: np.ndarray) -> tuple:
    """The pure arithmetic of the v-face zonal-width gap, split out to be tested.

    Returns ``(lat_mid, gap)`` where ``gap`` is POSITIVE when legoESM's v-face
    is zonally WIDER than NEMO's -- i.e. when the cosine at the tracer midpoint
    exceeds the cosine at NEMO's own v-point latitude.  Identically zero when
    the two latitude conventions agree, which is the control that says the gap
    is a convention difference and not an artifact of this formula.
    """
    lat_mid = 0.5 * (gphit[:-1] + gphit[1:])
    gap = np.cos(np.deg2rad(lat_mid)) / np.cos(np.deg2rad(gphiv[:-1])) - 1.0
    return lat_mid, gap


def vface_shape_test(vm: dict, bias_v: np.ndarray, vel_v: np.ndarray,
                     wetv: np.ndarray) -> dict:
    """Does the metric gap's LATITUDE SHAPE match the measured deficit's?

    A candidate whose coefficient is the right size at two rows is not yet a
    candidate: the size has to track across the basin, or the agreement at
    those two rows is denominator shopping.  Both fields are reduced the same
    way -- an RMS over each row's wet cells -- and the comparison is a row-wise
    correlation plus the location of each field's peak.
    """
    _, gap = vface_gap_from_latitudes(vm["_gphit"], vm["_gphiv"])
    ncol = gap.shape[1] // 2
    rows, meas, pred = [], [], []
    for j in range(bias_v.shape[0]):
        if not wetv[j].any() or j >= gap.shape[0]:
            continue
        rv = float(np.sqrt((vel_v[j][wetv[j]] ** 2).mean()))
        if rv <= 0.0:
            continue
        m = float(np.sqrt((bias_v[j][wetv[j]] ** 2).mean())) / rv
        p = float(abs(gap[j, ncol]))
        rows.append(j)
        meas.append(m)
        pred.append(p)
    meas_a, pred_a = np.array(meas), np.array(pred)
    corr = (float(np.corrcoef(pred_a, meas_a)[0, 1])
            if pred_a.std() > 0 and meas_a.std() > 0 else float("nan"))
    peak = int(rows[int(np.argmax(meas_a))])
    sample = [j for j in (1, 3, 65, 100, 129, 137, 169, 190, 196)
              if j in rows]
    ratios = [m / max(p_, 1e-300) for m, p_ in zip(meas, pred)]
    # THE FLAG.  Sections 3 and 4 raise one when their evidence fails; this
    # section narrated its own failure in prose and left the reader to notice.
    # A candidate whose latitude structure does not track the measurement is
    # NOT SUPPORTED by this test, however well it lands at two rows -- which is
    # exactly the two-row reading section 3 had to retract.
    supported = bool(abs(corr) > 0.5) if np.isfinite(corr) else False
    return {"n_rows": len(rows), "corr": corr,
            "SHAPE_SUPPORTS_CANDIDATE": supported,
            "SHAPE_TEST_VERDICT": ("SUPPORTED" if supported
                                   else "NOT SUPPORTED -- right order, wrong "
                                        "latitude structure"),
            "per_row_ratio_min": float(min(ratios)),
            "per_row_ratio_max": float(max(ratios)),
            "measured_peak_row": peak,
            "measured_peak": float(meas_a.max()),
            "rows": [{"j": j, "measured": meas[rows.index(j)],
                      "predicted": pred[rows.index(j)],
                      "ratio": (meas[rows.index(j)]
                                / max(pred[rows.index(j)], 1e-300))}
                     for j in sample]}


def vface_zonal_metric_gap(seqdump: str) -> dict:
    """legoESM's zonal width of the v-face against NEMO's own e1v.

    RAISED BY ADVERSARIAL REVIEW 2026-08-26 and verified here from first
    principles rather than relayed.  NEMO builds its DINO mesh isotropically --
    `e1v == e2v` bit-for-bit -- with both evaluated at NEMO's own v-point
    latitude `gphiv`.  legoESM's `nemo_isotropic` convention overrides the
    v-face MERIDIONAL width to match, but leaves the v-face ZONAL width on
    `R·cos(lat_v)·dlon` with `lat_v` taken as the arithmetic midpoint of the two
    adjacent tracer latitudes.  On DINO's stretched meridional grid that
    midpoint is NOT `gphiv`, so the two cosines differ.

    Why it matters here, and why it is section 6 rather than a footnote: this
    sits INSIDE the barotropic loop, so substituting NEMO's slow forcing does
    not remove it, and it feeds the metric-complete rotation coefficient that
    the card under test has active.  A rotation coefficient too large by a
    fraction makes the balanced wall-NORMAL velocity too small by the same
    fraction -- which is the sign and the shape of the meridional bias measured
    in section 2.

    Reported as a PREDICTION, not a verdict: the override arm that would settle
    ownership has not been run, and the pre-registered bars for it are in the
    accompanying document.
    """
    path = os.path.join(seqdump, "domain_cfg_out.nc")
    if not os.path.exists(path):
        return {"available": False, "reason": f"no domain_cfg_out.nc at {path}"}
    try:
        import netCDF4 as nc
    except ImportError:
        return {"available": False, "reason": "netCDF4 not importable"}
    d = nc.Dataset(path)
    e1v = np.array(d["e1v"][0])
    e2v = np.array(d["e2v"][0])
    gphit = np.array(d["gphit"][0])
    gphiv = np.array(d["gphiv"][0])

    lat_mid, gap = vface_gap_from_latitudes(gphit, gphiv)
    ncol = gap.shape[1] // 2
    rows = {int(j): float(gap[j, ncol])
            for j in (0, 1, 2, 50, 99, 145, 190, 195, 196)
            if j < gap.shape[0]}
    return {"available": True,
            "nemo_mesh_is_isotropic_e1v_eq_e2v": bool(
                np.abs(e1v - e2v).max() == 0.0),
            "nemo_e1v_minus_e2v_max_abs": float(np.abs(e1v - e2v).max()),
            "max_gphiv_minus_midpoint_deg": float(
                np.abs(gphiv[:-1] - lat_mid).max()),
            "relative_gap_by_row": rows,
            "relative_gap_max_abs": float(np.abs(gap).max()),
            "relative_gap_at_equator": float(
                np.abs(gap[np.argmin(np.abs(gphiv[:-1, ncol])), ncol])),
            "_gphit": gphit, "_gphiv": gphiv,
            "note": ("POSITIVE gap = legoESM's v-face zonally wider than "
                     "NEMO's. Prediction only; the override arm is unrun.")}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-dir", default=DEFAULT_MAP_DIR)
    ap.add_argument("--seqdump", default=os.environ.get(
        "DINO_NEMO_RUN_SEQDUMP", DEFAULT_SEQDUMP))
    ap.add_argument("--response", type=float, default=MEASURED_RESPONSE,
                    help="loop response to a steady forcing; measured by "
                         "substep_traj_compare.py (R_deposit)")
    ap.add_argument("--dt-s", type=float, default=MEASURED_DT_S)
    ap.add_argument("--vface-arm", default="none", choices=_VFACE_ARMS,
                    help="which v-face metric arm these maps are: 'none' (the "
                         "plain frozen-forcing arm), 'nemo' (NEMO's own e1v "
                         "substituted), or 'stagger' (the un-shifted array, "
                         "the registered staggering control). Every map in "
                         "the directory must carry this arm.")
    ap.add_argument("--coriolis-arm", default="none", choices=_CORIOLIS_ARMS,
                    help="which vertex-Coriolis arm these maps are: 'none', "
                         "'nemo' (NEMO's own ff_f substituted) or 'stagger' "
                         "(the un-shifted array, the registered staggering "
                         "control). Independent of --vface-arm: the #1455 "
                         "pair's JOINT arm sets BOTH to 'nemo'. Every map in "
                         "the directory must carry this arm.")
    ap.add_argument("--out", default=os.path.join(
        REPO, "results", "dino_1455", "baro_fixed_bias_wall_map.json"))
    args = ap.parse_args()

    print(provenance())
    st = constancy_selftest()
    print("\n=== SELF-TEST of the constancy statistic (known answers) ===")
    print(f"  constant field   : departure share {st['constant_departure_share']:.3e} "
          f"(want 0), corr_min {st['constant_corr_min']:+.4f} (want +1)")
    # The expected share for the alternating case is NOT ~1, and it is worth
    # deriving rather than eyeballing.  For n states with signs +,-,+,-,+ the
    # state mean is base/n; the largest departure is the one on a MINUS state,
    # |-base - base/n| = base(n+1)/n; so the share is (n+1)/n divided by 1/n,
    # i.e. exactly n+1 = 6.0 at five states.  Two earlier drafts got this wrong
    # in opposite directions -- one printed "(want ~1)" beside a printed 6.000,
    # the next asserted n-1 and aborted the probe.  Both were caught by this
    # gate, which is the argument for the gate.
    n_st = 5
    want_alt = float(n_st + 1)
    print(f"  alternating field: departure share "
          f"{st['alternating_departure_share']:.3f} (want {want_alt:.1f} at "
          f"{n_st} states), corr_min {st['alternating_corr_min']:+.4f} "
          f"(want -1)")
    # SystemExit, not assert: a bare assert vanishes under `python -O` and this
    # gate is the only thing standing between a broken statistic and a number.
    if not (st["constant_departure_share"] < 1e-12
            and st["constant_corr_min"] > 0.99
            and st["alternating_corr_min"] < -0.99
            and abs(st["alternating_departure_share"] - want_alt) < 0.05):
        raise SystemExit(
            "FATAL: the constancy statistic fails its own known answers "
            f"({st!r}); every number below would be unearned.")

    M = load_maps(args.map_dir)
    wetu, wetv = M["wetu"][0] > 0.5, M["wetv"][0] > 0.5
    wgt = M["wgt_primary"][0]
    prov = assert_map_provenance(M["_stamps"], args.seqdump, args.vface_arm,
                                 args.coriolis_arm)
    inv = wind_step_invariance(M["_stamps"])
    print("\n=== MAP PROVENANCE GATE ===")
    print(f"  v-face metric arm these maps are scored as: {args.vface_arm!r} "
          f"(checked per map against each map's own stamp)")
    print(f"  vertex-Coriolis arm these maps are scored as: "
          f"{args.coriolis_arm!r} (checked per map against each map's own "
          f"stamp)")
    print(f"  producer's OWN knobs, checked against their ON state: "
          f"{prov['producer_knobs']}")
    print(f"  inert knobs (belong to other probes; recorded, NOT a wind-on "
          f"witness): {prov['inert_knobs_recorded_not_checked']}")
    print(f"  legoESM wind-on directly stamped in the maps: "
          f"{prov['lego_wind_on_directly_stamped']}  <- the oracle side IS "
          f"witnessed (nonzero increment on every wet u face); this side "
          f"rests on the clock checks above, which are a proxy")
    print(f"  each state carries its own oracle dump; --seqdump matches one "
          f"of them: {prov['cli_seqdump_matches_map']}")
    if inv["max_relative_spread"] is None:
        print(f"  wind-increment step-invariance UNTESTED ({inv['note']})")
    else:
        print(f"  wind increment across {inv['n_dumps']} state dumps: max "
              f"relative spread {inv['max_relative_spread']:.2e} -> "
              f"step-invariant {inv['STEP_INVARIANT']}")
        if not inv["STEP_INVARIANT"]:
            raise SystemExit(
                "FATAL: the wind increment differs between the states' own "
                "oracle dumps, so comparing a single directory's increment "
                "against a five-state mean bias is not one experiment.")

    comps = {}
    for tag, key, wet, vkey, label in (
            ("u", "dU_sub", wetu, "nemo_Ubar_avg", "TANGENTIAL to the walls"),
            ("v", "dV_sub", wetv, "nemo_Vbar_avg", "WALL-NORMAL")):
        series, vel = M[key], M[vkey].mean(axis=0)
        bias = series.mean(axis=0)
        con = constancy(series, wet)
        j_s, j_n = wall_rows(wet)
        prof = row_profile(bias, wet)
        wi = wall_and_interior(bias, wet)
        # The claim is about the WALL ROWS, so the constancy backing it is
        # measured there too, not only basin-wide (review 2026-08-26: the
        # reduction must match the claim).
        wall_mask = np.zeros_like(wet)
        wall_mask[j_s] = wet[j_s]
        wall_mask[j_n] = wet[j_n]
        comps[tag] = {
            "label": label, "wall_rows": [j_s, j_n],
            "constancy": con,
            "constancy_at_wall_rows": constancy(series, wall_mask),
            "row_profile": prof,
            "interior_median_rms": wi["interior_median"],
            "wall_and_interior": wi,
            "wall_rms": {"south": wi["south"], "north": wi["north"]},
            "mirrored_pairs": mirrored_pairs(bias, vel, wet),
            "_bias": bias}

        print(f"\n=== [{tag}] {label} -- the state-CONSTANT in-loop bias ===")
        print("  per-state RMS   : "
              + ", ".join(f"{v:.4e}" for v in con["per_state_rms"]))
        print(f"  state-mean RMS  : {con['state_mean_rms']:.4e}")
        print("  departure RMS   : "
              + ", ".join(f"{v:.3e}" for v in con["departure_rms"]))
        print(f"  -> the field reproduces state to state to "
              f"{100 * con['max_departure_share']:.2f}% at worst; "
              f"min cross-state corr {con['cross_state_corr_min']:+.5f}")
        cw = comps[tag]["constancy_at_wall_rows"]
        print(f"  wall rows {j_s} / {j_n}: rms "
              f"{comps[tag]['wall_rms']['south']:.3e} (south) vs "
              f"{comps[tag]['wall_rms']['north']:.3e} (north); "
              f"interior median {comps[tag]['interior_median_rms']:.3e} "
              f"over {wi['n_interior_rows']} rows")
        print(f"  constancy RESTRICTED TO THE WALL ROWS (the reduction the "
              f"claim is about): worst departure "
              f"{100 * cw['max_departure_share']:.3f}%, min corr "
              f"{cw['cross_state_corr_min']:+.5f}")
        print("  mirrored wall pairs (d = rows from the wall):")
        print("    d | j_S  bias_S    | j_N  bias_N    | amp N/S  vel N/S | "
              "cell corr S / N  | FLAG")
        for p in comps[tag]["mirrored_pairs"][:6]:
            cs = p["cell_fit_south"]["corr"]
            cn = p["cell_fit_north"]["corr"]
            print(f"   {p['d']:2d} | {p['j_south']:3d} {p['bias_south']:.3e} | "
                  f"{p['j_north']:3d} {p['bias_north']:.3e} | "
                  f"{p['bias_amp_ratio_n_over_s']:7.2f} "
                  f"{p['vel_amp_ratio_n_over_s']:7.2f} | "
                  f"{cs:+.3f} / {cn:+.3f} | "
                  f"{'AMPLITUDE-ONLY' if p['AMPLITUDE_COINCIDENCE_FLAG'] else ''}")
        top = sorted((r for r in prof if r["rms"] is not None),
                     key=lambda r: -r["rms"])[:8]
        shares = {r["j"]: two_dx_share(bias, wet, r["j"]) for r in top}
        comps[tag]["two_dx_share_top_rows"] = shares
        print("  top rows by bias RMS (rms | signed | coherence | 2dx share):")
        for r in top:
            sh = shares[r["j"]]
            print(f"    j={r['j']:3d}  {r['rms']:.3e}  {r['signed']:+.3e}  "
                  f"{r['coherence']:.2f}   "
                  + ("n/a" if sh is None else f"{sh:.3f}"))

    print("\n=== WIND-STRESS PLACEMENT CANDIDATE ===")
    wind = wind_candidate(args.seqdump, comps["u"]["_bias"], comps["v"]["_bias"],
                          wetu, wetv, wgt, args.response, args.dt_s)
    print(f"  meridional wind increment EXACTLY zero everywhere: "
          f"{wind['v_increment_exactly_zero']} "
          f"(max |v| = {wind['v_increment_max_abs']:.3e})")
    print(f"  zonal increment: peak {wind['u_increment_peak']:.4e} at row "
          f"{wind['u_increment_peak_row']}; at the south wall "
          f"{wind['u_increment_at_south_wall']:.4e} "
          f"({wind['wall_vs_peak_suppression_south']:.0f}x below peak); "
          f"at the north wall {wind['u_increment_at_north_wall']:.4e} "
          f"({wind['wall_vs_peak_suppression_north']:.0f}x below peak)")
    print("  STRUCTURAL: this bias is measured in the arm where NEMO's OWN "
          "zu_frc/zv_frc\n              is substituted into legoESM's loop, so "
          "every difference in how\n              legoESM ASSEMBLES its slow "
          "forcing -- the wind term included --\n              is already "
          "removed by construction.")
    print(f"  rotated-bound kernel on the REAL saved weights: "
          f"{wind['rotated_bound_kernel_sum_w_k2_over_2']:.1f} "
          f"(the synthetic uniform-68 case used by the tests is 787.75 -- "
          f"they differ because the real window is uniform over only 45 of 68 "
          f"substeps, starting at 24)")
    _ang = wind["rotation_angle_rad_at_window_end"]
    _ov = wind["small_angle_overstatement_frac"]
    print(f"  SMALL-ANGLE CAVEAT: the rotation angle reaches "
          f"{max(_ang.values()):.2f} rad by window end, so the small-angle "
          f"step is used outside its stated regime. DERIVED overstatement, "
          f"as a fraction OF THE SMALL-ANGLE BOUND ITSELF "
          f"(1 - exact/approx, both averaged over this window's weights): "
          + ", ".join(f"{t} {100 * vv:.1f}%" for t, vv in sorted(_ov.items()))
          + " (the exact response is that much SMALLER, i.e. CONSERVATIVE), "
            "so this stays an upper bound.")
    _sc = wind["row_profile_shape_corr_with_bias"]
    print("  row-profile shape correlation with the bias (WHOLE-DOMAIN "
          "context, not a wall statistic): "
          + ("n/a" if _sc is None else f"{_sc:+.3f}"))
    for lbl, b in wind["bounds"].items():
        print(f"  [{lbl}, R={b['response']}, 100% of the wind term]")
        print(f"     ZONAL   bias/wind = "
              f"{b['ratio_bias_over_full_wind_south']:.2f}x south "
              f"({'excludes' if b['u_EXCLUDES_south'] else 'DOES NOT EXCLUDE'})"
              f", {b['ratio_bias_over_full_wind_north']:.2f}x north "
              f"({'excludes' if b['u_EXCLUDES_north'] else 'DOES NOT EXCLUDE'})")
        for tag in ("south", "north"):
            r = b["rotated_meridional"][tag]
            co = wind["rotated_forcing_colocation"][tag]
            # How much room is there above the 3x margin?  The northern
            # exclusion at R=1 clears by only ~1.3x, i.e. it would FLIP at a
            # modestly larger response factor.  Printing the headroom stops
            # "excludes" reading as comfortable when it is marginal.
            head = r["ratio_bias_over_rotated_wind"] / 3.0
            print(f"     ROTATED {tag:5s} v-bias/rotated-wind = "
                  f"{r['ratio_bias_over_rotated_wind']:.2f}x "
                  f"({'excludes' if r['EXCLUDES'] else 'DOES NOT EXCLUDE'}"
                  f", clears the 3x margin by {head:.2f}x)"
                  f"  [v response row {co['v_response_row']}, forcing from u "
                  f"row {co['u_forcing_row_used']} of "
                  f"{co['u_forcing_rows_considered']}]"
                  f"   [rotated deposit "
                  f"{r['rotated_v_deposit_from_FULL_wind_term']:.3e} vs "
                  f"measured {r['measured_v_bias']:.3e} m/s]")

    print("\n=== WALL GEOMETRY ===")
    geo = wall_geometry(args.seqdump)
    if geo.get("available"):
        print(f"  ({geo['grid']})")
        for tag, r in geo["rows"].items():
            print(f"  {tag:14s} j={r['j']:3d} lat={r['lat_deg_mean_wet']:+7.3f} "
                  f"n_wet={r['n_wet']} mean depth {r['mean_depth_m']:.1f} m "
                  f"e2t {r['e2t_m_mean_wet']:.1f} m")
        sym = geo["symmetry"]
        print(f"  |lat| differs by {sym['abs_lat_diff_deg']:.4f} deg, "
              f"depth ratio N/S {sym['depth_ratio_n_over_s']:.4f}, "
              f"e2t ratio N/S {sym['e2t_ratio_n_over_s']:.4f}")
        if sym["GEOMETRY_ASYMMETRY_FLAG"]:
            print(f"  *** GEOMETRY IS NOT SYMMETRIC: worst departure "
                  f"{100 * sym['worst_fractional_asymmetry']:.1f}% (column "
                  f"depth). A geometric driver scaling linearly with that "
                  f"could produce ~{sym['depth_ratio_n_over_s']:.2f}x of "
                  f"north/south contrast; the measured zonal bias contrast is "
                  f"11.7x, so the depth asymmetry does not account for it -- "
                  f"but 'symmetric' is the wrong word and is not used. ***")
        print(f"  {sym['note']}")
    else:
        print(f"  UNAVAILABLE: {geo['reason']}")

    print("\n=== V-FACE ZONAL METRIC (leading untested candidate) ===")
    vm = vface_zonal_metric_gap(args.seqdump)
    if vm.get("available"):
        print(f"  NEMO's mesh is isotropic (e1v == e2v bit-for-bit): "
              f"{vm['nemo_mesh_is_isotropic_e1v_eq_e2v']} "
              f"(max|e1v-e2v| = {vm['nemo_e1v_minus_e2v_max_abs']:.3e} m)")
        print(f"  NEMO's v-point latitude is NOT the tracer midpoint: they "
              f"differ by up to {vm['max_gphiv_minus_midpoint_deg']:.2e} deg")
        print("  relative gap (legoESM wider than NEMO), by row:")
        for j, g_ in sorted(vm["relative_gap_by_row"].items()):
            print(f"    j={j:3d}  {g_:+.4e}")
        print(f"  -> peaks at BOTH walls, ~zero at the equator "
              f"({vm['relative_gap_at_equator']:.2e}); max "
              f"{vm['relative_gap_max_abs']:.3e}")
        # THE SHAPE TEST, over the WHOLE basin and not only the two walls.
        # Quoting the candidate at the wall rows alone would be the same
        # denominator-shopping that this probe had to retract elsewhere: the
        # measured relative deficit turns out to PEAK in the interior, which
        # the candidate's own latitude shape does not reproduce.
        shp = vface_shape_test(vm, comps["v"]["_bias"],
                               M["nemo_Vbar_avg"].mean(axis=0), wetv)
        vm["shape_test"] = shp
        print("  PREDICTED gap vs MEASURED relative wall-normal deficit "
              "(|bias|rms / |Vbar|rms), per row:")
        for r in shp["rows"]:
            print(f"    j={r['j']:3d}  measured {r['measured']:.3e}  "
                  f"predicted {r['predicted']:.3e}  ratio "
                  f"{r['ratio']:.2f}")
        print(f"  row-wise correlation predicted vs measured over "
              f"{shp['n_rows']} rows: {shp['corr']:+.3f}")
        print(f"  measured deficit peaks at row {shp['measured_peak_row']} "
              f"({shp['measured_peak']:.3e}); the predicted gap peaks at the "
              f"WALLS.")
        print(f"  per-row ratio spans {shp['per_row_ratio_min']:.2f} to "
              f"{shp['per_row_ratio_max']:.1f}")
        print(f"  *** SHAPE TEST: {shp['SHAPE_TEST_VERDICT']} *** "
              f"(supports candidate: {shp['SHAPE_SUPPORTS_CANDIDATE']})")
        print("  THIS IS A PREDICTION, NOT A VERDICT: the override arm that "
              "would settle ownership has not been run.")
    else:
        print(f"  UNAVAILABLE: {vm['reason']}")

    for c in comps.values():
        c.pop("_bias", None)
    payload = {"provenance": provenance(), "selftest": st,
               "components": comps, "wind_candidate": wind, "geometry": geo,
               "map_provenance": prov, "wind_step_invariance": inv,
               "vface_zonal_metric": {k: v for k, v in vm.items()
                                     if not k.startswith("_")},
               "response_used": args.response, "dt_s_used": args.dt_s,
               "stamps": M["_stamps"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    print(f"\n[artifact] {args.out}")


if __name__ == "__main__":
    main()
