"""#1455 next-action 1: score the Coriolis PAIR's registered arm test.

FIVE directories, THREE registered arms.  The registered arms are E1V, F and
JOINT; BASE is the baseline they are all scored against and CTRLF is the
staggering control.  "Three-arm test" in the pre-registration means the three
substitution arms, and this file scores five directories to produce them.

This is a COMPARISON HARNESS, not a new measurement.  Every number it reports
comes out of the committed probe ``baro_fixed_bias_wall_map.py`` -- its map
loader (which refuses a hole in the five-state series), its provenance gate
(which refuses a mislabelled or mixed arm directory), and its ``constancy``
statistic (which is the state-constant reduction the verdict is registered on).
Nothing is re-derived here; what is added is the ARM AXIS, which that probe has
no notion of because it scores one directory at a time.

WHY A JOINT ARM AT ALL (docs/ocean/fidelity/dino_campaign_synthesis.md sec 3
item 1).  legoESM's vertex Coriolis and its v-face zonal metric are a
CANCELLING PAIR: their signed latitude profiles correlate at +1.000, they enter
the same EEN rotation coefficient ``e1v * f`` with opposite signs, and the arm
already run removed the smaller of the two.  Fixing one half of a cancelling
pair can make the metric WORSE, which this campaign has recorded three times.
The registered object is therefore the pair, and the verdict is registered on
the JOINT arm alone.  See PREREG_coriolis_pair.md, committed before any arm was
scored.

THE ARMS, and the reduction each is scored on, are declared here rather than
inferred from the directory names -- a directory whose maps carry a different
arm stamp aborts, it does not get relabelled.

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/coriolis_pair_arm_compare.py \\
      --arm-root results/dino_1455/arms

Writes ``<arm-root>/coriolis_pair_arm_compare.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))


def _load_probe():
    """Import the COMMITTED scorer as a module and use ITS functions."""
    path = os.path.join(_THIS_DIR, "baro_fixed_bias_wall_map.py")
    spec = importlib.util.spec_from_file_location("baro_fixed_bias_wall_map",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_probe()

#: (arm key, directory, expected v-face stamp, expected Coriolis stamp).  The
#: expected stamps are the gate: a directory that does not carry them aborts.
#:
#: WHAT THE CONTROL CONTROLS, stated because it is NOT one variable off the arm
#: it gates.  CTRLF is ("none", "stagger"): it feeds the UN-SHIFTED ff_f on the
#: unsubstituted-metric base.  What it establishes is that the loop RESPONDS to
#: the vertex-Coriolis array at all -- a property of the array's consumer, not
#: of the metric alongside it -- so arm F is what it directly gates and JOINT,
#: which feeds the same array through the same consumer, inherits it.  A
#: ("nemo", "stagger") control would be one variable off JOINT and is the
#: stricter design; it was not run, and that is a limitation of this table, not
#: a claim about it.
ARMS = (
    ("BASE",  "base",  "none", "none"),
    ("E1V",   "e1v",   "nemo", "none"),
    ("F",     "f",     "none", "nemo"),
    ("JOINT", "joint", "nemo", "nemo"),
    ("CTRLF", "ctrlf", "none", "stagger"),
)

#: The three zonally-coherent bands the residual actually decomposes into
#: (dino_wall_fixed_bias.md sec 2).  Rows are inclusive, on the v grid.
BANDS = (("southern interior 57-73", 57, 73),
         ("northern interior 121-153", 121, 153),
         ("northern lobe 185-197", 185, 197))

#: PRE-REGISTERED, in PREREG_coriolis_pair.md, before any arm was scored.
OWNER_BAR = 50.0        # % collapse, on BOTH the basin-wide and lobe reductions
REFUTED_BAR = 10.0      # % collapse
CONTROL_BAR = 50.0      # % ABOVE baseline the staggering control must reach


def _row_mask(wet: np.ndarray, j0: int, j1: int) -> np.ndarray:
    """``wet`` restricted to rows ``j0..j1`` inclusive.

    A band that selects NO wet cell is fatal, not empty.  ``constancy`` on an
    empty selection returns NaN, and NaN is False against every one of the
    verdict's comparisons -- so a band whose rows fell off the grid would sail
    through as "PARTIAL", indistinguishable from a real measurement.  The
    northern-lobe band carries half the registered verdict, so this is the
    difference between a verdict and a silent nothing.
    """
    m = np.zeros_like(wet)
    m[j0:j1 + 1] = wet[j0:j1 + 1]
    if not m.any():
        raise SystemExit(
            f"FATAL: the row band {j0}-{j1} selects no wet cell on a "
            f"{wet.shape[0]}-row grid. An empty band reduces to NaN, and NaN "
            "reads as PARTIAL in the verdict rule.")
    return m


def transport_direction(acc_w: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Unit vector along the SIGNED, area-weighted transport reduction.

    The deposit scalar this campaign quotes as a circumpolar-transport
    stand-in is a signed weighted sum, i.e. the projection of the residual
    field onto one direction.  Materialising that direction lets the question
    "how much of the residual even lives in the transport mode" be answered by
    a projection instead of by comparing two incommensurable reductions.
    """
    g = np.zeros_like(acc_w, dtype=float)
    g[wet] = acc_w[wet]
    n = float(np.linalg.norm(g[wet]))
    if n == 0.0:
        raise SystemExit("FATAL: the transport direction is identically zero "
                         "on the wet mask; a projection onto it is undefined.")
    return g / n


def transport_alignment(field: np.ndarray, direction: np.ndarray,
                        wet: np.ndarray) -> float:
    """|<field, direction>| / ||field||, on the wet mask.  Sign-blind: removing
    and adding the same mode are equally 'in' it."""
    f = field[wet]
    nf = float(np.linalg.norm(f))
    if nf == 0.0:
        return 0.0
    return float(abs(f @ direction[wet]) / nf)


def collapse_pct(arm: float, base: float) -> float:
    """Per-cent collapse of an arm's residual against the baseline's.

    Negative means the arm made it WORSE; the staggering control is scored on
    the negation of this, so a sign slip here would make a control that never
    fired look like it did.
    """
    return 100.0 * (1.0 - arm / max(base, 1e-300))


def control_pct_above_baseline(arm: float, base: float) -> float:
    """How far ABOVE the baseline a control arm's residual sits, in per cent."""
    return -collapse_pct(arm, base)


def verdict(basin: float, lobe: float) -> str:
    """The PRE-REGISTERED rule, and the ONLY place it is written down.

    Registered on the JOINT arm alone: OWNER needs BOTH the basin-wide and the
    northern-lobe reduction past the bar -- a candidate that collapses the
    basin while leaving the lobe alone is exactly the case the pair lesson says
    must not read as ownership.  A collapse past 100% pushed the residual
    through zero into the opposite sign, which is over-correction.
    """
    if not (np.isfinite(basin) and np.isfinite(lobe)):
        raise SystemExit(
            f"FATAL: a non-finite collapse reached the verdict rule "
            f"(basin={basin!r}, lobe={lobe!r}); every comparison below would "
            "be False and the rule would fall through to PARTIAL.")
    if basin > 100.0 or lobe > 100.0:
        return "OVERSHOOT (over-correction, not ownership)"
    if basin > OWNER_BAR and lobe > OWNER_BAR:
        return "OWNER"
    if basin < REFUTED_BAR and lobe < REFUTED_BAR:
        return "REFUTED"
    return "PARTIAL"


def reductions(maps: dict) -> dict:
    """Every registered reduction of ONE arm, all through ``M.constancy``."""
    wetv = maps["wetv"][0] > 0.5
    wetu = maps["wetu"][0] > 0.5
    j_s, j_n = M.wall_rows(wetv)
    out = {
        "wall_normal_basin": M.constancy(maps["dV_sub"], wetv),
        "wall_normal_wall_rows": M.constancy(
            maps["dV_sub"], _row_mask(wetv, j_s, j_s) | _row_mask(wetv, j_n, j_n)),
        "wall_normal_south_wall": M.constancy(maps["dV_sub"],
                                              _row_mask(wetv, j_s, j_s)),
        "wall_normal_north_wall": M.constancy(maps["dV_sub"],
                                              _row_mask(wetv, j_n, j_n)),
        "tangential_basin": M.constancy(maps["dU_sub"], wetu),
    }
    for name, j0, j1 in BANDS:
        out[f"wall_normal_band {name}"] = M.constancy(maps["dV_sub"],
                                                      _row_mask(wetv, j0, j1))
    out["_wall_rows"] = [int(j_s), int(j_n)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-root", default=os.path.join(
        _REPO, "results", "dino_1455", "arms"))
    ap.add_argument("--seqdump", default=os.environ.get(
        "DINO_NEMO_RUN_SEQDUMP", M.DEFAULT_SEQDUMP))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.arm_root,
                                        "coriolis_pair_arm_compare.json")

    # The instrument's own self-test FIRST.  A constancy statistic that cannot
    # tell a constant field from an alternating one would make every number
    # below unearned, and this probe's whole method is that statistic.
    st = M.constancy_selftest()
    if not (st["constant_departure_share"] < 1e-12
            and st["alternating_corr_min"] < -0.99):
        raise SystemExit(
            f"FATAL: the constancy statistic fails its own known answers "
            f"({st!r}); every number below would be unearned.")

    print("=" * 78)
    print("THE CORIOLIS PAIR -- registered three-arm test (#1455 action 1)")
    print("=" * 78)
    print(f"  bars, PRE-REGISTERED in PREREG_coriolis_pair.md: OWNER > "
          f"{OWNER_BAR:.0f}% on BOTH the basin-wide and the northern-lobe "
          f"reduction, REFUTED < {REFUTED_BAR:.0f}%, and the staggering "
          f"control must be > {CONTROL_BAR:.0f}% ABOVE baseline.")

    red, stamps = {}, {}
    for key, sub, vf, cor in ARMS:
        d = os.path.join(args.arm_root, sub, "maps")
        if not os.path.isdir(d):
            raise SystemExit(
                f"FATAL: arm {key} has no map directory at {d}. A missing arm "
                "is a hole in a registered test, not a dropped row.")
        maps = M.load_maps(d)
        # THE GATE.  A joint-arm map and an e1v-arm map are well-formed arrays
        # on an identical wet mask; only the stamp separates them, so the arm
        # this table CALLS a directory has to be the arm its maps SAY they are.
        prov = M.assert_map_provenance(maps["_stamps"], args.seqdump, vf, cor)
        red[key] = reductions(maps)
        stamps[key] = {"dir": d, "vface_arm": prov["vface_arm"],
                       "coriolis_arm": prov["coriolis_arm"],
                       "producer_knobs": prov["producer_knobs"]}
        print(f"  arm {key:6s} <- {d}  (stamped vface={prov['vface_arm']!r} "
              f"coriolis={prov['coriolis_arm']!r})")

    base = red["BASE"]
    # STATE-CONSTANCY, per arm, printed BEFORE any collapse: a collapse
    # comparing a fixed field against a wobbling one is not like for like.
    print("\n" + "-" * 78)
    print("IS EACH ARM'S RESIDUAL STILL A FIXED FIELD? (worst state departure "
          "/ min signed cross-state correlation)")
    print("-" * 78)
    for key, *_ in ARMS:
        c = red[key]["wall_normal_basin"]
        print(f"  {key:6s} worst departure {100 * c['max_departure_share']:.2f}%"
              f"   min corr {c['cross_state_corr_min']:+.5f}")

    keys = [k for k, *_ in ARMS]
    rows = [k for k in red["BASE"] if not k.startswith("_")]
    print("\n" + "-" * 78)
    print("THE COLLAPSE TABLE -- state-constant RMS, and % collapse vs BASE")
    print("-" * 78)
    hdr = f"  {'reduction':34s}" + "".join(f"{k:>12s}" for k in keys)
    print(hdr)
    table = {}
    for r in rows:
        vals = {k: red[k][r]["state_mean_rms"] for k in keys}
        line = f"  {r[:34]:34s}" + "".join(f"{vals[k]:12.4e}" for k in keys)
        print(line)
        coll = {k: collapse_pct(vals[k], vals["BASE"]) for k in keys}
        print(f"  {'  -> % collapse vs BASE':34s}"
              + "".join(f"{coll[k]:+11.1f}%" for k in keys))
        table[r] = {"state_mean_rms": vals, "collapse_pct": coll}

    # ADDITIVITY -- printed for continuity with the registration, and it does
    # NOT carry the pair claim.  The premise this block used to state ("a
    # cancelling pair predicts the joint arm EXCEEDS the sum of the halves") is
    # UNSOUND and is withdrawn: the two perturbation FIELDS superpose exactly
    # (measured at ~3.5e-06 relative), so `joint - sum` in PER CENT is a
    # second-order norm-geometry effect fully determined by that superposition
    # and carries no information about cancellation either way.  The evidence
    # against the pair is the field-level superposition, not these points.
    print("\n" + "-" * 78)
    print("ADDITIVITY (continuity with the registration; the points below are")
    print("NOT evidence about the pair -- see the field-level superposition)")
    print("-" * 78)
    for r in rows:
        c = table[r]["collapse_pct"]
        s = c["E1V"] + c["F"]
        print(f"  {r[:34]:34s} E1V {c['E1V']:+7.1f}% + F {c['F']:+7.1f}% = "
              f"{s:+7.1f}%   vs JOINT {c['JOINT']:+7.1f}%   "
              f"(joint - sum = {c['JOINT'] - s:+7.1f} points)")

    # THE CONTROL.  If it does not fire the loop is insensitive to this array
    # and NO verdict may be issued from the correct arm either.
    _cw = table["wall_normal_basin"]["state_mean_rms"]
    ctrl_worse_pct = control_pct_above_baseline(_cw["CTRLF"], _cw["BASE"])
    control_fired = ctrl_worse_pct > CONTROL_BAR
    print("\n" + "-" * 78)
    print("THE REGISTERED STAGGERING CONTROL")
    print("-" * 78)
    print(f"  un-shifted ff_f: wall-normal basin residual is "
          f"{ctrl_worse_pct:+.1f}% ABOVE baseline "
          f"(bar: > {CONTROL_BAR:.0f}%)  -> "
          f"{'FIRES' if control_fired else 'DOES NOT FIRE'}")
    # THE CONTROL'S OWN AMPLITUDE, printed next to its result, because a
    # control that fires at a much larger perturbation than the candidate does
    # NOT establish that the loop resolves the candidate.  A one-row shift of
    # ff_f on this stretched mesh is ~1.28e-02 relative; the candidate's own
    # gap is ~5.48e-05.  So the control demonstrates sensitivity at 234x the
    # signal, and it is arm F's OWN measured response -- reported in the table
    # above -- that demonstrates sensitivity at the signal's own amplitude.
    # THE CANDIDATE'S AMPLITUDE DEPENDS ON WHICH TREE THIS IS.  On the
    # two-Earth tree the row-map median gap was 5.480e-05 (rotation rate +
    # convention); with one rate it is 3.902e-05 (convention alone), measured
    # by coriolis_omega_routing_audit.py on each tree.  Quoting the old number
    # after the fix UNDERSTATES how much larger the control is than the signal,
    # i.e. it errs in the flattering direction (adversarial review of the
    # scoring).  Both are printed so the ratio can never be read off the wrong
    # one.
    _CTRL_AMP = 1.281e-02          # coeff-ok: one-row shift of ff_f, measured
    for _tag, _sig in (("two-Earth tree (rate + convention)", 5.480e-05),
                       ("one-Earth tree (convention alone)", 3.902e-05)):
        print(f"  control amplitude {_CTRL_AMP:.3e} vs the candidate's "
              f"{_sig:.3e} on the {_tag}: {_CTRL_AMP / _sig:.0f}x")
    print(f"  -> the control licenses 'the loop READS this array', NOT 'the "
          f"loop RESOLVES the candidate's amplitude'. Arm F's own measured "
          f"response is the only evidence for the latter.")

    # THE VERDICT, BUILT FROM THE MEASURED VALUES, never hardcoded, and
    # registered on the JOINT arm alone.
    jb = table["wall_normal_basin"]["collapse_pct"]["JOINT"]
    jl = table["wall_normal_band northern lobe 185-197"]["collapse_pct"]["JOINT"]

    got = verdict(jb, jl)
    # D3: the VOID is applied to the verdict STRING, not carried beside it in a
    # separate key.  A reader quoting `verdict` out of the JSON must not be
    # able to get "OWNER" from a run whose control never fired -- the
    # pre-registration forbids issuing a verdict at all in that case.
    if not control_fired:
        got = f"VOID (staggering control did not fire) [would have read {got}]"
    print("\n" + "=" * 78)
    print("THE REGISTERED VERDICT -- on the JOINT arm, from the measured "
          "values")
    print("=" * 78)
    print(f"  JOINT basin-wide collapse    {jb:+.1f}%   (OWNER bar "
          f"> {OWNER_BAR:.0f}%)")
    print(f"  JOINT northern-lobe collapse {jl:+.1f}%   (OWNER bar "
          f"> {OWNER_BAR:.0f}%)")
    print(f"  VERDICT: {got}")
    if not control_fired:
        print("  ^ THE CONTROL DID NOT FIRE, so this verdict is VOID: the "
              "loop's insensitivity to the array would produce the same "
              "collapse for a candidate that is simply not there.")
    print(f"  90-day gate CONDITION (pre-registered): run it only if the "
          f"joint arm clears {OWNER_BAR:.0f}% -> "
          f"{'RUN IT' if (jb > OWNER_BAR and control_fired) else 'DO NOT RUN IT'}")

    out = {
        "provenance": M.provenance(),
        "bars": {"owner_pct": OWNER_BAR, "refuted_pct": REFUTED_BAR,
                 "control_pct_above_baseline": CONTROL_BAR},
        "arms": stamps,
        "table": table,
        "control_pct_above_baseline": ctrl_worse_pct,
        "control_fired": bool(control_fired),
        "joint_basin_collapse_pct": jb,
        "joint_northern_lobe_collapse_pct": jl,
        "verdict": got,
        "verdict_void_control_did_not_fire": bool(not control_fired),
        "run_the_90d_gate": bool(jb > OWNER_BAR and control_fired),
        "constancy_selftest": st,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
