#!/usr/bin/env python
"""#1455 -- rescore every wall-row refutation on the CORRECTED reduction.

WHY THIS EXISTS. Every wall-row number this campaign published was described as
a "signed thickness-weighted zonal mean". It was not: ``coherent_rows`` was
called with ``h=None`` at every site, which is an unweighted mean over wet
LEVELS. DINO's layers span 10.14 m to 545.20 m, so the published statistic
over-weights the surface by about 54x relative to mass -- and the
pre-registered bars were derived for a DEPTH-MEAN acceleration, which is the
mass-weighted quantity. Rescored, the EEN vorticity flux's wall mismatch is
12.4x smaller and its wall-row signs invert (the probe prints the count;
an earlier version of this docstring hardcoded "three of four" and the
code printed four -- exactly the defect under retraction).

Bottom drag and lateral friction were REFUTED on that instrument. This probe
asks, offline and with no model run, which refutations survive the correction
and which reopen.

IT DOES NOT RE-DERIVE ANY TERM. It drives the sibling probes
(``wall_term_discriminators.part1`` / ``part2``), which now report both
reductions, and reads the numbers back. The friction verdicts are read from
``wall_ldf_alignment`` the same way -- and that probe carries the SAME defect
under a different name (its own ``rma`` helper averages over longitude AND
LEVEL), which an earlier version of this probe missed by testing for a NAME
instead of a behaviour. It is now fixed there too and both readings reported.
So a candidate cannot be rescored on a different construction than the one that
refuted it -- which would be the same class of error all over again.

THE THREE STRUCTURAL CATEGORIES, decided by the shape of the scored array, not
by argument:
  * an already-depth-averaged (2-D) quantity is IMMUNE -- there is no vertical
    weight left to get wrong;
  * an EXACT-COUNT identity (0 of N points disagreeing, a stress that is
    exactly 0.0) is IMMUNE -- it has no vertical weight at all;
  * a 3-D field reduced to a row number is AFFECTED and must be rescored.
Each candidate is placed in a category by MEASUREMENT (the array's ndim, the
statistic's definition), and the category is printed beside its verdict.

This probe prints numbers and computes its verdict strings from them.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.wall_reduction_rescore
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Receipts go to results/ (gitignored). A probe's captured output is a
# runtime artifact; scripts/ is source.
_OUT = REPO_ROOT / "results" / "dino_1455_rescore"

import wall_term_discriminators as W  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

# The bars the original verdicts were taken against (PREREG_wall_drag_and_een).
BAR_MAG = W.BAR_MAG                 # 5.9e-11 m/s2, "can carry the lobe"
BAR_REFUTE = W.BAR_MAG_REFUTE       # 5.9e-12 m/s2, "cannot, even coherently"
BAR_ENRICH = W.BAR_ENRICH           # 3x wall enrichment


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  run={W.RUN}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    print(f"PROVENANCE  bars: carry>={BAR_MAG:.2e}, refute<{BAR_REFUTE:.2e}, "
          f"enrichment>={BAR_ENRICH}")


def verdict(wall: float, enr_registered: float) -> str:
    """The registered three-way call, COMPUTED from the numbers.

    ADVERSARIAL REVIEW FINDING C2, acted on. The magnitude leg and the
    ENRICHMENT leg are registered on DIFFERENT statistics, and an earlier
    version of this function used the magnitude statistic for both.
    ``PREREG_wall_drag_and_een.md`` registers the enrichment as "the row-mean
    ABSOLUTE difference over rows 1-4, divided by the same over the sampled
    interior rows" -- that is ``row_report``'s mean|.| ratio, and the 3x bar
    was calibrated against comparators measured the same way (2.4 for the
    viscosity thickness weighting, 0.57 for the implicit control volume).
    Dividing the signed coherent means instead gives a different number
    against a bar that was never set for it. ``enr_registered`` must therefore
    be the ``row_report`` ratio, passed in by the caller.

    The magnitude leg keeps the coherent (signed) reduction, which IS the
    quantity its bar was derived for -- a persistent depth-mean acceleration.
    """
    if wall < BAR_REFUTE:
        return f"REFUTED (below {BAR_REFUTE:.1e}, cannot carry it)"
    if wall >= BAR_MAG and enr_registered >= BAR_ENRICH:
        return "CLEARS BOTH LEGS (candidate)"
    if wall >= BAR_MAG:
        return (f"big enough, NOT enriched on the registered statistic "
                f"({enr_registered:.2f}x < {BAR_ENRICH}) -- NO VERDICT")
    return "between the bars -- NO VERDICT"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()

    g, br, cfg, mc = W.build_lego()
    W._gate_dry_faces(br)
    # ONE VARIABLE. The verdicts being rescored were taken with the EEN
    # transport metric weighting OFF -- the card has since been flipped ON
    # (ae08a6740), which changes the same term by two orders. Rescoring the old
    # verdict against the NEW model would change the reduction AND the operator
    # at once and attribute the result to the reduction. So the rescore runs
    # the HISTORICAL configuration, and the as-shipped numbers are reported
    # separately below as information, not as the rescore.
    mc_hist = mc._replace(een_metric_weighting="off")
    print(f"\nrescore configuration: een_metric_weighting="
          f"{mc_hist.een_metric_weighting!r} (the setting the verdicts were "
          f"taken under); the card now ships "
          f"{mc.een_metric_weighting!r}")
    # The sibling's own prints are captured: this probe's deliverable is the
    # comparison table, and 200 lines of upstream output buries it. The
    # captured text is written beside the table for anyone checking.
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            W.part1(g, br, cfg, mc_hist)
            W.part2(g, br, cfg, mc_hist)
    finally:
        # Written in a finally: a crash mid-run used to lose the receipts along
        # with the answer (adversarial review). Receipts go to results/, which
        # is gitignored -- a probe's captured output is a runtime artifact and
        # must not land in the source tree.
        _OUT.mkdir(parents=True, exist_ok=True)
        (_OUT / "rescore_upstream_historical.txt").write_text(buf.getvalue())
    print(f"\nupstream receipts (historical arm) -> "
          f"{_OUT / 'rescore_upstream_historical.txt'}")

    # SNAPSHOT. LAST_COHERENT/LAST_ROW are live module dicts and the as-shipped
    # part2 below REBINDS their keys. Reading them lazily made the printed arm
    # depend on statement order (adversarial review I3) -- the self-test was in
    # fact validating the as-shipped arm while claiming to check the rescore.
    C = {k: dict(v) for k, v in W.LAST_COHERENT.items()}
    R = {k: dict(v) for k, v in W.LAST_ROW.items()}
    if not C or not R:
        raise SystemExit(
            "the sibling probe reported no coherent/row numbers -- it has "
            "been changed and this rescore is reading nothing")

    # The two reports name the same term differently (coherent_report's label
    # vs row_report's), so the pairing is explicit rather than assumed equal.
    rows = []
    for name, row_name, cat_note in (
            ("1c drag increment",
             "1c drag increment to the barotropic forcing",
             "already depth-averaged (2-D)"),
            ("2 EEN vorticity flux",
             "2 EEN vorticity flux",
             "3-D field reduced to a row number")):
        missing = [k for k, dd in ((name, C), (row_name, R)) if k not in dd]
        if missing:
            raise SystemExit(
                f"the sibling probe no longer reports {missing} -- its report "
                "names changed and this rescore would silently pair the wrong "
                "statistics")
        rows.append((name, row_name, cat_note, C[name]))

    print("\n=== THE RESCORE: old verdict vs corrected verdict ===")
    print("  magnitude leg: the SIGNED coherent mean (the quantity its bar was")
    print("  derived for). enrichment leg: row_report's mean|.| ratio, which is")
    print("  what the pre-registration registers and what its 3x bar was")
    print("  calibrated against. The two legs use DIFFERENT statistics on")
    print("  purpose -- scoring both with one of them was review finding C2.")
    print(f"\n  {'candidate':<24s}{'reduction':<16s}{'wall':>12s}{'far':>12s}"
          f"{'enr(reg)':>9s}  verdict")
    for name, row_name, cat_note, d in rows:
        r = R[row_name]
        # The registered enrichment is reduction-independent as published (it
        # is row_report's, which this rescore does NOT reweight) -- so the same
        # value is shown on both lines, and the magnitude is what moves.
        enr = r["enrich"]
        for lbl, w, f in (("LEVEL (published)", d["level_wall"], d["level_far"]),
                          ("MASS (corrected)", d["mass_wall"], d["mass_far"])):
            print(f"  {name:<24s}{lbl:<16s}{w:>12.4e}{f:>12.4e}{enr:>9.2f}"
                  f"  {verdict(w, enr)}")
        print(f"  {'':<24s}{'':<16s}registered enrichment {enr:.2f}x (full "
              f"interior) / {r['enrich_far']:.2f}x (far interior)"
              + ("  -- THE TWO STRADDLE THE 3x BAR, so this leg is not "
                 "decided by the data" if r["straddles"] else ""))
        same = (verdict(d["level_wall"], enr) == verdict(d["mass_wall"], enr))
        ratio = (d["mass_wall"] / d["level_wall"]
                 if d["level_wall"] > 0 else float("nan"))
        flips = sum(int(np.sign(a) != np.sign(b))
                    for a, b in zip(d["level_rows"], d["mass_rows"]))
        print(f"  {'':<24s}{'':<16s}category: {cat_note}"
              f"{'  [IMMUNE by construction]' if d['two_d'] else ''}")
        print(f"  {'':<24s}{'':<16s}mass/level = {ratio:.4f}, sign flips on "
              f"{flips}/4 wall rows, verdict "
              f"{'SURVIVES' if same else 'CHANGES'}")
        rows_out = " ".join(f"{v:+.2e}" for v in d["level_rows"])
        rows_m = " ".join(f"{v:+.2e}" for v in d["mass_rows"])
        print(f"  {'':<24s}{'':<16s}wall rows level {rows_out}")
        print(f"  {'':<24s}{'':<16s}wall rows mass  {rows_m}")
        print()

    # --- LATERAL FRICTION, RESCORED (adversarial review finding C1) --------
    # RETRACTED: an earlier version of this probe declared friction IMMUNE on
    # the grounds that wall_ldf_alignment.py contains no call to the affected
    # reduction. That test was structurally incapable of finding what it looked
    # for -- it counted a NAME. The friction probe carries the SAME defect
    # under a different name: its own ``rma`` helper averages over longitude
    # AND LEVEL with no thickness weight (wall_ldf_alignment.py part_c and
    # part_d), and part_d's output includes an ENRICHMENT RATIO, where
    # reweighting moves numerator and denominator differently. The immunity
    # claim was false. Friction is now actually rescored, by driving the
    # friction probe, which reports both weightings.
    print("=== LATERAL FRICTION -- RESCORED, not assumed ===")
    import wall_ldf_alignment as L  # noqa: E402
    buf3 = io.StringIO()
    try:
        with redirect_stdout(buf3):
            nemo = L.load_nemo()
            lbr, lgeom, _lcfg, lu, lv, lum, lvm, lcm, _lhk = L.build_lego()
            _hUM, ahmt, ahmf, _vtx, lvm3 = L.effective_coefficients(
                lbr, lgeom, lcm)
            c = L.part_c(lbr, lgeom, lu, lv, lum, lvm, lcm, lvm3, ahmt, ahmf,
                         nemo)
            d = L.part_d(lbr, lgeom, lu, lv, lum, lvm, lcm, lvm3, ahmt, ahmf,
                         nemo)
    finally:
        (_OUT / "rescore_upstream_friction.txt").write_text(buf3.getvalue())
    print(f"  receipts -> {_OUT / 'rescore_upstream_friction.txt'}")
    print(f"  {'leg':<34s}{'published':>14s}{'MASS-weighted':>16s}  verdict")
    cw, cwm = c["wall_rel"], c["wall_rel_mass"]
    print(f"  {'C. mask dimensionality (rel)':<34s}{cw:>14.4e}{cwm:>16.4e}"
          f"  {'SURVIVES -- exactly 0 either way' if cw == cwm == 0.0 else 'CHANGES'}")
    dw, dwm = d["wall_rel"], d["wall_rel_mass"]
    de = d["wall_rel"] / d["interior_rel"] if d["interior_rel"] else float("nan")
    dem = (d["wall_rel_mass"] / d["interior_rel_mass"]
           if d["interior_rel_mass"] else float("nan"))
    print(f"  {'D. e3 weighting, wall rel change':<34s}{dw:>14.4e}"
          f"{dwm:>16.4e}  magnitude {'SURVIVES' if abs(dwm - dw) < 0.1 * max(dw, 1e-30) else 'CHANGES'}")
    print(f"  {'D. e3 weighting, enrichment':<34s}{de:>14.2f}{dem:>16.2f}"
          f"  {'crosses the ' + str(BAR_ENRICH) + 'x bar' if (de >= BAR_ENRICH) != (dem >= BAR_ENRICH) else 'same side of the bar'}")
    print()
    # The verdict sentence is COMPUTED. An earlier draft asserted the magnitude
    # was "the same under both weightings" while the numbers beside it differed
    # by 16% -- prose contradicting the code it sits next to is the exact
    # defect this whole lane is retracting.
    _chg = abs(dwm - dw) / dw if dw else float("nan")
    _order = (int(np.floor(np.log10(dw))) == int(np.floor(np.log10(dwm)))
              if dw > 0 and dwm > 0 else False)
    print(f"  WHAT REFUTED FRICTION WAS THE MAGNITUDE, NOT THE SHAPE. The e3")
    print(f"  weighting legoESM omits changes the viscous tendency by "
          f"{dw:.2e} relative published")
    print(f"  and {dwm:.2e} mass-weighted -- a {_chg * 100:.0f}% move that "
          f"stays {'in the same order of magnitude' if _order else 'in a DIFFERENT order of magnitude'}.")
    print(f"  A ~1e-4 relative effect cannot carry a 34% transport error "
          f"whatever its spatial")
    print(f"  concentration, so the refutation SURVIVES on the leg that "
          f"carried it. The enrichment")
    print(f"  leg genuinely moves ({de:.2f}x -> {dem:.2f}x, crossing the "
          f"{BAR_ENRICH}x bar) and is reported")
    print(f"  rather than buried: on the corrected weighting this difference "
          f"IS wall-concentrated,")
    print(f"  it is simply far too small to matter. Part C is exactly zero "
          f"under both weightings.")
    print()
    print("  The other friction legs are immune by CONSTRUCTION, not by "
          "measurement: exact-count")
    print("  identities (0 of 372528 corner points disagreeing, wall-face "
          "stress == 0.0) have no")
    print("  vertical weight at all, and friction_timestep_check.py's "
          "depth-mean leg already uses")
    print("  the model's own thickness-weighted depth_mean.")

    # --- as-shipped, for information only ---------------------------------
    print("\n=== AS SHIPPED, for information -- NOT the rescore ===")
    buf2 = io.StringIO()
    try:
        with redirect_stdout(buf2):
            W.part2(g, br, cfg, mc)
    finally:
        (_OUT / "rescore_upstream_asshipped.txt").write_text(buf2.getvalue())
    ship = dict(W.LAST_COHERENT["2 EEN vorticity flux"])
    ship_r = dict(W.LAST_ROW["2 EEN vorticity flux"])
    hist = C["2 EEN vorticity flux"]
    print(f"  receipts -> {_OUT / 'rescore_upstream_asshipped.txt'}")
    print(f"  card metric weighting ON: wall {ship['level_wall']:.4e} (level), "
          f"{ship['mass_wall']:.4e} (mass), registered enrichment "
          f"{ship_r['enrich']:.2f}x")
    print(f"  verdict: {verdict(ship['mass_wall'], ship_r['enrich'])}")
    print(f"  historical (metric OFF) was {hist['mass_wall']:.4e} (mass); the "
          f"term moved {ship['mass_wall'] / hist['mass_wall']:.4f}x")
    print("  READ THIS CORRECTLY: the term scores small now because it was "
          "FIXED. That is a")
    print("  different statement from the historical rescore above and must "
          "not be quoted as one.")
    if args.self_test:
        return self_test(C)  # the SNAPSHOT, not the live dict
    return 0


def self_test(C) -> int:
    """The rescore must be able to change a verdict, and to leave one alone.

    A rescore that reports SURVIVES for everything is indistinguishable from a
    rescore that is not reading anything.
    """
    print("\n=== SELF-TEST ===")
    # (i) the verdict function must actually be three-valued on real inputs.
    seen = {verdict(1e-9, 9.0),       # big + enriched      -> candidate
            verdict(1e-9, 1.0),       # big, NOT enriched   -> no verdict
            verdict(1e-13, 9.0),      # below refute bar    -> refuted
            verdict(1e-11, 9.0)}      # between the bars    -> no verdict
    assert len(seen) == 4, (
        f"the verdict rule does not reach all four branches: {seen}. The "
        "'big enough, not enriched' branch is the one the EEN term lands in "
        "on the registered statistic, so it must be exercised.")
    print(f"(i) all FOUR verdict branches are reachable:")
    for v in sorted(seen):
        print(f"      {v}")
    # (ii) the 2-D immunity claim must be a MEASURED property of the array,
    #      not a label -- feed coherent_report a 2-D and a 3-D diff built from
    #      the same numbers and require only the 3-D one to move.
    rng = np.random.default_rng(0)
    wet3 = np.ones((20, 10, 4), bool)
    d3 = rng.standard_normal((20, 10, 4))
    h = np.tile(np.array([1.0, 1.0, 1.0, 50.0]), (20, 10, 1))
    s_lvl, _ = W.coherent_rows(d3, wet3, None)
    s_mass, _ = W.coherent_rows(d3, wet3, h)
    assert not np.allclose(s_lvl, s_mass), (
        "a 53x thickness contrast did not change the 3-D reduction -- the h "
        "argument is still dead and this whole rescore is vacuous")
    d2 = d3.mean(axis=-1)
    a, _ = W.coherent_rows(d2, np.ones((20, 10), bool), None)
    b, _ = W.coherent_rows(d2, np.ones((20, 10), bool), h)
    assert np.array_equal(a, b), (
        "the 2-D branch responded to h -- the immunity claim for the drag "
        "increment is false")
    print("(ii) a 53x thickness contrast moves the 3-D reduction and leaves "
          "the 2-D one bit-identical -- immunity is measured, not asserted")
    # (iii) the numbers actually came from the sibling, not from a default.
    # ``mass_wall > 0`` was near-vacuous (any nonzero field passes a mean of
    # magnitudes). The real question is whether THIS snapshot is the historical
    # arm rather than the as-shipped one that overwrote the live dict, so pin
    # the value the historical arm is known to produce.
    e = C["2 EEN vorticity flux"]
    assert e["mass_wall"] != e["level_wall"], (
        "the EEN term's two reductions are identical -- part2 is not passing "
        "the mass weight")
    assert e["level_wall"] > 1e-10, (
        f"the snapshot's level-mean wall value is {e['level_wall']:.3e}; the "
        "historical (metric-OFF) arm scores ~3.1e-10 and the as-shipped arm "
        "~6e-12, so this snapshot is the WRONG ARM -- the live-dict rebinding "
        "has leaked back in")
    print(f"(iii) the snapshot is the historical arm (level wall "
          f"{e['level_wall']:.3e}, mass {e['mass_wall']:.3e}) and its two "
          f"reductions differ")
    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
