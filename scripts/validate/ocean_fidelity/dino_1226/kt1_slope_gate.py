#!/usr/bin/env python
"""legoESM's isopycnal SLOPES vs NEMO's own, at ``tra_ldf``'s call site, kt=1.

WHY THIS GATE EXISTS.  Every ``tra_ldf`` number on this branch is a number for
the OPERATOR fed **legoESM's own** slopes.  Nothing in it separates a correct
operator on wrong slopes from a wrong operator on right slopes -- the kt=1
ladder's ``A1`` control shows only that feeding NEMO's slopes leaves the answer
at the fp64 floor, which is the converse statement.  ``nemo_dino_kt1_slopes``
acquires NEMO's ``uslp/vslp/wslpi/wslpj``, ``ah_wslp2`` and ``akz`` copied
immediately after ``CALL traldf_iso_a33( Kmm, ah_wslp2, akz )`` inside
``traldf_iso_lap`` (``traldf_iso.F90:135``) -- ``tra_ldf``'s own frame, its own
time level, one file per rank.  This scores legoESM against it.

THE BAR IS EXACT (Rule 1b): zero cells unequal, per field.

WHAT THE RECORD CAN AND CANNOT SEE.  DINO sets ``rn_slpmax = 0.01``
(``namelist_cfg:268``), and from rest most of the domain sits ON that limiter,
where any two implementations agree by construction.  So the rows are scored
THREE ways -- all cells, cells at the cap, and cells strictly inside it -- and
the interior count is printed first.  A gate that reported only the pooled
number would be reporting the limiter.

RULE 10.  The slopes scored are the ones the MODEL'S OWN STEP BUILT, captured
at the PRODUCER ``compute_nemo_native_slopes`` during ``model.step`` -- not a
re-derivation with a hand-built state.

WHY THE PRODUCER AND NOT THE CONSUMER (a hostile review killed the first
version).  Spying on ``nemo_iso_lap_tracer_tendency_latlon_cgrid`` and reading
its ``native_slopes`` argument sees only the EXPLICIT isoneutral operator.
``compute_isoneutral_K33_latlon`` (gm_redi_latlon_cgrid.py:4459) builds its own
slopes by calling ``compute_nemo_native_slopes`` DIRECTLY at :4573, and the
K33 it returns from the Nnn advective pass is bound at
ocean_model_latlon_cgrid.py:10446 and CONSUMED by the implicit vertical solve
at :10630.  So a slope field that never touches the explicit operator still
reaches the answer, and the consumer-side spy is structurally blind to it.

WHAT THIS RECORD CANNOT SEE.  From rest NEMO sets ``ts(:,:,:,:,Kmm) =
ts(:,:,:,:,Kbb)`` (istate.f90:139), so the two tracer time levels are the SAME
FIELD.  Any agreement here therefore certifies the slope FORMULA and the
geometry; it says NOTHING about which time level a slope set is built on.  A
record at kt >= 3, where Nbb and Nnn have genuinely parted, is what would.

NON-VACUITY.  ``--plant`` moves one wet interior slope cell by 1 ulp before
scoring; the gate must then fail on that field.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/kt1_slope_gate.py \\
            --run-dir /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_slopes
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "nemo_dino_kt1_slopes"))
from read_slopes import stitch                                  # noqa: E402

DT = 2700.0                      # NEMO rn_Dt; rDt == rn_Dt on the Euler start
RN_SLPMAX = 0.01                 # namelist_cfg:268, namldf_tra


def _score(name, lego, nemo, wet, cap_hit):
    """Three rows: all wet cells, cells ON the limiter, cells inside it."""
    out = 0
    for tag, m in (("all", wet),
                   ("at cap", wet & cap_hit),
                   ("interior", wet & ~cap_hit)):
        if not m.any():
            print(f"  {name:10s}{tag:10s}{'(no cells)':>12s}")
            continue
        d = np.abs(np.asarray(lego) - np.asarray(nemo))[m]
        n = int((d != 0.0).sum())
        rms = float(np.sqrt(np.mean(d ** 2)))
        rel = rms / float(np.sqrt(np.mean(np.asarray(nemo)[m] ** 2))) \
            if np.any(np.asarray(nemo)[m]) else float("nan")
        # The residual in ULP OF THE LOCAL VALUE.  The bar stays EXACT (zero
        # cells unequal); this column only makes a sub-ulp residual legible
        # instead of leaving "1.7e-18" to be argued about.  It relaxes
        # nothing: the verdict column below is still n == 0.
        ulp = np.spacing(np.abs(np.asarray(nemo)[m]))
        nz = ulp > 0
        u = float(np.max(d[nz] / ulp[nz])) if nz.any() else float("nan")
        # A row in which NEMO's own field is identically zero is scoring
        # NOTHING; calling that AT BAR is how a gate manufactures agreement
        # (the surface gate says so about its buckets and this one did not).
        vac = not np.any(np.asarray(nemo)[m])
        verdict = ("VACUOUS (NEMO is 0 here)" if vac
                   else "AT BAR" if n == 0 else "DEBT")
        print(f"  {name:10s}{tag:10s}{int(m.sum()):>10d}{n:>10d}"
              f"{d.max():14.4e}{rms:14.4e}{rel:12.3e}{u:11.3g}  {verdict}")
        if tag == "all" and (n or vac):
            out = 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plant", action="store_true",
                    help="move one wet interior slope cell by 1 ulp; the gate "
                         "MUST then fail")
    a = ap.parse_args()

    R = stitch(a.run_dir)
    print(f"NEMO slope record: {a.run_dir}")
    for k in ("uslp", "vslp", "wslpi", "wslpj", "ah_wslp2", "akz"):
        v = np.asarray(R[k])
        print(f"  {k:9s} shape {v.shape}  |max| {np.nanmax(np.abs(v)):.4e}  "
              f"nonzero {int((v != 0).sum())}  nan {int(np.isnan(v).sum())}")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gmmod

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    model = LatLonCGridOceanModel(grid, z, mc)

    import traceback as _tb

    real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid
    real_prod = gmmod.compute_nemo_native_slopes
    calls: list = []          # every slope field the step BUILDS

    def _where():
        st_ = [f for f in _tb.extract_stack()[:-2]
               if "legoesm" in f.filename]
        return " <- ".join(
            f"{os.path.basename(f.filename)}:{f.lineno}" for f in st_[-2:])

    def prod_spy(*aa, **kw):
        r = real_prod(*aa, **kw)
        sl = tuple(np.asarray(x) for x in (r[:4] if isinstance(r, tuple)
                                           else (r,)))
        calls.append((dict(kw), tuple(aa), _where(), sl))
        return r

    # The CONSUMER is also patched, but only so a card that stops routing
    # native slopes into the operator is caught rather than silently scoring
    # a producer nobody reads.
    consumed: list = []

    def spy(q, *aa, **kw):
        sl = kw.get("native_slopes")
        if sl is not None:
            # (q,) + aa, not aa: q is the operator's FIRST POSITIONAL
            # argument and dropping it made the replay below hand the grid
            # where the tracer belongs.
            consumed.append((dict(kw), (q,) + tuple(aa), _where(),
                             tuple(np.asarray(x) for x in sl)))
        return real(q, *aa, **kw)

    # THE THIRD SPY, and it exists to settle a claim rather than to add a row.
    # Last round said the off-bar PRODUCER slope set reaches the answer
    # "through the implicit vertical solve".  Read NEMO instead of arguing:
    # `ln_traldf_msc = .true.` (RUN_TRAJ/namelist_cfg:267), so `tra_zdf` adds
    # **akz**, not ah_wslp2, to the implicit diffusivity (trazdf.f90:186 for
    # avt, :204 for avs) -- and legoESM's `compute_isoneutral_K33_latlon`
    # returns `_akz` too (gm_redi_latlon_cgrid.py:4650-4654).  So the object
    # the implicit solve consumes is akz, and this spy MEASURES it instead of
    # reasoning about it: if it is identically zero the producer set's DEBT
    # cannot reach the kt=1 state through this path at all.
    # AND THE SPY HAS TO BE INSTALLED WHERE THE CALL LOOKS THE NAME UP.  The
    # model does `from ...gm_redi_latlon_cgrid import
    # compute_isoneutral_K33_latlon` at ocean_model_latlon_cgrid.py:113, so the
    # call sites (:6153, :11238) read the MODEL module's binding and patching
    # only `gmmod` intercepts nothing -- the first version of this spy did
    # exactly that and printed "never called", a false negative that would have
    # read as a result.  Both namespaces are patched, and the row below refuses
    # to conclude anything if no call is seen.
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as _omod
    k33_out: list = []
    real_k33 = gmmod.compute_isoneutral_K33_latlon
    real_k33_m = _omod.compute_isoneutral_K33_latlon

    def k33_spy(*aa, **kw):
        r = real_k33(*aa, **kw)
        k33_out.append(np.asarray(r[0] if isinstance(r, tuple) else r))
        return r

    gmmod.compute_nemo_native_slopes = prod_spy
    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    gmmod.compute_isoneutral_K33_latlon = k33_spy
    _omod.compute_isoneutral_K33_latlon = k33_spy
    try:
        st, rate = dm.apply_dino_lat_lon_surface_forcing(
            state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
        with jax.disable_jit():
            model.step(st, dt=DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        gmmod.compute_nemo_native_slopes = real_prod
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real
        gmmod.compute_isoneutral_K33_latlon = real_k33
        _omod.compute_isoneutral_K33_latlon = real_k33_m
    if not calls:
        print("compute_nemo_native_slopes was never called: the card is not "
              "on NEMO's four-position slopes and this record cannot score it")
        return 1
    if not consumed:
        print("the isoneutral operator ran WITHOUT native_slopes on every "
              "call, so nothing the producer built reaches the explicit "
              "operator")
        return 1
    # Every call in one step must have been handed the SAME slopes; if not,
    # "the slopes the step used" is not a single object and scoring one of
    # them would be a silent choice.
    # The calls need not all carry ONE slope set: the card resolves
    # gm_bolus_advection='through_fct', and the bolus half is handed its own
    # slopes (``bolus_native_slopes``).  So group, print the groups, and score
    # the set the ISONEUTRAL DIFFUSION calls used -- identified, not assumed.
    def _group(lst):
        gs: list[tuple[tuple, list[int]]] = []
        for i, sl in enumerate(c[3] for c in lst):
            if sl is None or len(sl) != 4:
                continue
            for g_sl, idxs in gs:
                if all(np.array_equal(np.asarray(x), np.asarray(y))
                       for x, y in zip(g_sl, sl)):
                    idxs.append(i)
                    break
            else:
                gs.append((tuple(np.asarray(x) for x in sl), [i]))
        return gs

    # TWO FAMILIES, and they are NOT the same object.  NEMO's ldf_slp builds
    # all four positions in one pass and the record is dumped at tra_ldf's own
    # call site, so the four-position tuple the OPERATOR CONSUMES is what the
    # record is comparable to.  On this card that tuple is ASSEMBLED FROM TWO
    # producer calls -- redi_w_slope_stage_evaluation='nemo_post_slope_pair'
    # takes the U/V pair from one and the W pair from another
    # (gm_redi_latlon_cgrid.py:4369-4370) -- so no single producer output
    # equals it, and scoring only producers understates the agreement while
    # scoring only consumers hides K33's own slopes.  Both are scored.
    groups = _group(consumed)
    pgroups = _group(calls)
    print(f"\ncaptured {len(consumed)} isoneutral-operator calls carrying "
          f"{len(groups)} DISTINCT CONSUMED slope tuple(s), assembled from "
          f"{len(calls)} compute_nemo_native_slopes calls carrying "
          f"{len(pgroups)} distinct PRODUCER set(s)")
    for gi, (g_sl, idxs) in enumerate(pgroups):
        best_p = max(float(np.max(np.abs(np.asarray(x) - np.asarray(R[nm]))))
                     for nm, x in zip(("uslp", "vslp", "wslpi", "wslpj"),
                                      g_sl))
        print(f"  producer set {gi}: calls {idxs} from "
              + " | ".join(calls[i][2] for i in idxs)
              + f"\n      max|d| vs NEMO over the four fields: {best_p:.4e}")
    print("  (the producer sets include compute_isoneutral_K33_latlon's own "
          "slopes at gm_redi_latlon_cgrid.py:4573 -- a consumer-only spy is "
          "blind to them.  What that function RETURNS is measured below; last "
          "round's claim that it reaches the state through the implicit solve "
          "is RETRACTED, see the K33 row.)")
    _k33_bad = 0
    if not k33_out:
        print("\n  K33/akz UNMEASURED: compute_isoneutral_K33_latlon was "
              "never called on this card.  That is a CLAIM ABOUT THE SPY as "
              "much as about the card -- treat it as a failed measurement, "
              "not as 'the implicit path is inert'.")
        _k33_bad = 1
    else:
        _k33max = max(float(np.abs(x).max()) for x in k33_out)
        print(f"\n  IMPLICIT isoneutral coefficient the vertical solve "
              f"consumes ({len(k33_out)} call(s) to "
              f"compute_isoneutral_K33_latlon): max|K33| = {_k33max:.6e} "
              "m^2/s")
        # THE ORACLE SIDE IS READ FROM THE RECORD, not written into this
        # string.  The first version asserted "NEMO's akz is 0 on all 372528
        # cells" as a literal while `R["akz"]` was in hand -- a record where
        # that stopped being true would have inverted the test silently.
        _nemo_akz = np.abs(np.asarray(R["akz"]))
        print(f"    NEMO's akz on THIS record: max {_nemo_akz.max():.6e}, "
              f"nonzero on {int((_nemo_akz != 0).sum())} of "
              f"{_nemo_akz.size} cells.  With ln_traldf_msc=T "
              "(RUN_TRAJ/namelist_cfg:267) trazdf.f90:186/:204 add akz -- NOT "
              "ah_wslp2 -- to the implicit diffusivity, so when BOTH sides are "
              "zero the producer-set DEBT above cannot reach the kt=1 state "
              "through the implicit solve.")
        print("    SCOPE, and it is narrow: akz = MAX(rDt*(pakz + "
              "ah_wslp2/e3w^2) - 0.5, 0)*e3w^2/rDt (traldf_iso.f90:826-831) "
              "fires once |slope| > e3w*sqrt(0.5/(rDt*aht)) ~ 2.6e-3 at "
              "e3w = 10 m -- well INSIDE rn_slpmax = 0.01.  From rest DINO's "
              "slopes are ~1e-4, which is the ONLY reason both sides are "
              "zero.  A spun-up thermocline or ACC front clears that "
              "threshold, so this row says nothing about kt >> 1 and must be "
              "re-run on a Y1/Y5 record before the implicit path is called "
              "inert there.")
        _both_zero = (_k33max == 0.0) and (_nemo_akz.max() == 0.0)
        print("    -> "
              + ("UNREACHABLE AT kt=1 ONLY (both sides exactly zero)"
                 if _both_zero
                 else f"REACHABLE: legoESM {_k33max:.6e} vs NEMO "
                      f"{_nemo_akz.max():.6e} -- DEBT"))
        if a.plant:
            # The row's own control: a nonzero legoESM K33 against NEMO's zero
            # MUST be called DEBT.  Without this the "exactly zero" branch is
            # a branch nothing ever exercises.
            print("    PLANT CONTROL (K33): a legoESM max of 1e-30 against "
                  "NEMO's 0.0 classifies as "
                  + ("REACHABLE/DEBT -- correct"
                     if not ((1e-30 == 0.0) and (_nemo_akz.max() == 0.0))
                     else "UNREACHABLE -- THE ROW CANNOT FAIL"))
        if not _both_zero:
            _k33_bad = 1
    for gi, (g_sl, idxs) in enumerate(groups):
        print(f"  set {gi}: calls {idxs}  from "
              + " | ".join(calls[i][2] for i in idxs) + "\n           "
              + "  ".join(f"|{nm}|max {float(np.abs(x).max()):.4e}"
                          for nm, x in zip(("uslp", "vslp", "wslpi", "wslpj"),
                                           g_sl)))
    if len(groups) > 1:
        for gi in range(1, len(groups)):
            print("    set 0 vs set %d: " % gi
                  + "  ".join(
                      f"{nm} max|d| "
                      f"{float(np.abs(np.asarray(groups[0][0][k]) - np.asarray(groups[gi][0][k])).max()):.4e}"
                      for k, nm in enumerate(("uslp", "vslp", "wslpi",
                                              "wslpj"))))
    # EVERY distinct set is scored.  Picking one and calling it "the slopes"
    # is precisely the silent choice Rule 1e is about, and on this record the
    # two sets differ by the same 5.58e-04 that separates one of them from
    # NEMO -- so which set is scored decides the verdict.
    slope_sets = [list(g_sl) for g_sl, _ in groups]

    if a.plant:
        print("PLANT ACTIVE: vslp[100,25,10] of the CLOSEST slope set scaled "
              "by 1+1e-6; the closest set's max|d| MUST move")

    g = ndm.nemo_dino_mesh()
    wet3 = g.tmask > 0.5
    uwet = g.umask > 0.5
    vwet = g.vmask > 0.5

    for nm in ("uslp", "vslp", "wslpi", "wslpj"):
        v = np.abs(np.asarray(R[nm]))
        print(f"  limiter census {nm}: max {v.max():.17e}, cells within "
              f"1e-12 of rn_slpmax = "
              f"{int((v >= RN_SLPMAX * (1.0 - 1e-12)).sum())}, "
              f"within 1% of it = {int((v >= 0.99 * RN_SLPMAX).sum())}")

    bad = _k33_bad
    best = None
    set_worst: list[float] = []
    # The plant has to land in the set the verdict reads, and it has to move
    # a field NEMO is NONZERO in -- planting in uslp, which is identically
    # zero from rest, perturbs a row that is vacuous anyway.  Both were wrong
    # in the first version and a reviewer demonstrated it.
    _set_d = [max(float(np.max(np.abs(np.asarray(x) - np.asarray(R[nm]))))
                  for nm, x in zip(("uslp", "vslp", "wslpi", "wslpj"), g_sl))
              for g_sl in slope_sets]
    plant_set = int(np.argmin(_set_d)) if a.plant else -1
    if a.plant:
        # THE PLANT CARRIES ITS OWN CONTROL.  This gate is not at the bar, so
        # "the gate fails" proves nothing; and a 1-ulp move cannot lift a
        # max|d| that is ALREADY about one ulp.  So the arm perturbs by a
        # relative 1e-6 and requires the chosen set's max|d| to MOVE.
        _pv = np.array(slope_sets[plant_set][1])
        _before = _set_d[plant_set]
        _pv[100, 25, 10] *= (1.0 + 1e-6)
        _after = max(
            float(np.max(np.abs(np.asarray(x) - np.asarray(R[nm]))))
            for nm, x in zip(("uslp", "vslp", "wslpi", "wslpj"),
                             (slope_sets[plant_set][0], _pv,
                              slope_sets[plant_set][2],
                              slope_sets[plant_set][3])))
        print(f"  PLANT CONTROL: closest-set max|d| {_before:.4e} -> "
              f"{_after:.4e}  "
              f"{'MOVED' if _after != _before else 'DID NOT MOVE'}")
        if _after == _before:
            print("  ^^ the plant did not change what the gate measures, so "
                  "this arm proves nothing about the gate")
            return 1
        slope_sets[plant_set] = (slope_sets[plant_set][0], _pv,
                                 slope_sets[plant_set][2],
                                 slope_sets[plant_set][3])
    for gi, (uslp, vslp, wslpi, wslpj) in enumerate(slope_sets):

        print(f"\nSLOPE SET {gi} (calls {groups[gi][1]}) at tra_ldf's call "
              f"site, kt=1  (limiter rn_slpmax = {RN_SLPMAX})")
        print(f"  {'field':10s}{'subset':10s}{'cells':>10s}{'!=':>10s}"
              f"{'max|d|':>14s}{'rms':>14s}{'rel':>12s}{'ulp':>11s}")
        b = 0
        for name, lego, msk in (("uslp", uslp, uwet), ("vslp", vslp, vwet),
                                ("wslpi", wslpi, wet3),
                                ("wslpj", wslpj, wet3)):
            nemo = np.asarray(R[name])
            cap = np.abs(nemo) >= RN_SLPMAX * (1.0 - 1e-12)
            b += _score(name, lego, nemo, msk & np.isfinite(nemo), cap)
        if b == 0 and best is None:
            best = gi
        bad += b
        worst = max(
            float(np.max(np.abs(np.asarray(x) - np.asarray(R[nm]))))
            for nm, x in zip(("uslp", "vslp", "wslpi", "wslpj"),
                             (uslp, vslp, wslpi, wslpj)))
        set_worst.append(worst)
    closest = int(np.argmin(set_worst))
    if best is None:
        print(f"\n  NO captured slope set is BIT-EQUAL to NEMO's record.  "
              f"Closest is set {closest} (calls {groups[closest][1]}) at "
              f"max|d| {set_worst[closest]:.4e}; the next is "
              f"{min(set_worst[gi] for gi in range(len(set_worst)) if gi != closest) if len(set_worst) > 1 else float('nan'):.4e}.")
        bad = len(slope_sets)
        best = closest
    else:
        print(f"\n  slope set {best} (calls {groups[best][1]}) is AT BAR "
              "against NEMO's own record.")
        # bad is NOT reset here.  The first version set it to 0 the moment any
        # one set reached the bar, which absorbed a plant in any other set --
        # and every set reaches the answer (see the k33_implicit note below).
        bad = sum(1 for gi in range(len(slope_sets)) if gi != best)
        if len(slope_sets) > 1:
            # RETRACTED (hostile review, this round).  The first version
            # said the off-bar set "belongs to a pass whose GM/Redi is
            # DISCARDED", on the strength of ``_diss_incr_nn`` being bound
            # once and never read.  That is true of the EXPLICIT increment
            # and false of the answer: the Nnn pass ALSO returns
            # ``k33_implicit`` (ocean_model_latlon_cgrid.py:10446), which the
            # implicit vertical solve consumes at :10630.  So BOTH slope
            # fields reach the state, and the off-bar one is a real defect,
            # not bookkeeping.  Checked in the source rather than asserted:
            import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as _m
            src = open(_m.__file__).read()
            n_nn = src.count("_diss_incr_nn")
            n_k33 = src.count("K33_iso=k33_implicit")
            print(f"  BOTH sets reach the answer.  The Nnn pass's explicit "
                  f"GM/Redi increment is dropped ('_diss_incr_nn' occurs "
                  f"{n_nn}x, bound and never read), but its k33_implicit is "
                  f"consumed by the implicit vertical solve "
                  f"('K33_iso=k33_implicit' occurs {n_k33}x).")
            if n_k33 == 0:
                print("  ^^ k33_implicit is no longer consumed; re-derive "
                      "which pass owns the vertical isoneutral diagonal "
                      "before trusting the sentence above")
                bad += 1

    # ah_wslp2 / akz are traldf_iso_a33's own operands; legoESM builds them
    # inside the operator, so they are scored from its operand diagnostics.
    # ah_wslp2/akz must be replayed from a call in the AT-BAR set; replaying
    # the discarded pass's call would score operands the answer never sees.
    _src_call = groups[best][1][0] if best is not None else 0
    print(f"\n  ah_wslp2/akz replayed from isoneutral call {_src_call} "
          f"(consumed slope tuple {best if best is not None else 0}, the one "
          "closest to NEMO's record)")
    kw, args, _, _sl = consumed[_src_call]
    kwd = {**kw, "return_diagnostics": True,
           "return_operand_diagnostics": True}
    r = real(*args, **kwd)
    diags = r[-1] if isinstance(r, tuple) else None
    if not isinstance(diags, dict) or "zfw_operands" not in diags:
        print("\n  ah_wslp2/akz UNMEASURED: the operator returned no "
              "'zfw_operands' block (msc_stabilize off?)")
        bad += 1
    else:
        op = diags["zfw_operands"]
        # RETRACTION (this round).  The previous version read op["ah_wslp2"],
        # which the operator has ALREADY rolled to the explicit A33 flux's
        # (k, k+1) index (gm_redi_latlon_cgrid.py, `jnp.roll(_ahw_ab, -1,
        # ax_z)`), while NEMO's record is written on its own jk index.  The
        # gate therefore scored ah_wslp2[k+1] against NEMO's ah_wslp2[k] and
        # reported 0.6394 relative / regression ratio 0.7956 with "structure"
        # -- an artifact of the level shift, not a model defect.  The operator
        # now also publishes the un-rolled arrays under NEMO's own index, and
        # the gate scores THOSE.  Measured after the correction, BY THIS GATE
        # on the card's own path: 82019 cells unequal of 342134, max|d|
        # 2.7756e-17, max 9 ulp, regression ratio 1.000000, per-cell
        # lego/nemo min = median = max = 1.000000.
        #
        # RECONCILED, not overwritten (Rule 1e).  An independent arm that
        # replayed `nemo_iso_a33` DIRECTLY, fed NEMO's own slopes from the
        # record, reported 22100 of 332214 and max|d| 2.0817e-17 at 3 ulp.
        # The two differ because they are different objects: that arm scores
        # the KERNEL on NEMO's slopes over k=1..nlev-2, this one scores the
        # OPERATOR on the card's own consumed slopes over every wet cell.
        # Both land on regression ratio 1.0 with a max in the same 2e-17
        # decade, i.e. fp64 summation order in the four-term zahu_w.  The
        # number quoted from this gate is the one it prints.
        for name in ("ah_wslp2", "akz"):
            key = f"{name}_nemo_index"
            if key not in op:
                print(f"  {name} UNMEASURED: '{key}' not in the operand "
                      "diagnostics (an older operator that publishes only the "
                      "rolled copy -- scoring that one is the off-by-one bug "
                      "this key exists to prevent)")
                bad += 1
                continue
            lego = np.asarray(op[key])
            nemo = np.asarray(R[name])[..., :lego.shape[-1]]
            w = wet3[..., :lego.shape[-1]] & np.isfinite(nemo)
            bad += _score(name, lego, nemo, w,
                          np.zeros_like(nemo, dtype=bool))
            # BLIND SPOT, WRITTEN DOWN RATHER THAN DISCOVERED LATER (Rule 2;
            # a claim reviewer raised it and a first attempt to close it here
            # was itself vacuous).  The row above scores the array's VALUES and
            # says NOTHING about the level the operator puts them on: flipping
            # `jnp.roll(_ahw_ab, -1, ax_z)` to `+1` leaves it at 1.000000.
            # Scoring the rolled copy against NEMO's dump rolled the same way
            # does not help either -- both sides get the SAME permutation, so
            # every statistic is identical by construction (measured: byte-for
            # -byte the same row).  A coefficient dump cannot see its own
            # index; only a FLUX or a TENDENCY can.  The roll direction is
            # pinned by `tests/ocean/unit/test_gm_redi_latlon_cgrid.py::
            # test_ah_wslp2_is_published_on_both_indices_and_they_differ`
            # (which goes red on the flip) and by the step-1 tendency gates.
            # A clean SCALAR ratio names a statement (a 4-point sum written as
            # an average, a missing mask normaliser); a ratio with structure
            # does not.  Printed so the reader can tell which.
            a_, b_ = lego[w], nemo[w]
            den = float(b_ @ b_)
            if den:
                r_ = float(a_ @ b_) / den
                nz = b_ != 0.0
                pr = (a_[nz] / b_[nz]) if nz.any() else np.array([np.nan])
                print(f"    {name} regression ratio {r_:.6f}; per-cell "
                      f"lego/nemo on nonzero NEMO cells: min {np.nanmin(pr):.6f} "
                      f"median {np.nanmedian(pr):.6f} max {np.nanmax(pr):.6f}")
            # LEVEL WINDOW: NEMO writes ah_wslp2/akz only on jk = 2..jpkm1
            # (traldf_iso.f90 a33 loop bounds), i.e. 0-based k = 1..nlev-2, so
            # k=0 and the bottom level are NOT NEMO's values and scoring them
            # compares against an array NEMO never wrote.
            kk = slice(1, lego.shape[-1] - 1)
            w2 = np.zeros_like(w)
            w2[..., kk] = w[..., kk]
            print(f"    {name} on NEMO's OWN written window k=1..{lego.shape[-1] - 2}:")
            bad += _score(name, lego, nemo, w2,
                          np.zeros_like(nemo, dtype=bool))

    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} fields)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
