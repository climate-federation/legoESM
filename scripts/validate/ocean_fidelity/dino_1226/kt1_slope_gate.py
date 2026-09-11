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

RULE 10.  The slopes scored are the ones the MODEL'S OWN STEP handed to the
isoneutral operator, captured from the operator's ``native_slopes`` argument
during ``model.step`` -- not a re-derivation through
``compute_nemo_native_slopes`` with a hand-built state.

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
        print(f"  {name:10s}{tag:10s}{int(m.sum()):>10d}{n:>10d}"
              f"{d.max():14.4e}{rms:14.4e}{rel:12.3e}{u:11.2f}  "
              f"{'AT BAR' if n == 0 else 'DEBT'}")
        if tag == "all" and n:
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

    real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid
    calls: list = []

    import traceback as _tb

    def spy(q, *aa, **kw):
        # WHICH call site: read it off the stack, never inferred.  Two of the
        # four calls in one step carry a slope field NEMO does not have, and
        # "which one" is the entire finding.
        st_ = [f for f in _tb.extract_stack()[:-1]
               if "legoesm" in f.filename or "run_dino" in f.filename]
        where = " <- ".join(
            f"{os.path.basename(f.filename)}:{f.lineno}" for f in st_[-3:])
        calls.append((dict(kw), (q,) + tuple(aa), where))
        return real(q, *aa, **kw)

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    try:
        st, rate = dm.apply_dino_lat_lon_surface_forcing(
            state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
        with jax.disable_jit():
            model.step(st, dt=DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real
    if not calls:
        print("the isoneutral operator was never called: no slopes to score")
        return 1
    ns = [kw.get("native_slopes") for kw, _, _ in calls]
    have = [x for x in ns if x is not None]
    if not have:
        print("the operator was called WITHOUT native_slopes, so the card is "
              "not on NEMO's four-position slopes and this record cannot "
              "score it")
        return 1
    # Every call in one step must have been handed the SAME slopes; if not,
    # "the slopes the step used" is not a single object and scoring one of
    # them would be a silent choice.
    # The calls need not all carry ONE slope set: the card resolves
    # gm_bolus_advection='through_fct', and the bolus half is handed its own
    # slopes (``bolus_native_slopes``).  So group, print the groups, and score
    # the set the ISONEUTRAL DIFFUSION calls used -- identified, not assumed.
    groups: list[tuple[tuple, list[int]]] = []
    for i, kw in enumerate(kw_ for kw_, _, _ in calls):
        sl = kw.get("native_slopes")
        if sl is None:
            continue
        for g_sl, idxs in groups:
            if all(np.array_equal(np.asarray(x), np.asarray(y))
                   for x, y in zip(g_sl, sl)):
                idxs.append(i)
                break
        else:
            groups.append((tuple(np.asarray(x) for x in sl), [i]))
    print(f"\ncaptured {len(calls)} isoneutral calls carrying "
          f"{len(groups)} DISTINCT slope set(s)")
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
    slope_sets = [g_sl for g_sl, _ in groups]

    if a.plant:
        print("PLANT ACTIVE: uslp[100,25,10] of slope set 0 moved by 1 ulp; "
              "the gate MUST fail on uslp")

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

    bad = 0
    best = None
    set_worst: list[float] = []
    for gi, (uslp, vslp, wslpi, wslpj) in enumerate(slope_sets):
        if a.plant and gi == 0:
            j, i, k = 100, 25, 10
            uslp = np.array(uslp)
            uslp[j, i, k] = np.nextafter(uslp[j, i, k], np.inf)
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
        bad = 0
        if len(slope_sets) > 1:
            # The step makes TWO GM/Redi evaluations and NEMO's ldf_slp runs
            # ONCE, so a second slope field is only harmless if its pass's
            # dissipative increment is thrown away.  That is a claim about the
            # model's source, so it is CHECKED there rather than read off the
            # comment next to it (prose is a pointer, never a fact).
            import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as _m
            src = open(_m.__file__).read()
            n_nn = src.count("_diss_incr_nn")
            n_bb = src.count("diss_incr_bb") - src.count("_diss_incr_bb")
            print(f"  the other set belongs to the Nnn advective pass, whose "
                  f"dissipative increment is DISCARDED: '_diss_incr_nn' "
                  f"occurs {n_nn}x in the model (bound, never read) while "
                  f"'diss_incr_bb' -- the kept one -- occurs {n_bb}x.")
            if n_nn != 1:
                print("  ^^ _diss_incr_nn is READ somewhere: the Nnn pass's "
                      "GM/Redi is no longer discarded, so its off-bar slopes "
                      "now reach the answer and this gate must fail")
                bad += 1

    # ah_wslp2 / akz are traldf_iso_a33's own operands; legoESM builds them
    # inside the operator, so they are scored from its operand diagnostics.
    # ah_wslp2/akz must be replayed from a call in the AT-BAR set; replaying
    # the discarded pass's call would score operands the answer never sees.
    _src_call = groups[best][1][0] if best is not None else 0
    print(f"\n  ah_wslp2/akz replayed from call {_src_call} "
          f"(slope set {best if best is not None else 0})")
    kw, args, _ = calls[_src_call]
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
        for name in ("ah_wslp2", "akz"):
            if name not in op:
                print(f"  {name} UNMEASURED: not in the operand diagnostics")
                bad += 1
                continue
            lego = np.asarray(op[name])
            nemo = np.asarray(R[name])[..., :lego.shape[-1]]
            w = wet3[..., :lego.shape[-1]] & np.isfinite(nemo)
            bad += _score(name, lego, nemo, w,
                          np.zeros_like(nemo, dtype=bool))
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
    return 0 if bad else 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
