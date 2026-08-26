#!/usr/bin/env python
"""Term-by-term momentum budget on DINO's ZONAL wall rows, both models.

WHY.  legoESM's free surface carries a SUSTAINED 2-step excess on the 100
land-adjacent cells whose land neighbour lies to the north or south -- the
first WET rows against the closed end walls at about +/-69.5 deg (see
``docs/ocean/fidelity/dino_wall_flicker_source.md``).  Damping there is equal
between the models to within the bound that measurement supports, so a
sustained level difference means either a SOURCE (some term's stencil injects
alternating-sign noise the oracle's does not) or a locally different SLOW
FIELD (every term matches; only the state differs, and the Nyquist high-pass
partially passes its larger curvature).  An amplitude cannot separate those.
A per-term budget can.

HOW, and why it costs nothing.  NEMO's per-step dumps at day 180 already carry
the full per-term momentum trends (``utrd_*``/``vtrd_*``) and ten consecutive
dumps exist on disk.  For each matched state this bridges NEMO's own state
(before level included) into legoESM, evaluates ONE
``tendencies_with_diagnostics`` call -- the model's supported per-term
breakdown, whose components sum to du_dt to machine precision -- and compares
term against term.  No time integration, no GPU-hours.

THE SCOPE LIMIT THAT TRAVELS WITH EVERY RESULT FROM THIS PROBE.  Three of
NEMO's nine trends have no counterpart in the baroclinic diagnostic set:

  utrd_spg  the surface-pressure-gradient (barotropic) term.  legoESM applies
            it inside the barotropic substep loop.  **This is the term a
            free-surface source would most likely live in.**
  utrd_zdf  vertical viscosity; applied implicitly in legoESM.
  utrd_atf  the Asselin filter; not a tendency in this sense.

So a null over the six comparable terms does NOT establish "slow field".  It
narrows the source to {spg, zdf, atf} or to the slow field.  The probe says so
in its own output rather than leaving the reader to notice.

The decision rule is pre-registered in ``/tmp/dino_flicker/PREREGISTRATION.md``
(RUN 3) and is restated in ``verdict_inputs`` below so the artifact carries it.
This tool prints numbers and the rule's inputs; it does NOT print a verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_wave_twin as ewt  # noqa: E402

#: NEMO trend groups paired with legoESM diagnostic fields.  The vortcor
#: pairing (lego's EEN vortcor against NEMO's rvo+pvo) is the one
#: ``budget_pointwise.py`` already uses: lego's diagnostics exclude the SPLIT
#: Coriolis but the EEN vortcor term carries planetary f, so pvo belongs with
#: it.  Flagged rather than assumed -- if that pairing is wrong, this term's
#: numbers are wrong and no other term's are.
TERM_MAP = {
    "KE_PGF": (("hpg", "keg"), ("KE_PGF_u", "KE_PGF_v")),
    "vortcor": (("rvo", "pvo"), ("vortcor_u", "vortcor_v")),
    "vertadv": (("zad",), ("vertadv_u", "vertadv_v")),
    "lateral_visc": (("ldf",), ("Ah_lap_u", "Bh_bilap_u", "Cs_smag_u",
                                "Cl_leith_u")),
    "botdrag": (("bfr", "bfri"), ("botdrag_u", "botdrag_v")),
    "surface_phys": (("tau",), ("phys_u", "phys_v")),
}
#: NEMO trends with no baroclinic-diagnostic counterpart.  Reported, never
#: silently dropped.
UNCOMPARABLE = ("spg", "zdf", "atf")

#: A term whose oracle rms on a band is below this is DEAD there; a relative
#: difference against it would be division by a structural zero.
DEAD_TERM_MS2 = 1e-12


def t_point_bands(wet: np.ndarray) -> dict:
    """The three T-point bands, from ``eta_wave_twin.wall_direction_partition``.

    Reused rather than re-derived: this is the same partition the flicker
    result is stated on, so the budget and the amplitude cannot drift apart.
    """
    w = ewt.wall_direction_partition(wet, periodic_i=True)
    return {"zonal_wall": w["zonal_wall"],
            "meridional_wall": w["meridional_wall"],
            "interior": w["off_wall"]}


def to_u_points(band: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """T-point band -> the U faces it bounds (either neighbour in the band).

    A "wall row" on a T grid and on a velocity face are DIFFERENT SETS OF
    CELLS, and conflating them is the staggering trap this campaign has been
    burned by.  The selection is stated here and named in the output.
    """
    out = band | np.roll(band, -1, axis=1)
    return out & wet


def to_v_points(band: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """T-point band -> the V faces it bounds.

    For a ZONAL wall (land to the north or south) the V face is the one the
    wall constrains, so this is the selection that matters most there.
    """
    out = band.copy()
    out[:-1, :] |= band[1:, :]
    return out & wet


def _volume_weights(mask2d: np.ndarray, area: np.ndarray, e3: np.ndarray,
                    mask3d: np.ndarray) -> np.ndarray:
    """Volume weights over a 2-D band, ZEROED on dry levels.

    Volume, not area: these are per-level tendencies and a thin surface level
    must not count like a 200 m abyssal one.

    The 3-D mask is not optional and its absence was a real defect the land
    control caught: a 2-D band selects COLUMNS, and a column wet at the
    surface is dry below the topography, so weighting every level of it counts
    land.  DINO's end-wall rows are shallow, which is exactly where that bites
    hardest.
    """
    if mask2d.sum() == 0:
        raise SystemExit("_volume_weights: empty band")
    w = area[mask2d][:, None] * e3[None, :] * mask3d[mask2d]
    if float(w.sum()) <= 0.0:
        raise SystemExit("_volume_weights: band carries no wet volume")
    return w


def weighted_rms(field3d: np.ndarray, mask2d: np.ndarray,
                 area: np.ndarray, e3: np.ndarray,
                 mask3d: np.ndarray) -> float:
    w = _volume_weights(mask2d, area, e3, mask3d)
    x = field3d[mask2d]
    if x.shape != w.shape:
        raise SystemExit(f"weighted_rms: field {x.shape} vs weights {w.shape}")
    return float(np.sqrt(float((w * x ** 2).sum()) / float(w.sum())))


def weighted_mean(field3d: np.ndarray, mask2d: np.ndarray,
                  area: np.ndarray, e3: np.ndarray,
                  mask3d: np.ndarray) -> float:
    w = _volume_weights(mask2d, area, e3, mask3d)
    return float((w * field3d[mask2d]).sum() / float(w.sum()))


def alternation_count(signs: list[float]) -> int:
    """How many of the N-1 step transitions flip sign.

    KEPT, BUT DEMOTED, and the reason is a defect this probe shipped in its
    first version: a sign-flip count is BLIND to an alternating component that
    is smaller than the steady offset it rides on.  The measured differences
    here are steady offsets of ~4e-9 m/s2, so an alternating part of even 1%
    would never flip the sign and the count would read 0 whether or not a
    source existed -- i.e. the statistic could not fail.  It is reported for
    continuity; ``nyquist_amplitude`` is the statistic the source question is
    decided on.
    """
    s = [np.sign(v) for v in signs]
    return int(sum(1 for a, b in zip(s[:-1], s[1:]) if a != 0 and b != 0
                   and a != b))


def nyquist_amplitude(series: list[float]) -> float:
    """The 2-step (Nyquist) amplitude of a per-state series.

    The SAME operator the flicker measurement uses on the sea surface
    (``eta_wave_twin.two_dt_component``), applied here to the per-state
    band-mean term difference -- so the budget and the amplitude result are
    decided by one estimator rather than two that could disagree.  For
    ``x[n] = A*(-1)**n`` it returns A.

    This is what a sign-flip count cannot see: an alternating component riding
    on a larger steady offset.
    """
    a = np.asarray(series, dtype=float)
    if a.size < 3:
        raise SystemExit("nyquist_amplitude needs at least 3 states")
    return float(np.abs(ewt.two_dt_component(a[:, None, None])).mean())


def _llz(a: np.ndarray) -> np.ndarray:
    """NEMO (t,z,y,x) -> (y,x,z), the layout every array here is compared in."""
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def resolve_alignment(lego: np.ndarray, nemo: np.ndarray, wet3: np.ndarray,
                      axis: int) -> dict:
    """MEASURE which face offset lines the two models' velocity points up.

    legoESM's U array has one more longitude face than NEMO's and its V array
    one more latitude row, and which face index means which physical face is a
    CONVENTION, not something to infer from a docstring.  So it is measured:
    the offset is chosen by correlation against the oracle over the whole wet
    domain, and if no offset correlates decisively the probe REFUSES rather
    than quietly picking the best of three bad options -- a misaligned pairing
    would put every "difference" on the wrong cells, which is exactly how a
    wall-row finding gets manufactured.
    """
    best = None
    scores = {}
    for off in (0, 1):
        sl = [slice(None)] * 3
        sl[axis] = slice(off, off + nemo.shape[axis])
        cand = lego[tuple(sl)]
        if cand.shape != nemo.shape:
            continue
        x, y = cand[wet3], nemo[wet3]
        if x.std() == 0.0 or y.std() == 0.0:
            # A term that is identically zero on one side has an UNDEFINED
            # correlation -- it is a DEAD term, not an invalid offset, and the
            # first version of this function conflated the two and aborted the
            # whole budget on it.  Say so and let the band statistics mark it
            # dead; alignment is moot for a field of zeros.
            return {"offset": 0, "correlations": {},
                    "dead": True,
                    "reason": ("one side is identically zero on the wet "
                               "domain, so no correlation can select an "
                               "offset")}
        c = float(np.corrcoef(x, y)[0, 1])
        scores[off] = c
        if best is None or c > scores[best]:
            best = off
    if best is None:
        raise SystemExit(
            f"resolve_alignment: no offset produced matching shapes -- lego "
            f"{lego.shape}, nemo {nemo.shape}, axis {axis}")
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] < 0.5:
        raise SystemExit(
            f"resolve_alignment: best correlation {ranked[0]:.3f} on axis "
            f"{axis} -- the term pairing or the layout is wrong, and a budget "
            "built on it would attribute differences to the wrong cells. "
            f"scores={scores}")
    return {"offset": int(best), "correlations": scores, "dead": False}


def take(lego: np.ndarray, off: int, axis: int, n: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = slice(off, off + n)
    return lego[tuple(sl)]


def one_state(kt: int, run_stepdump: str, run_traj: str, recipe: str) -> dict:
    """Per-term lego-vs-NEMO comparison at ONE matched state."""
    import netCDF4
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kamm_twin_90d as ktw  # noqa: PLC0415

    br, cfg, mc, model, forcing, sf, st = ktw._build_twin_state(  # noqa: SLF001
        recipe, run_traj, run_stepdump, bridge_before=True,
        restart_file=f"DINO_{kt:08d}_restart.nc")
    _tend, diag = model.tendencies_with_diagnostics(
        st, surface_forcing=sf, dt=ktw.DT)

    mm = netCDF4.Dataset(os.path.join(run_stepdump, "mesh_mask.nc"))
    umask = _llz(mm.variables["umask"][0]) > 0.5
    vmask = _llz(mm.variables["vmask"][0]) > 0.5
    tmask = _llz(mm.variables["tmask"][0]) > 0.5
    e1u = np.asarray(mm.variables["e1u"][0]).squeeze()
    e2u = np.asarray(mm.variables["e2u"][0]).squeeze()
    e1v = np.asarray(mm.variables["e1v"][0]).squeeze()
    e2v = np.asarray(mm.variables["e2v"][0]).squeeze()
    e3 = np.asarray(mm.variables["e3t_1d"][:]).squeeze()
    wet = tmask[..., 0]
    nxt = os.path.join(run_stepdump, f"DINO_{kt + 1:08d}_restart.nc")
    if not os.path.exists(nxt):
        raise SystemExit(f"no NEMO dump at kt={kt + 1}: {nxt}")
    ds = netCDF4.Dataset(nxt)
    kt_in = int(np.squeeze(ds.variables["kt"][:]))
    if kt_in != kt + 1:
        raise SystemExit(f"{nxt} carries kt={kt_in}, expected {kt + 1}")

    bands = t_point_bands(wet)
    pts = {"u": {k: to_u_points(v, umask[..., 0]) for k, v in bands.items()},
           "v": {k: to_v_points(v, vmask[..., 0]) for k, v in bands.items()}}
    area = {"u": e1u * e2u, "v": e1v * e2v}
    msk3 = {"u": umask, "v": vmask}

    def lego_term(names: tuple, comp: str) -> np.ndarray:
        """Sum the legoESM diagnostic fields that make up one NEMO trend."""
        tot = None
        for nm in names:
            base = nm[:-2] if nm.endswith(("_u", "_v")) else nm
            f = getattr(diag, f"{base}_{comp}")
            a = np.asarray(f.data if hasattr(f, "data") else f, dtype=float)
            tot = a if tot is None else tot + a
        return tot

    out: dict = {"kt": kt, "kt_next": kt + 1, "bands": {}, "alignment": {},
                 "uncomparable_nemo_trends": list(UNCOMPARABLE)}
    for comp, axis in (("u", 1), ("v", 0)):
        wet3 = msk3[comp]
        for term, (ntrd, lnames) in TERM_MAP.items():
            nemo = None
            for t in ntrd:
                v = _llz(ds.variables[f"{comp}trd_{t}"][0]).astype(float)
                nemo = v if nemo is None else nemo + v
            names = tuple(n for n in lnames if n.endswith(f"_{comp}")) or lnames
            lego_full = lego_term(names, comp)
            key = f"{comp}:{term}"
            al = resolve_alignment(lego_full, nemo, wet3, axis)
            out["alignment"][key] = al
            lego = take(lego_full, al["offset"], axis, nemo.shape[axis])
            d = (lego - nemo) * wet3
            rec = {}
            for bname, bmask in pts[comp].items():
                n_rms = weighted_rms(nemo * wet3, bmask, area[comp], e3, wet3)
                rec[bname] = {
                    "nemo_rms_ms2": n_rms,
                    "lego_rms_ms2": weighted_rms(lego * wet3, bmask,
                                                 area[comp], e3, wet3),
                    "diff_rms_ms2": weighted_rms(d, bmask, area[comp], e3,
                                                 wet3),
                    "diff_mean_ms2": weighted_mean(d, bmask, area[comp], e3,
                                                   wet3),
                    "cells": int(bmask.sum()),
                    "dead": bool(n_rms < DEAD_TERM_MS2),
                }
                rec[bname]["relative_diff"] = (
                    None if rec[bname]["dead"]
                    else rec[bname]["diff_rms_ms2"] / n_rms)
            out["bands"][key] = rec
    return out


def verdict_inputs(states: list) -> dict:
    """The pre-registered rule's INPUTS, per term. Not a verdict.

    RUN 3 of /tmp/dino_flicker/PREREGISTRATION.md:
      SOURCE     -- some comparable term with R(zonal) >= 3*R(interior) AND a
                    2-step signature in its zonal band-mean difference.  The
                    registration wrote that second condition as a SIGN-FLIP
                    COUNT; that statistic is blind to an alternating part
                    smaller than the steady offset it rides on, so it is
                    reported but the condition is DECIDED on
                    ``nyquist_over_steady``.  The substitution is recorded as
                    a deviation from the registration rather than made
                    silently.
      SLOW FIELD -- every comparable term with R(zonal) <= 1.5*R(interior) and
                    no term alternating on more than 1 of 4 -- and even then
                    the honest statement is "not in the six comparable terms",
                    because the barotropic surface-pressure-gradient term is
                    not among them.
      otherwise  -- INCONCLUSIVE.
    The two numbers each arm needs are computed here; the reading is left to
    the analysis, because a probe that prints its own verdict gets that verdict
    echoed back as evidence.
    """
    keys = sorted(states[0]["bands"])
    out = {}
    for k in keys:
        zo = [s["bands"][k]["zonal_wall"] for s in states]
        it = [s["bands"][k]["interior"] for s in states]
        if any(r["dead"] for r in zo) or any(r["dead"] for r in it):
            out[k] = {"dead_on_some_state": True}
            continue
        rz = float(np.mean([r["relative_diff"] for r in zo]))
        ri = float(np.mean([r["relative_diff"] for r in it]))
        means = [r["diff_mean_ms2"] for r in zo]
        steady = float(abs(np.mean(means)))
        nyq = nyquist_amplitude(means)
        out[k] = {
            "relative_diff_zonal": rz,
            "relative_diff_interior": ri,
            "enrichment_zonal_over_interior": (rz / ri) if ri > 0 else None,
            "zonal_diff_mean_by_state_ms2": means,
            # The source statistic. A steady operator offset gives a Nyquist
            # amplitude orders below the offset; an injecting stencil gives one
            # comparable to it.
            "zonal_steady_offset_ms2": steady,
            "zonal_nyquist_amplitude_ms2": nyq,
            "nyquist_over_steady": (nyq / steady) if steady > 0 else None,
            # Reported for continuity only -- see the docstring; it cannot see
            # an alternating part smaller than the steady offset.
            "sign_alternations": alternation_count(means),
            "n_states": len(states),
        }
    return out


def self_check(kt: int, run_stepdump: str) -> dict:
    """Controls that must pass before any number above is quoted.

    LAND: the band masks must exclude dry cells, so poisoning every dry cell
    leaves every band statistic bit-identical -- and, as in the flicker probe,
    it is run against UNMASKED weights so the band mask is the only defence.
    BAND DISJOINTNESS: the three bands must not overlap, or a cell would be
    counted twice and the "enrichment" of one band would leak into another.
    """
    import netCDF4
    mm = netCDF4.Dataset(os.path.join(run_stepdump, "mesh_mask.nc"))
    tmask = _llz(mm.variables["tmask"][0]) > 0.5
    umask = _llz(mm.variables["umask"][0]) > 0.5
    e1u = np.asarray(mm.variables["e1u"][0]).squeeze()
    e2u = np.asarray(mm.variables["e2u"][0]).squeeze()
    e3 = np.asarray(mm.variables["e3t_1d"][:]).squeeze()
    wet = tmask[..., 0]
    bands = t_point_bands(wet)
    ov = {}
    names = sorted(bands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov[f"{a}&{b}"] = int((bands[a] & bands[b]).sum())
    rng = np.random.default_rng(0)
    field = rng.normal(size=umask.shape)
    field[~umask] = 0.0
    poisoned = field.copy()
    sign = ((-1.0) ** np.arange(field.shape[-1]))
    poisoned[~umask] = 1.0e6 * np.broadcast_to(sign, field.shape)[~umask]
    area_unmasked = np.where(e1u * e2u > 0, e1u * e2u, 1.0)
    bu = to_u_points(bands["zonal_wall"], umask[..., 0])
    base = weighted_rms(field, bu, area_unmasked, e3, umask)
    pois = weighted_rms(poisoned, bu, area_unmasked, e3, umask)
    return {"band_overlaps": ov,
            "bands_disjoint": all(v == 0 for v in ov.values()),
            "land_poison_identical": bool(base == pois),
            "zonal_wall_u_faces": int(bu.sum()),
            "zonal_wall_t_cells": int(bands["zonal_wall"].sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kt0", type=int, default=5760)
    p.add_argument("--n-states", type=int, default=5)
    p.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p.add_argument("--run-stepdump", default=None)
    p.add_argument("--run-traj", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kamm_twin_90d as ktw  # noqa: PLC0415
    rs = a.run_stepdump or ktw.RUN_STEPDUMP
    rt = a.run_traj or ktw.RUN_TRAJ

    sc = self_check(a.kt0, rs)
    if not sc["bands_disjoint"]:
        raise SystemExit(f"SELF-CHECK FAILED: bands overlap {sc}")
    if not sc["land_poison_identical"]:
        raise SystemExit("SELF-CHECK FAILED: a band statistic moved when land "
                         "was poisoned under unmasked weights")

    states = [one_state(a.kt0 + i, rs, rt, a.recipe)
              for i in range(a.n_states)]
    doc = ewt.provenance({
        "probe": "zonal_wall_momentum_budget",
        "recipe": a.recipe, "kt0": a.kt0, "n_states": a.n_states,
        "run_stepdump": rs,
        "self_check": sc,
        "uncomparable_nemo_trends": list(UNCOMPARABLE),
        "scope_limit": (
            "utrd_spg/vtrd_spg (the barotropic surface-pressure-gradient "
            "term) has NO counterpart in the baroclinic diagnostic set and is "
            "the term a free-surface source would most likely live in. A null "
            "over the six comparable terms narrows the source to "
            "{spg, zdf, atf} or to the slow field; it does not establish "
            "either."),
        "verdict_inputs": verdict_inputs(states),
        "states": states})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in doc.items() if k != "states"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
