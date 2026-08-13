#!/usr/bin/env python
"""Arctic SALT INVENTORY vs VOLUME: does the +1.4 psu bias have a salt cause?

THE QUESTION.  Every grid carries an Arctic surface-salinity bias of about
+1.4 psu against NEMO at day 30 (tripolar +1.35, MPAS +1.42, FESOM2 +1.48) --
common-mode across three unrelated meshes, so it is shared physics rather than
discretization.  The working hypothesis has been a FRESHWATER/VOLUME deficit,
which is what routing SSS restoring as a real water flux would address.

THE CONTRARY HYPOTHESIS (codex round-3 steel-man, job 9387815): the bias is a
SALT-INVENTORY EXCESS instead.  That distinction decides whether a 13 GPU-hour
arm is worth launching, because real-water restoring conserves ``h*S`` -- it
moves water and dilutes, but it cannot remove salt MASS -- and it is suppressed
under sea ice, which is where much of the Arctic bias sits.

    S_mean = M_salt / V

so a +1.4 psu mean error is either a numerator (salt) or a denominator
(volume) problem, and the two demand different fixes.

PRE-REGISTERED VERDICT RULE (fixed before running, so it cannot drift):

  * SALT-DOMINATED  -- |dM_salt/M_salt| substantially exceeds |dV/V|, i.e. the
    salt inventory alone explains most of dS_mean:  the water-routing
    hypothesis is REFUTED for this bias.  DO NOT launch the arm; the lever is
    somewhere in the salt budget (ice brine, the runoff/river channel, or the
    initial inventory).
  * VOLUME-DOMINATED -- the reverse: SUPPORTED, launch the arm.
  * MIXED -- neither term dominates.  Report both magnitudes and claim NO
    direction; that is a legitimate answer, not a failure.

INSTRUMENT VALIDATION, run before the numbers are quoted (this campaign has
already retracted eight claims taken from an unvalidated probe):

  1. the volume integral is checked against the SAME quantity computed from
     the mesh alone, which must agree to round-off -- if the volume is wrong
     the salt integral cannot be trusted either;
  2. the region is required to select a nonzero number of wet cells (the
     shared integrator raises otherwise, rather than returning a plausible 0);
  3. every NaN in a wet cell is FATAL, never nan-summed away.

Reuses ``global_tracer_content.tracer_content`` (locked conventions, fp64,
native-frame handling) rather than re-deriving the integral -- the ONLY change
there is the ``region_mask`` argument this probe needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "core"))
from legoesm import constants  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from global_tracer_content import (  # noqa: E402
    _git_sha, _native, load_mesh_depth_1d, load_mesh_latitude,
    load_mesh_metrics, tracer_content,
)

ARCTIC_LAT_DEG = 60.0


def _ice_masks(snapshot, wet2d):
    """Split the region into ice-covered and open water.

    Returns ``(ice, open_water, source)``.  When the snapshot carries no ice
    field the split is NOT invented: both come back None and the caller reports
    the combined number only.  A fabricated split would be worse than none --
    it is the ice-covered fraction that carries the mechanism claim.
    """
    for key in ("ice_concentration", "siconc", "ice_conc", "aice"):
        if key in snapshot:
            a = np.asarray(snapshot[key], dtype=np.float64)
            if a.ndim == 3:                      # categories -> total
                a = a.sum(axis=-1) if a.shape[-1] < a.shape[0] else a.sum(axis=0)
            if a.ndim == 2:
                a = _native(a)                 # (332,362) -> native (331,360)
            elif a.shape[:2] == (332, 362):
                a = _native(a).sum(axis=0)     # level/category-last -> summed
            ice = (a > 0.15) & wet2d             # 15 % = the standard edge
            return ice, (~ice) & wet2d, key
    return None, None, None


def _profile(S, dV, gdept, label, out=None):
    """Volume-weighted mean salinity per level, printed with the level depth.

    Mean (not content) so the comparison is not dominated by how much water a
    level happens to hold: a level-2 content difference and a level-40 content
    difference are not commensurate, but their MEANS are.
    """
    num = (S * dV).sum(axis=(1, 2))
    den = dV.sum(axis=(1, 2))
    ok = den > 0.0
    mean = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    rows = []
    for k in range(mean.size):
        if not ok[k]:
            continue
        rows.append({"level": k, "depth_m": float(gdept[k]),
                     "mean_S_psu": float(mean[k]), "volume_m3": float(den[k])})
    print(f"[profile:{label}] level depth_m mean_S volume_m3")
    for r in rows:
        print(f"[profile:{label}] {r['level']:3d} {r['depth_m']:9.2f} "
              f"{r['mean_S_psu']:9.4f} {r['volume_m3']:.4e}")
    if out is not None:
        out["profile"] = rows
    return rows


def _nemo_main(a) -> int:
    """Arctic salt/volume from NEMO's own output, using NEMO's own metrics."""
    try:
        import netCDF4 as nc
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit(f"netCDF4 required: {exc}")

    e1t, e2t, _e3t_ref, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)

    ds = nc.Dataset(a.nemo_gridt)
    try:
        it = a.nemo_month - 1
        def v(name):
            # netCDF4 returns a MASKED array; `np.asarray` drops the mask and
            # keeps the raw fill values (~1e20 on land), which then multiply
            # into the integral.  The first version did exactly that and
            # produced 2.37e52 psu.m3 against NEMO's own 7.29e20 -- caught by
            # the saltc check below, which is why that check exists.
            # `filled(0.0)` makes land contribute nothing whatever the mask
            # alignment turns out to be.
            arr = ds.variables[name][it]
            return np.ma.filled(np.ma.masked_invalid(arr), 0.0).astype(np.float64)
        S = v("so")                                   # (75, 331, 360)
        e3t = v("e3t")                                # NEMO's LIVE thickness
        saltc = v("saltc") if "saltc" in ds.variables else None
        # Guard the fill-value class explicitly rather than trusting the mask:
        # any |value| above this is not ocean salinity or thickness.
        for _nm, _a in (("so", S), ("e3t", e3t)):
            if np.abs(_a).max() > 1.0e6:
                raise SystemExit(
                    f"FATAL: {_nm} still carries fill values "
                    f"(max |{_nm}| = {np.abs(_a).max():.3e}); the integral "
                    "would be contaminated")
    finally:
        ds.close()

    if S.shape != tmask.shape:
        raise SystemExit(
            f"NEMO so {S.shape} does not match the mesh frame {tmask.shape}; "
            "the grid_T is expected ALREADY native (331, 360)")

    wet = tmask > 0.5
    arctic = (lat >= a.arctic_lat) & wet.any(axis=0)
    if not arctic.any():
        raise SystemExit(f"FATAL: no wet cells north of {a.arctic_lat} deg")

    sel = wet & arctic[None]
    dV = e1t[None] * e2t[None] * e3t * sel
    if not np.isfinite(S[sel]).all():
        raise SystemExit("FATAL: non-finite NEMO salinity in a wet Arctic cell")
    V = float(dV.sum())
    M = float((S * dV).sum())

    # INSTRUMENT CHECK against NEMO's OWN column salt-content diagnostic.
    # This is a stronger check than the mesh-volume one: it validates the SALT
    # integral, computed by a different code (NEMO's), on the same cells.
    if saltc is not None:
        area = e1t * e2t * arctic
        # saltc is a column integral [psu.m] (rho-free); compare shapes first
        if saltc.shape != arctic.shape:
            raise SystemExit(f"saltc {saltc.shape} vs mask {arctic.shape}")
        M_nemo_diag = float((saltc * area).sum())
        # UNITS, read from the file rather than assumed: saltc carries
        # `units = "PSU*kg/m2"`, i.e. rho * S * dz, while this probe integrates
        # S * dV in psu.m3.  Convert with NEMO's OWN rau0 (phycst.F90 = 1026,
        # `constants.rho_ocean_nemo`) -- NOT legoESM's 1025 rho_ocean, which
        # would leave a spurious 0.1 % gap and invite a fudged tolerance.
        M_cmp = M * constants.rho_ocean_nemo
        rel = abs(M_cmp - M_nemo_diag) / abs(M_nemo_diag)
        print(f"[check] salt integral vs NEMO's own saltc "
              f"(x rau0={constants.rho_ocean_nemo:g}): rel {rel:.3e}")
        if rel > 5e-3:
            raise SystemExit(
                f"FATAL: probe salt*rau0 {M_cmp:.6e} vs NEMO saltc "
                f"{M_nemo_diag:.6e} "
                f"(rel {rel:.2e}) -- the integral is not measuring what NEMO "
                "reports, so no verdict may be drawn from it")
    else:
        print("[check] no saltc in the file -- salt integral UNVALIDATED")

    print(f"[{a.label}] Arctic (>{a.arctic_lat:g}N): V {V:.6e} m3   "
          f"M_salt {M:.6e} psu.m3   S_mean {M / V:.4f} psu")
    _prof = None
    if a.profile:
        _prof = _profile(S, dV, load_mesh_depth_1d(a.mesh_mask), a.label)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps({
            "generated_by": str(_HERE), "git_sha": _git_sha(_HERE.parents[3]),
            "label": a.label, "nemo_gridt": str(a.nemo_gridt),
            "nemo_month": a.nemo_month, "arctic_lat_deg": a.arctic_lat,
            "arctic": {"volume_m3": V, "salt_content_psu_m3": M,
                       "mean_S_psu": M / V},
            "profile": _prof,
        }, indent=2))
        print(f"[json] {a.out_json}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshot", help="legoESM day-30 .npz")
    src.add_argument("--nemo-gridt", help="NEMO grid_T .nc (the reference arm)")
    p.add_argument("--nemo-month", type=int, default=1,
                   help="1-based month index into the NEMO file (default 1)")
    p.add_argument("--mesh-mask", required=True, help="NEMO mesh_mask .nc")
    p.add_argument("--label", required=True)
    p.add_argument("--arctic-lat", type=float, default=ARCTIC_LAT_DEG)
    p.add_argument("--out-json", default=None)
    p.add_argument("--profile", action="store_true",
                   help="Also print the Arctic VOLUME-WEIGHTED mean salinity "
                        "per level. With the inventory and volume both matching "
                        "NEMO, the surface bias must be a vertical "
                        "REDISTRIBUTION -- this locates the compensating "
                        "deficit instead of leaving it inferred.")
    a = p.parse_args()

    # --- NEMO reference branch -------------------------------------------
    # NEMO's grid_T is ALREADY on the native (331, 360) frame -- applying
    # `_native` to it would slice a second time.  Verified from the file:
    # so (t, 75, 331, 360), zos (t, 331, 360), e3t (t, 75, 331, 360).
    # It also carries its OWN time-varying `e3t`, so the volume needs no
    # z-star reconstruction, and `saltc` (column salt content), which is used
    # below as an INDEPENDENT check on this probe's salt integral.
    if a.nemo_gridt:
        return _nemo_main(a)

    z = dict(np.load(a.snapshot))
    # load_mesh_metrics returns a 4-TUPLE (read it, do not assume a dict --
    # the first version of this probe indexed it by name and died at line 96).
    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)

    T3d, S3d = _native(z["T"]), _native(z["S"])
    # `_native` already dispatches on ndim (2-D -> (nj,ni), 3-D -> level-first),
    # so eta is passed straight through.  Wrapping it in a leading axis, as the
    # first version did, built a (1, 332, 362) array that tripped its shape
    # assertion -- the layout IS the API, which that function's own docstring
    # says in so many words.
    eta = _native(z["eta"])

    wet2d = (tmask > 0.5).any(axis=0)
    arctic = (lat >= a.arctic_lat) & wet2d
    n_arctic = int(arctic.sum())
    if n_arctic == 0:
        raise SystemExit(f"FATAL: no wet cells north of {a.arctic_lat} deg")

    # ---- INSTRUMENT VALIDATION -------------------------------------------
    # The volume the integrator reports must match the volume the mesh implies,
    # at eta = 0.  If this fails the salt number is meaningless.
    ref = tracer_content(np.zeros_like(T3d), np.zeros_like(S3d),
                         np.zeros_like(eta), e1t, e2t, e3t, tmask,
                         region_mask=arctic)
    mesh_vol = float((e1t[None] * e2t[None] * e3t
                      * (tmask > 0.5) * arctic[None]).sum())
    rel = abs(ref["volume_m3"] - mesh_vol) / mesh_vol
    if rel > 1e-12:
        raise SystemExit(
            f"FATAL: instrument check failed -- integrator volume "
            f"{ref['volume_m3']:.6e} vs mesh {mesh_vol:.6e} (rel {rel:.2e})")
    print(f"[check] volume integral matches the mesh to {rel:.2e} "
          f"({n_arctic} Arctic wet columns)")

    out = {
        "generated_by": str(_HERE),
        "git_sha": _git_sha(_HERE.parents[3]),
        "label": a.label,
        "snapshot": str(a.snapshot),
        "arctic_lat_deg": a.arctic_lat,
        "n_arctic_columns": n_arctic,
        "instrument_volume_rel_err": rel,
    }

    whole = tracer_content(T3d, S3d, eta, e1t, e2t, e3t, tmask,
                           region_mask=arctic)
    out["arctic"] = whole
    print(f"[{a.label}] Arctic (>{a.arctic_lat:g}N): "
          f"V {whole['volume_m3']:.6e} m3   "
          f"M_salt {whole['salt_content_psu_m3']:.6e} psu.m3   "
          f"S_mean {whole['mean_S_psu']:.4f} psu")

    if a.profile:
        wet3 = (tmask > 0.5) & arctic[None]
        dV_ref = e1t[None] * e2t[None] * e3t * wet3
        H = (e3t * (tmask > 0.5)).sum(axis=0)
        colw = H > 0.0
        dil = np.where(colw, (H + eta * colw) / np.where(colw, H, 1.0), 0.0)
        _profile(S3d, dV_ref * dil[None], load_mesh_depth_1d(a.mesh_mask),
                 a.label, out)

    ice, openw, src = _ice_masks(z, arctic)
    if ice is None:
        print("[split] no ice field in the snapshot -- combined number only "
              "(the split is NOT invented)")
        out["ice_split"] = None
    else:
        out["ice_source"] = src
        for name, msk in (("ice_covered", ice), ("open_water", openw)):
            n = int(msk.sum())
            if n == 0:
                print(f"[split] {name}: no cells")
                out[name] = None
                continue
            r = tracer_content(T3d, S3d, eta, e1t, e2t, e3t, tmask,
                               region_mask=msk)
            out[name] = dict(r, n_columns=n)
            print(f"[{a.label}] {name:12s} ({n:5d} cols): "
                  f"V {r['volume_m3']:.4e}  M_salt {r['salt_content_psu_m3']:.4e}"
                  f"  S_mean {r['mean_S_psu']:.4f}")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(out, indent=2))
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
