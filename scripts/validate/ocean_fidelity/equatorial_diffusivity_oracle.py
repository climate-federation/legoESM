#!/usr/bin/env python
"""NEMO's own equatorial diffusivities: the target the closure has to hit.

The tripole's equatorial thermocline sits at the right depth and is 2.3 times
too sharp, its cold tongue is too warm, and its undercurrent is four times too
weak while the pressure gradient driving it is right.  One number produces all
three if the turbulence closure mixes momentum too well and heat too poorly:
the turbulent Prandtl number, viscosity over diffusivity.

NEMO's own five-day output carries ``avm`` (viscosity), ``avt`` (heat
diffusivity), ``avs`` (salt) and ``bn2`` on the same grid our runs are scored
against, so the oracle's Prandtl profile is a file read rather than an
inference.  This prints it, and prints the pieces it is built from, so a later
comparison against our closure has a target with a known provenance instead of
a remembered number.

THE ORACLE'S NUMBER COMES FIRST, and it is established without reference to
ours.  A previous round of this campaign quoted "NEMO 1.85" for the equatorial
Prandtl number from a probe whose controls were never shown; the oracle table
is printed on its own before any comparison, so the target does not move when
our side changes.

OUR SIDE IS OPTIONAL (``--legoesm-snapshot``) and is read from the model's own
``A_v``/``K_v``, stored by ``run_omip_core2.py --kprofile-snapshots``.  It is
never reconstructed from TKE and a mixing length: a re-derived lookalike
compared against NEMO's published field is a different quantity wearing the
same axis labels.

READ IT AS A TARGET, NOT A VERDICT.  A Prandtl profile is not a defect.  What
it does is say what our closure must reproduce, and at which depths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Above this, NEMO's diffusivity is the convective-adjustment sentinel rather
#: than the turbulence closure's answer (rn_avevd is 10 or 100 m2/s; the real
#: ocean's vertical diffusivity does not exceed ~1e-1 m2/s outside deep
#: convection).  Used only to report how much of a box is convecting.
CONVECTIVE_AVT = 1.0            # m2/s


def _load(gridw: Path, rec: int, halfwidth: float):
    import xarray as xr
    ds = xr.open_dataset(gridw, decode_times=False)
    for v in ("avm", "avt", "bn2"):
        if v not in ds:
            raise SystemExit(
                f"{gridw}: no {v!r}. Present: {sorted(ds.data_vars)}. This "
                "probe needs the diffusivity fields; a grid_W written without "
                "them cannot answer the question.")
    lat = np.asarray(ds["nav_lat"].values, dtype=np.float64)
    lon = np.asarray(ds["nav_lon"].values, dtype=np.float64) % 360.0
    depth_name = "depthw" if "depthw" in ds else "deptht"
    z = np.abs(np.asarray(ds[depth_name].values, dtype=np.float64))

    def grab(name):
        a = np.asarray(ds[name].values, dtype=np.float64)
        if a.ndim == 4:
            a = a[rec]
        a = np.where(np.abs(a) > 1e10, np.nan, a)
        return np.moveaxis(a, 0, -1)                  # (y, x, z)

    return grab("avm"), grab("avt"), grab("bn2"), lat, lon, z


def _load_ours(snapshot: Path):
    """Our own avm/avt from a snapshot written with ``--kprofile-snapshots``.

    These are the model's OWN ``A_v``/``K_v`` -- the arrays the implicit
    vertical solves consumed -- not a re-derivation from TKE and a mixing
    length.  A snapshot without them is a hard error rather than a fallback:
    an inferred profile compared against NEMO's published one would be a
    different quantity wearing the same axis labels.

    The diffusivities live at INTERFACES (nlev-1 of them), so the depth axis
    is the midpoint of the adjacent cell centres.  N2 is not read here; the
    stratification comparison is ``equatorial_n2_vs_nemo.py``'s job and
    duplicating it would give two numbers that could disagree.
    """
    z = np.load(snapshot)
    missing = [k for k in ("K_M_diag", "K_H_diag") if k not in z]
    if missing:
        raise SystemExit(
            f"{snapshot}: no {missing}. Re-run with --kprofile-snapshots; a "
            "snapshot without the closure's own diffusivities cannot be "
            "compared against NEMO's published avm/avt.")
    avm = np.asarray(z["K_M_diag"], dtype=np.float64)
    avt = np.asarray(z["K_H_diag"], dtype=np.float64)
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64) % 360.0
    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon) % 360.0
    nk = avm.shape[-1]
    # The TRUE interior interface depths, as the closure and NEMO's depthw
    # define them.  A cell-centre midpoint reconstruction is NOT the same: on
    # ORCA1 it put the nominal 64.96 m interface at 65.12 m, which drops it
    # from a <=65 m window that keeps NEMO's 64.98 m depthw -- an unequal
    # vertical window presented as a common-level comparison (codex 2026-08-23).
    if "z_interface_ref" in z:
        zk = np.abs(np.asarray(z["z_interface_ref"], dtype=np.float64))
        if zk.size != nk:
            raise SystemExit(
                f"{snapshot}: z_interface_ref has {zk.size} levels against "
                f"{nk} K levels; the depth axis and the data disagree.")
    else:
        zc = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
        if nk == zc.size:
            zk = zc
        else:
            raise SystemExit(
                f"{snapshot}: K has {nk} interior interfaces but the snapshot "
                "carries no z_interface_ref; a cell-centre midpoint is the "
                "wrong depth axis for a diffusivity. Re-run with "
                "--kprofile-snapshots on the current code, which stores it.")
    wet = np.asarray(z["land_mask"], dtype=np.float64) > 0.5
    avm = np.where(wet[..., None], avm, np.nan)
    avt = np.where(wet[..., None], avt, np.nan)
    return avm, avt, None, lat, lon, zk


def _table(avm, avt, bn2, band, z, depth_max, label):
    """One depth table for one model, and the closure/background summary."""
    kmax = min(int(np.searchsorted(z, depth_max)) + 1, z.size)
    print(f"{'depth':>8}{'avm':>11}{'avt':>11}{'Pr=avm/avt':>12}"
          f"{'N2':>11}{'conv':>7}{'wet':>7}")
    print(f"{'[m]':>8}{'[m2/s]':>11}{'[m2/s]':>11}{'median':>12}"
          f"{'[1/s2]':>11}{'frac':>7}{'':>7}")
    prof = []
    for k in range(kmax):
        m = avm[..., k][band]
        t = avt[..., k][band]
        b = (bn2[..., k][band] if bn2 is not None
             else np.full(m.shape, np.nan))
        ok = np.isfinite(m) & np.isfinite(t) & (t > 0)
        if not ok.any():
            continue
        conv = float(np.mean(t[ok] > CONVECTIVE_AVT))
        # Prandtl is a RATIO: taken per column, then reduced.  A ratio of
        # reductions is a different quantity.
        pr = float(np.median(m[ok] / t[ok]))
        _b = b[ok]
        prof.append((z[k], float(np.median(m[ok])), float(np.median(t[ok])),
                     pr, float(np.nanmedian(_b)) if np.isfinite(_b).any()
                     else np.nan, int(ok.sum()), conv))
        print(f"{z[k]:>8.1f}{prof[-1][1]:>11.4g}{prof[-1][2]:>11.4g}"
              f"{pr:>12.3f}{prof[-1][4]:>11.3g}{conv:>7.3f}"
              f"{prof[-1][5]:>7d}")

    if not prof:
        raise SystemExit(f"[{label}] no finite diffusivities in the box")
    # cols: depth, Pr, conv-frac, avt (heat), avm (viscosity)
    arr = np.array([(p[0], p[3], p[6], p[2], p[1]) for p in prof])

    def _band(lo, hi, title, tag):
        m = (arr[:, 0] >= lo) & (arr[:, 0] <= hi)
        if not m.any():
            print(f"[{label}] {title}: no levels in {lo:.0f}-{hi:.0f} m")
            return {}
        s = dict(pr=float(np.median(arr[m, 1])),
                 avt=float(np.median(arr[m, 3])),
                 avm=float(np.median(arr[m, 4])),
                 conv=float(arr[m, 2].max()))
        print(f"[{label}] {title} ({lo:.0f}-{hi:.0f} m):")
        print(f"  Prandtl median {s['pr']:.3f}  "
              f"avm {s['avm']:.3g}  avt {s['avt']:.3g} m2/s  "
              f"conv-frac max {s['conv']:.3f}")
        return {f"{tag}_{k}": v for k, v in s.items()}

    # THREE bands, because the cooling budget is generated between the mixed
    # layer and the thermocline, not in the top 60 m (GLM 2026-08-23).  The
    # 5-65 m band is the surface closure; the 65-105 m ENTRAINMENT band, from
    # the mixed-layer base to Z20, is where -w dT/dz and d(avt dT/dz)/dz peak
    # and where the missing cooling has to come from; below 100 m is
    # background, whose ratio says nothing about the closure and must not be
    # averaged with either.  Reporting avm AND avt separately (not just their
    # ratio) is what tells a Prandtl-RATIO defect (avm~NEMO, avt<<NEMO) from
    # TURBULENCE STARVATION (both <<NEMO): a ratio of two log-varying fields
    # is unstable and hides which one moved.
    print()
    summary = {}
    summary.update(_band(5.0, 65.0, "SURFACE closure", "sfc"))
    summary.update(_band(65.0, 105.0, "ENTRAINMENT zone", "ent"))
    _band(100.0, 1.0e9, "BACKGROUND (closure off)", "bkg")
    # Back-compat keys for callers that read the flat names.
    if "sfc_pr" in summary:
        summary["pr"], summary["avt"] = summary["sfc_pr"], summary["sfc_avt"]
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo-gridw", required=True, type=Path)
    ap.add_argument("--rec", type=int, default=5,
                    help="record (5-day file: 5 = days 26-30)")
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--lon-west", type=float, default=200.0)
    ap.add_argument("--lon-east", type=float, default=260.0,
                    help="the cold-tongue longitudes, where the bias lives")
    ap.add_argument("--depth-max", type=float, default=300.0)
    ap.add_argument("--legoesm-snapshot", nargs="*", default=None,
                    help="Our own snapshot(s) written with "
                         "--kprofile-snapshots. Each is reduced in the SAME "
                         "box, over the SAME depth window, with the SAME "
                         "per-column median, so the only thing differing "
                         "between the tables is the model.")
    a = ap.parse_args()

    def _box(lat, lon):
        return ((np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_west)
                & (lon <= a.lon_east))

    avm, avt, bn2, lat, lon, z = _load(a.nemo_gridw, a.rec, a.lat_halfwidth)
    band = _box(lat, lon)
    n = int(band.sum())
    if n == 0:
        raise SystemExit("no cells in the requested box")
    print(f"NEMO {a.nemo_gridw.name} record {a.rec}")
    print(f"box |lat| <= {a.lat_halfwidth}, {a.lon_west:.0f}-{a.lon_east:.0f}E "
          f"-> {n} columns\n")
    # MEDIANS, not means.  NEMO's convective adjustment replaces the closure's
    # diffusivity with a sentinel of order 10-100 m2/s wherever a column is
    # unstable, and a handful of such cells drags a mean far outside any
    # physical diffusivity -- the first version of this probe reported an
    # equatorial avt of 28 m2/s, which is roughly a thousand times anything
    # the real ocean does, and would have been quoted as the oracle's value.
    # The convective fraction is printed alongside so the reader can see how
    # much of the box the sentinel is covering.
    nemo = _table(avm, avt, bn2, band, z, a.depth_max, "NEMO")

    # --- ours, on the SAME box, the SAME depth window, the SAME reduction ---
    for snap in a.legoesm_snapshot or []:
        oavm, oavt, obn2, olat, olon, oz = _load_ours(Path(snap))
        oband = _box(olat, olon)
        if not oband.any():
            raise SystemExit(f"{snap}: no cells in the requested box")
        print(f"\n############ {Path(snap).parent.name} ############")
        print(f"{snap}\nbox -> {int(oband.sum())} columns  "
              f"(N2 column blank: stratification is "
              f"equatorial_n2_vs_nemo.py's measurement, not duplicated here)\n")
        ours = _table(oavm, oavt, obn2, oband, oz, a.depth_max,
                      Path(snap).parent.name)
        if ours and nemo:
            print(f"\n[{Path(snap).parent.name} vs NEMO]  ours vs NEMO (xratio):")
            for tag, band in (("sfc", "surface 5-65 m"),
                              ("ent", "entrainment 65-105 m")):
                if f"{tag}_pr" not in ours or f"{tag}_pr" not in nemo:
                    continue
                _r = lambda k: (ours[f"{tag}_{k}"] / nemo[f"{tag}_{k}"]
                                if nemo[f"{tag}_{k}"] else float("nan"))
                print(f"  {band:22s}  Pr {ours[f'{tag}_pr']:.2f}/"
                      f"{nemo[f'{tag}_pr']:.2f} (x{_r('pr'):.2f})   "
                      f"avm {ours[f'{tag}_avm']:.3g}/{nemo[f'{tag}_avm']:.3g} "
                      f"(x{_r('avm'):.2f})   "
                      f"avt {ours[f'{tag}_avt']:.3g}/{nemo[f'{tag}_avt']:.3g} "
                      f"(x{_r('avt'):.2f})")
            print("  RATIO defect (avm~1x, avt<<1x) vs STARVATION (both <<1x): "
                  "read avm and avt separately, never their Prandtl ratio "
                  "alone -- a length-scale knob moves both together and cannot "
                  "fix a partition (GLM 2026-08-23).")
            print("  Our snapshot is INSTANTANEOUS and NEMO's record is a "
                  "five-day MEAN; a diffusivity is intermittent, so read a "
                  "factor of two as a factor of two and not more.")

    print("\nNEMO's Prandtl mapping is Pr = min(10, max(1, zri/ri_cri)) with "
          "ri_cri = 2/(2 + rn_ediss/rn_ediff) = 2/(2 + 0.7/0.1) = 0.2222, i.e. "
          "Pr = min(10, max(1, 4.5*zri)) -- read from zdftke.F90:772 and :399, "
          "not from memory. legoESM's prandtl_ri_coeff of 4.5 is therefore "
          "FAITHFUL, so a Prandtl discrepancy has to come from the Richardson "
          "number's INPUTS (N2, avm, shear production), never from the mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
