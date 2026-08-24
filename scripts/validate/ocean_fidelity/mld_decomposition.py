#!/usr/bin/env python
"""T-MLD vs sigma-MLD decomposition: WHY is a mixed layer shallow?

WHY (GLM review, 2026-08-18).  FESOM2's warm-SH / cold-NH growing dipole +
shallow Antarctic MLD fits "under-mixing", but so would a fresh surface lens
(salinity-stratified: density-MLD shoals while the T profile stays deep) or a
vertical-coordinate artifact.  The discriminator is cheap and offline:

  * CLOSURE UNDER-MIXING  -> T-MLD and sigma-MLD are BOTH shallow, together
    (one sharp base set by where mixing stopped).
  * FRESH LENS / COORD    -> DECOUPLED: sigma-MLD << T-MLD, and the
    temperature-only density MLD (S held at its 10 m value) stays deep.

For each snapshot this prints, per latitude band, area-weighted means of
  MLD_sigma   : full de Boyer Montegut density threshold (0.01 wrt 10 m,
                the NEMO mldr10_1 convention used across this campaign);
  MLD_sigmaT  : same threshold with S frozen at the 10 m value -- the
                TEMPERATURE contribution to the stratification;
  MLD_T       : plain temperature threshold (|T - T_10m| >= 0.2 C, dBM).
plus the ratio sigma/sigmaT that carries the verdict: ~1 = temperature
controls the base (closure story); << 1 = salinity controls it (lens story).

Snapshots are the campaign's node-cloud/curvilinear npz convention
(T, S, lat_T, lon_T, z_center_ref, land_mask, H_bathy).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

BANDS = {
    "antarctic_S_of_45S": (-90.0, -45.0),
    "SH_midlat_45S_23S": (-45.0, -23.0),
    "NH_midlat_23N_45N": (23.0, 45.0),
    "arctic_N_of_45N": (45.0, 90.0),
}


def t_threshold_mld(T, z, wet, H, dT=0.2, ref_depth_m=10.0):
    """dBM temperature-criterion MLD: shallowest z>10 m with |T-T10| >= dT.

    LEVEL-RESOLUTION only (first crossing LEVEL depth, no sub-level
    interpolation): biased shallow by up to half a cell, which cancels in the
    sigma/sigmaT RATIO this probe reads.  Fully-mixed columns take the
    sea-floor depth (dBM convention, same as the density MLD).
    """
    T = np.where(wet > 0.5, T, np.nan)
    nlev = z.size
    Tref = np.empty(T.shape[:-1])
    it = np.searchsorted(z, ref_depth_m)
    it = min(max(it, 1), nlev - 1)
    w = (ref_depth_m - z[it - 1]) / (z[it] - z[it - 1])
    Tref = (1 - w) * T[..., it - 1] + w * T[..., it]
    dTa = np.abs(T - Tref[..., None])
    below = z[None, None, :] > ref_depth_m if T.ndim == 3 else z[None, :] > ref_depth_m
    hit = (dTa >= dT) & below & np.isfinite(dTa)
    first = np.where(hit.any(axis=-1), hit.argmax(axis=-1), -1)
    out = np.where(first >= 0, z[np.clip(first, 0, nlev - 1)], H)
    out = np.where(np.isfinite(Tref), out, np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="append", default=[],
                    help="label=path.npz (repeatable)")
    ap.add_argument("--nemo-gridt", default=None,
                    help="ORCA1 monthly grid_T -- the ORACLE's own ratio, the "
                         "control without which a model's S-controlled band "
                         "cannot be called a defect (Antarctic summer melt "
                         "water IS salinity-stratified in the real ocean).")
    ap.add_argument("--nemo-month", type=int, default=1)
    ap.add_argument("--delta-sigma", type=float, default=0.01)
    a = ap.parse_args()
    if not a.snapshot and not a.nemo_gridt:
        raise SystemExit("nothing to do: give --snapshot and/or --nemo-gridt")

    from legoesm.ocean.diagnostics import mixed_layer_depth

    specs = list(a.snapshot)
    if a.nemo_gridt:
        specs.append(f"NEMO_m{a.nemo_month:02d}=__NEMO__")

    for spec in specs:
        label, _, path = spec.partition("=")
        if path == "__NEMO__":
            import xarray as xr
            ds = xr.open_dataset(a.nemo_gridt, decode_times=False)
            from compare_omip_nemo import _nemo_record_months
            tdim = ds["to"].dims[0]
            nt = int(ds.sizes[tdim])
            months = _nemo_record_months(ds, tdim, nt)
            idx = [i for i in range(nt) if months[i] == a.nemo_month]
            T = np.nanmean(np.asarray(ds["to"].values)[idx], axis=0)
            S = np.nanmean(np.asarray(ds["so"].values)[idx], axis=0)
            T = np.where((np.abs(T) > 1e10) | (T == 0.0), np.nan, T)
            S = np.where((np.abs(S) > 1e10) | (S == 0.0), np.nan, S)
            # (nlev, ny, nx) -> (ny, nx, nlev), matching the snapshot layout
            T = np.moveaxis(T, 0, -1)
            S = np.moveaxis(S, 0, -1)
            lat = np.asarray(ds["nav_lat"].values, dtype=np.float64)
            lon2d = np.asarray(ds["nav_lon"].values, dtype=np.float64)
            zc = np.asarray(ds["deptht"].values, dtype=np.float64)
            mask = np.isfinite(T[..., 0]).astype(np.float64)
            wetlev = np.isfinite(T)
            # bottom depth = deepest wet level centre (adequate: only used
            # for fully-mixed columns)
            kbot = np.where(wetlev.any(-1), wetlev.sum(-1) - 1, 0)
            H = zc[np.clip(kbot, 0, zc.size - 1)]
            z0 = {"lon_T": lon2d}
            path = a.nemo_gridt
        else:
            z0 = np.load(path)
            T = np.asarray(z0["T"], dtype=np.float64)
            S = np.asarray(z0["S"], dtype=np.float64)
            lat = np.asarray(z0["lat_T"], dtype=np.float64)
            zc = np.abs(np.asarray(z0["z_center_ref"], dtype=np.float64))
            H = np.asarray(z0["H_bathy"], dtype=np.float64)
            mask = np.asarray(z0["land_mask"], dtype=np.float64)
        wet = ((zc[(None,) * H.ndim + (slice(None),)] < H[..., None])
               & (mask[..., None] > 0.5)).astype(np.float64)

        mld_sig = np.asarray(mixed_layer_depth(
            T, S, zc, delta_sigma=a.delta_sigma, wet_mask=wet,
            bottom_depth=H))
        # S frozen at its 10 m value: the TEMPERATURE-only stratification.
        it = int(np.searchsorted(zc, 10.0))
        S10 = S[..., min(it, zc.size - 1)]
        S_frozen = np.broadcast_to(S10[..., None], S.shape)
        mld_sigT = np.asarray(mixed_layer_depth(
            T, S_frozen, zc, delta_sigma=a.delta_sigma, wet_mask=wet,
            bottom_depth=H))
        mld_T = t_threshold_mld(T, zc, wet, H)

        # NODE MEANS LIE ON A REFINED MESH (caught 2026-08-18: FESOM's
        # shelf-concentrated nodes dominated the Antarctic band and
        # manufactured a 0.03 ratio the open ocean does not have).  Regrid
        # every MLD field to a uniform 1-degree grid FIRST, then take
        # cos(lat)-weighted band means -- same treatment for every mesh.
        from compare_omip_nemo import regrid_curv_to_latlon
        tgt_lat = np.arange(-89.5, 90.0, 1.0)
        tgt_lon = np.arange(0.5, 360.0, 1.0)
        lon = np.asarray(z0["lon_T"], dtype=np.float64)

        def rg(f):
            out, flag = regrid_curv_to_latlon(f, lat, lon, mask,
                                              tgt_lat, tgt_lon)
            return np.where(flag > 0.5, out, np.nan)

        mld_sig_g, mld_sigT_g, mld_T_g = rg(mld_sig), rg(mld_sigT), rg(mld_T)
        wgt2 = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones(tgt_lon.size)
        lat2 = tgt_lat[:, None] * np.ones(tgt_lon.size)
        print(f"\n=== {label} ({path}) ===")
        print(f"{'band':22s} {'sigma':>7} {'sigmaT':>7} {'T02':>7} "
              f"{'sig/sigT':>9}  verdict")
        for bn, (lo, hi) in BANDS.items():
            m = (lat2 >= lo) & (lat2 < hi)
            if not m.any():
                print(f"{bn:22s}  (no cells)")
                continue

            def am(f):
                v = f[m]
                ww = wgt2[m]
                good = np.isfinite(v)
                return float((v[good] * ww[good]).sum() / ww[good].sum())

            s_, st_, t_ = am(mld_sig_g), am(mld_sigT_g), am(mld_T_g)
            r = s_ / st_ if st_ > 0 else np.nan
            verdict = ("T-controlled (closure-class)" if r > 0.8
                       else "S-controlled (lens/coord-class)" if r < 0.6
                       else "mixed")
            print(f"{bn:22s} {s_:7.1f} {st_:7.1f} {t_:7.1f} {r:9.2f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
