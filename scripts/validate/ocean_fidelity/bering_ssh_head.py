"""Sea-level head across the Bering, ours vs NEMO's, on the same sector.

WHY THIS AND NOT ANOTHER FEATURE HUNT. NEMO drives +0.1681 Sv INTO the Arctic
through the 66N Pacific sector over its first thirty days and we drive 0.5303
Sv OUT, from an identical resting start with identical temperature, salinity,
forcing and bathymetry. Five candidate causes have been eliminated by reading
-- superseded zonal metrics, a spin-up mismatch, the strait geometry, absent
ice dynamics, absent ice-ocean stress -- and eliminating plausible features one
at a time is ranking guesses.

The leading-order control on a strait throughflow is the sea-level difference
across it. Measuring that BISECTS the problem instead of testing one term:

  heads differ in sign or size  -> the FORCING of the transport is wrong, and
                                   the cause is upstream of the strait in the
                                   basin-scale mass distribution;
  heads agree but transports do -> the RESPONSE to the same forcing is wrong,
                                   and the cause is local: channel friction,
                                   resolution, or the barotropic treatment.

Either answer names a class. Neither requires a NEMO rerun: NEMO's momentum
trends are not in its file_def, so a term-by-term oracle momentum budget does
not exist for this run, but sea surface height is written on both sides.

SECTOR. The longitude band is taken from the transport's OWN definition,
diagnostics_sections.py ``bering_pacific`` = (-180,-140) or (155,180), so the
head and the transport refer to the same water rather than to two boxes that
happen to both be called Bering. The latitude bands sit either side of the 66N
section the transport crosses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# From diagnostics_sections.py: the bering_pacific sector this transport uses.
BERING_LON_BANDS = ((-180.0, -140.0), (155.0, 180.0))
PACIFIC_LAT = (60.0, 65.5)     # south of the 66N section
ARCTIC_LAT = (66.5, 72.0)      # north of it


def _sq(a):
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


def in_sector(lat, lon):
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    band = np.zeros(lon.shape, dtype=bool)
    for lo, hi in BERING_LON_BANDS:
        band |= (lon >= lo) & (lon <= hi)
    return band, lat


def box_mean(field, wet, area, lat, lon, lat_band):
    """Area-weighted mean over wet cells in the sector and latitude band."""
    band, _ = in_sector(lat, lon)
    m = (band & wet & np.isfinite(field)
         & (lat >= lat_band[0]) & (lat <= lat_band[1]))
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    w = area[m]
    return float(np.sum(field[m] * w) / np.sum(w)), n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True,
                   help="our snapshot .npz carrying eta, land_mask, cell_area")
    p.add_argument("--mesh", required=True, help="NEMO mesh_mask / domain_cfg")
    p.add_argument("--nemo-gridt", required=True, help="NEMO 5-day grid_T")
    p.add_argument("--time-idx", type=int, default=5,
                   help="NEMO record; 5 = days 25-30, the transport's window")
    p.add_argument("--out", default=None)
    p.add_argument("--map-png", default=None,
                   help="write a map of ours-minus-NEMO sea level")
    a = p.parse_args(argv)

    import xarray as xr

    snap = dict(np.load(a.snapshot))
    eta = _sq(snap["eta"])
    wet = _sq(snap["land_mask"]) > 0.5
    area = _sq(snap["cell_area"])

    dm = xr.open_dataset(a.mesh, decode_times=False)
    lat = _sq(dm["gphit"].values)
    lon = _sq(dm["glamt"].values)

    # CONTROL: our snapshot is NEMO's full (jpj, jpi) grid, so it should align
    # with the mesh cell for cell with NO window. If it does not, every box
    # below is over the wrong water and the numbers are meaningless -- this is
    # the failure that has bitten this comparison before, so it is asserted
    # rather than assumed.
    if lat.shape != eta.shape:
        raise SystemExit(
            f"FATAL: mesh {lat.shape} does not match snapshot {eta.shape}; "
            "this probe assumes the snapshot carries NEMO's full grid. "
            "Refusing to guess a window.")
    tmask = _sq(dm["tmaskutil"].values if "tmaskutil" in dm
                else dm["tmask"].values) > 0.5
    agree = float((tmask == wet).mean())
    if agree < 0.99:
        raise SystemExit(
            f"FATAL: our land mask agrees with the mesh on only {agree:.4f} "
            "of cells; the grids are not aligned.")

    ours_pac, n_op = box_mean(eta, wet, area, lat, lon, PACIFIC_LAT)
    ours_arc, n_oa = box_mean(eta, wet, area, lat, lon, ARCTIC_LAT)

    dt = xr.open_dataset(a.nemo_gridt, decode_times=False)
    ssh_name = next((v for v in ("sossheig", "zos", "ssh", "sshn")
                     if v in dt.variables), None)
    if ssh_name is None:
        raise SystemExit(
            f"FATAL: {a.nemo_gridt} has no sea-surface-height variable; "
            f"looked for sossheig/zos/ssh/sshn among {list(dt.variables)[:20]}")
    ssh = _sq(dt[ssh_name].isel(time_counter=a.time_idx).values)
    nlat = _sq(dt["nav_lat"].values)
    nlon = _sq(dt["nav_lon"].values)
    # NEMO writes fill values on land; a wet test on SSH alone is unreliable,
    # so wetness comes from finiteness AND a plausible range.
    nwet = np.isfinite(ssh) & (np.abs(ssh) < 20.0)
    narea = np.ones_like(ssh)   # NEMO's own e1t*e2t is not in this file;
    # an unweighted mean over a small box is a different reduction from ours,
    # so ours is ALSO reported unweighted below as the like-for-like number.

    nemo_pac, n_np = box_mean(ssh, nwet, narea, nlat, nlon, PACIFIC_LAT)
    nemo_arc, n_na = box_mean(ssh, nwet, narea, nlat, nlon, ARCTIC_LAT)

    # LIKE FOR LIKE: unweighted on both sides, since NEMO's cell areas are not
    # in this file. The area-weighted pair is reported beside it so a reader
    # can see the reduction does not carry the result.
    ones = np.ones_like(eta)
    ours_pac_u, _ = box_mean(eta, wet, ones, lat, lon, PACIFIC_LAT)
    ours_arc_u, _ = box_mean(eta, wet, ones, lat, lon, ARCTIC_LAT)

    res = {
        "ours_pacific_m": ours_pac, "ours_arctic_m": ours_arc,
        "ours_head_m": ours_pac - ours_arc,
        "ours_pacific_unweighted_m": ours_pac_u,
        "ours_arctic_unweighted_m": ours_arc_u,
        "ours_head_unweighted_m": ours_pac_u - ours_arc_u,
        "nemo_pacific_m": nemo_pac, "nemo_arctic_m": nemo_arc,
        "nemo_head_m": nemo_pac - nemo_arc,
        "n_ours_pacific": n_op, "n_ours_arctic": n_oa,
        "n_nemo_pacific": n_np, "n_nemo_arctic": n_na,
        "ssh_variable": ssh_name, "time_idx": a.time_idx,
        "mask_agreement": agree,
    }

    print(f"[control] land-mask agreement with mesh: {agree:.4f}")
    print(f"[control] cell counts  ours {n_op}/{n_oa}   NEMO {n_np}/{n_na}")
    print(f"[control] NEMO ssh variable: {ssh_name}, record {a.time_idx}")
    print("")
    print("Sea level, Pacific side minus Arctic side, bering_pacific sector.")
    print("POSITIVE head = Pacific stands higher = drives flow INTO the Arctic.")
    print(f"  ours  pacific {ours_pac:+.4f}  arctic {ours_arc:+.4f}  "
          f"HEAD {ours_pac - ours_arc:+.4f} m")
    print(f"  ours  (unweighted, like-for-like)            "
          f"HEAD {ours_pac_u - ours_arc_u:+.4f} m")
    print(f"  NEMO  pacific {nemo_pac:+.4f}  arctic {nemo_arc:+.4f}  "
          f"HEAD {nemo_pac - nemo_arc:+.4f} m")
    print("")
    print("READ: heads differing in SIGN => the transport's FORCING is wrong "
          "and the cause is upstream. Heads AGREEING => the RESPONSE is wrong "
          "and the cause is local to the channel.")

    # ---- DECOMPOSITION: is the excess Arctic-specific or global? ----------
    # A uniform offset is a mass/volume bookkeeping difference (free-surface
    # reference, global freshwater budget) and says nothing about the Arctic.
    # A structured one is dynamical. The sector numbers already hint at both:
    # our Pacific side sits 2 cm high and our Arctic side 29 cm high, so there
    # is a small global term AND a large Arctic-specific one. This separates
    # them instead of asserting it.
    #
    # Our snapshot is NEMO's FULL (332,362) grid; NEMO's output is (331,360).
    # The window is verified against NEMO's own coordinates, never assumed --
    # an unverified window is how four earlier answers on this section were
    # wrong.
    j0, i0 = 0, 1
    ny_o, nx_o = ssh.shape
    ow = eta[j0:j0 + ny_o, i0:i0 + nx_o]
    lw = lat[j0:j0 + ny_o, i0:i0 + nx_o]
    gw = lon[j0:j0 + ny_o, i0:i0 + nx_o]
    wetw = wet[j0:j0 + ny_o, i0:i0 + nx_o]
    if ow.shape != ssh.shape:
        raise SystemExit(f"FATAL: window gave {ow.shape}, need {ssh.shape}")
    dlon = ((gw - nlon + 180.0) % 360.0) - 180.0
    ok = np.isfinite(dlon) & (np.abs(nlat) < 60.0)
    mis = float(np.nanmedian(np.abs(dlon[ok]))) if ok.any() else np.inf
    if not (mis < 1.0e-3):
        raise SystemExit(
            f"FATAL: window (j0={j0}, i0={i0}) gives median |dlon| {mis:.4g} "
            "deg against NEMO's own nav_lon; refusing to difference two grids "
            "that are not the same grid.")
    both = wetw & nwet & np.isfinite(ow) & np.isfinite(ssh)
    diff = np.where(both, ow - ssh, np.nan)
    res["window_median_dlon_deg"] = mis
    # AREA WEIGHTS ARE NOT OPTIONAL ON A TRIPOLE GRID. An unweighted mean
    # massively oversamples the Arctic, where eORCA1's cells are small and the
    # fold packs many of them, and it is exactly the failure CLAUDE.md names:
    # "a global statistic on a non-uniform grid needs area weights". The first
    # version of this block used np.nanmean and returned a global mean sea
    # level near -0.31 m on both sides, which is not a physical free-surface
    # mean and was the tell.
    aw = area[j0:j0 + ny_o, i0:i0 + nx_o]
    wsum = float(np.nansum(np.where(both, aw, 0.0)))

    def _gmean(f):
        return float(np.nansum(np.where(both, f * aw, 0.0)) / wsum)

    res["global_mean_ours_m"] = _gmean(ow)
    res["global_mean_nemo_m"] = _gmean(ssh)
    res["global_mean_diff_m"] = _gmean(np.where(both, ow - ssh, 0.0))
    res["global_mean_ours_unweighted_m"] = float(np.nanmean(ow[both]))
    res["global_mean_nemo_unweighted_m"] = float(np.nanmean(ssh[both]))
    res["global_area_m2"] = wsum
    print("")
    print(f"[window] verified: median |dlon| {mis:.2e} deg over {int(ok.sum())} cells")
    print(f"[global] AREA-WEIGHTED ours {res['global_mean_ours_m']:+.4f}  "
          f"NEMO {res['global_mean_nemo_m']:+.4f}  "
          f"DIFF {res['global_mean_diff_m']:+.4f} m   "
          f"(area {wsum:.3e} m2)")
    print(f"[global] unweighted, for contrast: ours "
          f"{res['global_mean_ours_unweighted_m']:+.4f}  NEMO "
          f"{res['global_mean_nemo_unweighted_m']:+.4f} m")
    print("[zonal] AREA-WEIGHTED mean(ours - NEMO) by latitude band:")
    for lo, hi in ((-90, -60), (-60, -30), (-30, 30), (30, 60),
                   (60, 70), (70, 80), (80, 90)):
        m = both & (nlat >= lo) & (nlat < hi)
        if int(m.sum()) < 10:
            continue
        ws = float(np.nansum(np.where(m, aw, 0.0)))
        v = float(np.nansum(np.where(m, diff * aw, 0.0)) / ws)
        res[f"zonal_diff_{lo}_{hi}"] = v
        res[f"zonal_area_{lo}_{hi}"] = ws
        print(f"   {lo:+4d}..{hi:+4d}  {v:+.4f} m   "
              f"(n={int(m.sum())}, area {ws:.3e} m2)")

    # ---- FALSIFIER FOR THE FRESHWATER READING --------------------------
    # The Arctic sea-level excess is structured and sits on the Siberian side,
    # which LOOKS like a freshwater accumulation. That is a pattern fitting a
    # story, which this session has twice mistaken for evidence. The mechanism
    # makes a checkable prediction: if freshwater piled up there, our surface
    # must be FRESHER than NEMO's IN THE SAME PLACE. Co-location is the test;
    # a fresh anomaly somewhere else, or none at all, refutes it.
    sal_name = next((v for v in ("so", "vosaline", "soce", "salinity")
                     if v in dt.variables), None)
    if sal_name is None:
        raise SystemExit(
            "FATAL: no salinity variable in the grid_T file; looked for "
            f"so/vosaline/soce/salinity among {sorted(dt.variables)}.")
    if "S" in snap:
        our_s = np.asarray(snap["S"])[..., 0]                 # surface level
        nem_s = _sq(dt[sal_name].isel(time_counter=a.time_idx).values[0]
                    if dt[sal_name].ndim == 4 else
                    dt[sal_name].isel(time_counter=a.time_idx).values)
        osw = our_s[j0:j0 + ny_o, i0:i0 + nx_o]
        sboth = both & np.isfinite(osw) & np.isfinite(nem_s) & (nem_s > 1.0)
        dS = np.where(sboth, osw - nem_s, np.nan)
        res["salinity_variable"] = sal_name
        print("")
        print(f"[salinity] surface, ours - NEMO ({sal_name}), by band:")
        for lo, hi in ((30, 60), (60, 70), (70, 80), (80, 90)):
            m = sboth & (nlat >= lo) & (nlat < hi)
            if int(m.sum()) < 10:
                continue
            v = float(np.nanmean(dS[m]))
            res[f"dS_{lo}_{hi}"] = v
            print(f"   {lo:+4d}..{hi:+4d}  {v:+.4f} psu   (n={int(m.sum())})")
        # THE CO-LOCATION TEST, stated as a number rather than read off a map:
        # correlation between the sea-level excess and the salinity anomaly
        # over the Arctic. Freshwater accumulation predicts a NEGATIVE
        # correlation -- higher where fresher. Near zero refutes it.
        m = sboth & (nlat >= 70.0) & np.isfinite(diff) & np.isfinite(dS)
        if int(m.sum()) > 100:
            c = float(np.corrcoef(diff[m], dS[m])[0, 1])
            res["corr_ssh_excess_vs_dS_north70"] = c
            res["n_corr"] = int(m.sum())
            print(f"[co-location] corr(sea-level excess, salinity anomaly) "
                  f"north of 70N = {c:+.3f}  (n={int(m.sum())})")
            print("   freshwater accumulation predicts NEGATIVE (higher where "
                  "fresher); near zero refutes it.")

    # ---- THE OTHER STERIC TERM ------------------------------------------
    # The salinity test refuted freshwater accumulation: north of 70N the
    # sea-level excess and the salinity anomaly are uncorrelated (-0.034), and
    # in 70-80N our water is SALTIER, which would stand LOWER, not higher.
    # Density has one other lever. Warm water stands higher, so if the excess
    # is thermosteric our Arctic must be WARMER in the same cells. If it is
    # neither, the excess is MASS -- water actually piled up -- and the cause
    # is dynamical convergence rather than any surface buoyancy flux.
    # bigthetao is Conservative Temperature, which is what NEMO writes under
    # ln_teos10 -- ORCA1 sets it. Omitting it is why the first version of this
    # block found nothing.
    tem_name = next((v for v in ("bigthetao", "votemper", "thetao", "toce",
                                 "temperature", "tos") if v in dt.variables),
                    None)
    if tem_name is None:
        raise SystemExit(
            "FATAL: no temperature variable in the grid_T file; looked for "
            "bigthetao/votemper/thetao/toce/temperature/tos among "
            f"{sorted(dt.variables)}. A SILENT SKIP here is how the first run "
            "of this block produced no output and no error.")
    if "T" in snap:
        our_t = np.asarray(snap["T"])[..., 0]
        nem_t = _sq(dt[tem_name].isel(time_counter=a.time_idx).values[0]
                    if dt[tem_name].ndim == 4 else
                    dt[tem_name].isel(time_counter=a.time_idx).values)
        otw = our_t[j0:j0 + ny_o, i0:i0 + nx_o]
        tboth = both & np.isfinite(otw) & np.isfinite(nem_t) & (nem_t > -5.0)
        dT = np.where(tboth, otw - nem_t, np.nan)
        res["temperature_variable"] = tem_name
        print("")
        print(f"[temperature] surface, ours - NEMO ({tem_name}), by band:")
        for lo, hi in ((30, 60), (60, 70), (70, 80), (80, 90)):
            m = tboth & (nlat >= lo) & (nlat < hi)
            if int(m.sum()) < 10:
                continue
            v = float(np.nanmean(dT[m]))
            res[f"dT_{lo}_{hi}"] = v
            print(f"   {lo:+4d}..{hi:+4d}  {v:+.4f} degC   (n={int(m.sum())})")
        m = tboth & (nlat >= 70.0) & np.isfinite(diff) & np.isfinite(dT)
        if int(m.sum()) > 100:
            ct = float(np.corrcoef(diff[m], dT[m])[0, 1])
            res["corr_ssh_excess_vs_dT_north70"] = ct
            print(f"[co-location] corr(sea-level excess, temperature anomaly) "
                  f"north of 70N = {ct:+.3f}  (n={int(m.sum())})")
            print("   thermosteric predicts POSITIVE (higher where warmer); "
                  "near zero with the salinity test already null leaves MASS.")

    # ---- THE RIGOROUS STERIC TEST: FULL COLUMN, NOT THE SURFACE ---------
    # Both co-location tests above are SURFACE fields, and sea level is set by
    # the whole column. A surface-only null is suggestive, not decisive. NEMO's
    # grid_T carries 3-D salinity (the surface tests used its top level), so
    # the halosteric term can be done properly; it only carries surface
    # temperature here, so the thermosteric term cannot.
    #
    # In the Arctic halosteric dominates thermosteric, and the surface result
    # already points the wrong way (we are SALTIER where we stand HIGHER), so
    # this is the test that can actually overturn or confirm the mass reading.
    if "S" in snap and sal_name is not None:
        nem_s3 = np.asarray(dt[sal_name].isel(time_counter=a.time_idx).values)
        while nem_s3.ndim > 3:
            nem_s3 = nem_s3[0]
        if nem_s3.ndim == 3:
            our_s3 = np.asarray(snap["S"])[j0:j0 + ny_o, i0:i0 + nx_o, :]
            nem_s3 = np.moveaxis(nem_s3, 0, -1)          # (y, x, z)
            nz = min(our_s3.shape[-1], nem_s3.shape[-1])
            dS3 = our_s3[..., :nz] - nem_s3[..., :nz]
            # beta from the model's OWN equation of state, not an invented
            # constant: the same Roquet polynomial the closure uses.
            from legoesm.ocean.eos import nemo_roquet_alpha_beta
            import jax.numpy as jnp
            dz_ref = np.asarray(dm["e3t_0"].values)
            while dz_ref.ndim > 3:
                dz_ref = dz_ref[0]
            dzc = np.moveaxis(dz_ref, 0, -1)[j0:j0 + ny_o,
                                             i0:i0 + nx_o, :nz]
            depth = np.cumsum(dzc, axis=-1) - 0.5 * dzc
            _, beta = nemo_roquet_alpha_beta(
                jnp.asarray(np.nan_to_num(nem_s3[..., :nz] * 0.0 + 0.0)),
                jnp.asarray(np.nan_to_num(nem_s3[..., :nz], nan=35.0)),
                jnp.asarray(depth))
            beta = np.asarray(beta)
            valid = np.isfinite(dS3) & np.isfinite(dzc) & (dzc > 1e-6)
            # halosteric height difference: MORE salt => LOWER sea level
            hal = -np.nansum(np.where(valid, beta * dS3 * dzc, 0.0), axis=-1)
            res["halosteric_variable"] = sal_name
            print("")
            print("[halosteric] FULL-COLUMN height difference implied by the "
                  "salinity difference (negative = our column stands LOWER):")
            for lo, hi in ((30, 60), (60, 70), (70, 80), (80, 90)):
                m = both & (nlat >= lo) & (nlat < hi) & np.isfinite(hal)
                if int(m.sum()) < 10:
                    continue
                v = float(np.nanmean(hal[m]))
                res[f"halosteric_{lo}_{hi}"] = v
                obs = res.get(f"zonal_diff_{lo}_{hi}", float("nan"))
                print(f"   {lo:+4d}..{hi:+4d}  halosteric {v:+.4f} m   vs "
                      f"OBSERVED sea-level excess {obs:+.4f} m")
            print("   If halosteric is small or NEGATIVE where the excess is "
                  "POSITIVE, density cannot be the cause and the excess is "
                  "MASS.")

    if a.map_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        im0 = ax[0].pcolormesh(diff, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[0].set_title("sea level, ours - NEMO (m), global", fontsize=9)
        plt.colorbar(im0, ax=ax[0], shrink=0.85)
        arc = np.where(nlat >= 60.0, diff, np.nan)
        im1 = ax[1].pcolormesh(arc, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[1].set_title("same, 60N and north", fontsize=9)
        ax[1].set_ylim(np.argmax((nlat >= 60).any(axis=1)), diff.shape[0])
        plt.colorbar(im1, ax=ax[1], shrink=0.85)
        fig.suptitle(
            "RED = our sea surface stands HIGHER than NEMO's. Day 25-30, both "
            "from rest. The Bering head is set by the Arctic side.",
            fontsize=10)
        if "corr_ssh_excess_vs_dS_north70" in res:
            fig2, bx = plt.subplots(1, 2, figsize=(13, 4.6))
            ja = int(np.argmax((nlat >= 60).any(axis=1)))
            for _b, _f, _t, _v in (
                    (bx[0], diff, "sea level, ours - NEMO (m)", 0.5),
                    (bx[1], dS, "surface salinity, ours - NEMO (psu)", 3.0)):
                _im = _b.pcolormesh(np.where(nlat >= 60.0, _f, np.nan),
                                    vmin=-_v, vmax=_v, cmap="RdBu_r")
                _b.set_title(_t + "  (60N and north)", fontsize=9)
                _b.set_ylim(ja, diff.shape[0])
                plt.colorbar(_im, ax=_b, shrink=0.85)
            fig2.suptitle(
                "CO-LOCATION TEST. Freshwater accumulation predicts the "
                "sea-level high (left, red) to sit on a FRESH anomaly "
                "(right, BLUE). corr north of 70N = "
                f"{res['corr_ssh_excess_vs_dS_north70']:+.3f}", fontsize=10)
            fig2.tight_layout()
            _pth = str(a.map_png).replace(".png", "_colocation.png")
            fig2.savefig(_pth, dpi=110)
            print(f"[map] {_pth}")
        fig.tight_layout()
        Path(a.map_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.map_png, dpi=110)
        print(f"[map] {a.map_png}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
