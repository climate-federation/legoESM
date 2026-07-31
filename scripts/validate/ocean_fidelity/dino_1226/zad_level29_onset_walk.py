"""#1226 dyn_adv ZAD -- level-29 ONSET walk (READ-ONLY measurement).

TARGET: the same open ``dyn_adv ZAD`` row as ``zad_recurrence_walk.py``/
``zad_vertical_metric_walk.py``/``dyn_zad_ldf_walk.py``/
``ww_inheritance_walk.py`` (all UNTOUCHABLE, findings cited not re-derived).

TASK A (harness fix, cheap, done first): ``zad_recurrence_walk.py`` STEP C
already built the population decomposition (commit d2d680d8d) -- the
headline err_norm's mask is the SURFACE u-mask broadcast to depth, which
includes below-seafloor cells NEMO's own dump carries as leftover Krhs
(``dynzad.F90:86`` has no per-level umask guard; NEMO discards it later at
``dynzdf.F90:121``). This script re-runs that exact decomposition (does not
re-derive it) and prints the corrected active-only row value + per-level
profile for the record.

TASK B (the real target): at Python level 29 there are ZERO masked/inactive
cells, yet S_act jumps ~2000x in RMS from level 28 to 29. This script:
  1. censuses the jk 29/30 transition (bathymetry shelf count, e3 ratio,
     gdepw, mi96 stretch parameters, any DINO bathymetry feature at ~1900 m
     besides the ~625 m sill),
  2. maps the level-29 error spatially (few columns vs basin-wide),
  3. decomposes ZAD's own inputs (ww, shear, e3u(Kmm), area factors, the
     assembled flux zWdzU) AT LEVEL 29 ONLY against NEMO's own dumps,
  4. requires magnitude arithmetic predicting the ~2000x jump before
     naming a cause.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.zad_level29_onset_walk
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import netCDF4 as nc
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, _u_east_to_face_periodic,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_geometry_and_density, _bc_vertical_and_depthmean_velocity,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    interp_cell_to_uface, compute_face_masks_3d,
)
from legoesm.ocean.vertical import OceanPartialCellCoordinate
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale (not re-derived): established RUN_DIR/DT/loader.
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import RUN_DIR, DT, _load_full_3d

register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment (stock dynzad.F90:86-119).")

for _name in ("zad_dump_du.bin",):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _err_norm(lego3, nemo3, mask2d):
    """IDENTICAL formula to zad_recurrence_walk.py/dyn_zad_ldf_walk.py's
    ``_err_norm`` (verified by inspection before use, metric-mixing rule)."""
    n_lat_c = min(lego3.shape[0], nemo3.shape[0], mask2d.shape[0])
    n_lon_c = min(lego3.shape[1], nemo3.shape[1], mask2d.shape[1])
    n_lev_c = min(lego3.shape[2], nemo3.shape[2])
    lo = np.asarray(lego3)[:n_lat_c, :n_lon_c, :n_lev_c]
    ne = np.asarray(nemo3)[:n_lat_c, :n_lon_c, :n_lev_c]
    m = mask2d[:n_lat_c, :n_lon_c]
    err = lo - ne
    err_by_level = np.array([
        float(np.sqrt(np.nanmean(err[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    rms_by_level = np.array([
        float(np.sqrt(np.nanmean(ne[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    tot_err_norm = (float(np.sqrt(np.mean(err_by_level ** 2)))
                    / float(np.sqrt(np.mean(rms_by_level ** 2))))
    max_abs = float(np.nanmax(np.abs(err)))
    return err, err_by_level, rms_by_level, tot_err_norm, max_abs


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zad_level29_onset_walk")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both' for this walk)")
    assert e3t_mode == "both", "run with LEGOESM_NEMO_E3T=both (task rule)"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    restart_path = os.path.join(RUN_DIR, "DINO_00057600_restart.nc")
    print(f"restart path: {restart_path}  step kt=57601 (established #1226 probe point)")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(restart_path, nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="zad_level29_onset_walk")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)

    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    u_mask_3d_np = np.asarray(g.umask) > 0.5

    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)

    # =========================================================================
    # SELF-CHECK 1 (mandatory, first): reproduce recorded headline (3.9993e-2
    # union) AND active-only (3.0332e-2) numbers before anything else.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce recorded baselines (union + active-only)")
    print("=" * 78)
    err_u0, ebl_u0, rbl_u0, tot_u0, maxabs_u0 = _err_norm(vertadv_u_f, nemo_zad_du, umask2)
    print(f"  u UNION err_norm={tot_u0:.4e}  (recorded 3.9993e-2)  max|diff|={maxabs_u0:.4e}")
    check1a = abs(tot_u0 - 3.9993e-2) < 2e-3

    n_lat_d = min(err_u0.shape[0], u_mask_3d_np.shape[0], umask2.shape[0])
    n_lon_d = min(err_u0.shape[1], u_mask_3d_np.shape[1], umask2.shape[1])
    n_lev_d = min(err_u0.shape[2], u_mask_3d_np.shape[2])
    err_d = err_u0[:n_lat_d, :n_lon_d, :n_lev_d]
    act_d = u_mask_3d_np[:n_lat_d, :n_lon_d, :n_lev_d]
    wet2d_d = umask2[:n_lat_d, :n_lon_d]
    N = int(wet2d_d.sum())
    denom = float(np.sqrt(np.mean(rbl_u0[:n_lev_d] ** 2)))

    S_act = np.zeros(n_lev_d)
    S_inact = np.zeros(n_lev_d)
    cnt_act = np.zeros(n_lev_d, dtype=int)
    cnt_inact = np.zeros(n_lev_d, dtype=int)
    for k in range(n_lev_d):
        sel_act = wet2d_d & act_d[..., k]
        sel_inact = wet2d_d & ~act_d[..., k]
        S_act[k] = float(np.sum(err_d[..., k][sel_act] ** 2))
        S_inact[k] = float(np.sum(err_d[..., k][sel_inact] ** 2))
        cnt_act[k] = int(sel_act.sum())
        cnt_inact[k] = int(sel_inact.sum())

    num_act = float(np.sqrt(np.mean(S_act / N)))
    num_tot = float(np.sqrt(np.mean((S_act + S_inact) / N)))
    en_act, en_tot = num_act / denom, num_tot / denom
    print(f"  u ACTIVE-ONLY err_norm={en_act:.4e}  (recorded 3.0332e-2)")
    print(f"  reconstructed UNION from decomposition = {en_tot:.4e}  (must equal SELF-CHECK1's {tot_u0:.4e})")
    check1b = abs(en_act - 3.0332e-2) < 2e-3
    check1_ok = check1a and check1b
    print(f"  SELF-CHECK 1: {'PASSED' if check1_ok else 'FAILED -- STOP, harness unverified'}")
    if not check1_ok:
        return 1

    # =========================================================================
    # TASK A: report the harness-corrected row value (active-only cells,
    # i.e. the population NEMO actually keeps) + per-level profile.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK A (HARNESS FIX, not production): u err_norm restricted to the")
    print("3-D umask population (cells NEMO's dynzdf.F90:121 keeps), levels 24-35")
    print("=" * 78)
    ebl_act = np.sqrt(S_act / N) / denom
    for k in range(24, min(n_lev_d, 36)):
        print(f"  lev {k:2d} (jk {k+1:2d}): n_active={cnt_act[k]:5d}  n_inactive_wet2d={cnt_inact[k]:5d}  "
              f"err_by_level(active)={ebl_act[k]:.4e}")
    print(f"\n  TASK A HEADLINE (active-only, harness-corrected): {en_act:.4e}")
    print("  This is a HARNESS fix only -- production legoESM already zeros")
    print("  below-seafloor u-faces correctly (max|vertadv_u|=0.0 there, prior")
    print("  finding); the ORIGINAL union number mixed in cells NEMO itself")
    print("  discards before use. Not recorded to fidelity_bar_gate.py here.")

    # =========================================================================
    # TASK B STEP 1: census the jk 29/30 transition.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK B STEP 1: census of Python level 28 -> 29 -> 30 -> 31 transition")
    print("=" * 78)
    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
        gdepw_1d = np.asarray(ds.variables["gdepw_1d"][0], dtype=np.float64)
        gdept_1d = np.asarray(ds.variables["gdept_1d"][0], dtype=np.float64)
        e3t_1d = np.asarray(ds.variables["e3t_1d"][0], dtype=np.float64)
        mbathy = np.asarray(ds.variables["mbathy"][0], dtype=np.int64)  # NEMO 1-indexed bottom T-level count
        e1t = np.asarray(ds.variables["e1t"][0], dtype=np.float64)
        e2t = np.asarray(ds.variables["e2t"][0], dtype=np.float64)
        e1u = np.asarray(ds.variables["e1u"][0], dtype=np.float64)
        e2v = np.asarray(ds.variables["e2v"][0], dtype=np.float64)

    print("  1-D reference ladder (levels 27-32, python idx, gdepw/gdept in m, e3t_1d in m):")
    for k in range(26, 32):
        print(f"    lev {k:2d} (jk {k+1:2d}): gdepw_1d={gdepw_1d[k]:8.2f}  gdept_1d={gdept_1d[k]:8.2f}  e3t_1d={e3t_1d[k]:.3f}")
    ratio_2928 = float(e3t_1d[29] / e3t_1d[28]) if e3t_1d[28] != 0 else float("nan")
    ratio_3029 = float(e3t_1d[30] / e3t_1d[29]) if e3t_1d[29] != 0 else float("nan")
    print(f"  e3t_1d[29]/e3t_1d[28] = {ratio_2928:.4f}   e3t_1d[30]/e3t_1d[29] = {ratio_3029:.4f}")
    print("  (a smooth tanh stretch predicts a ratio close to 1 here -- no")
    print("  discontinuity in the 1-D reference ladder itself at this level;")
    print("  any per-level jump must come from a 2-D/3-D field, not the ladder.)")

    # Bathymetry shelf census: how many T-columns have their WET COLUMN
    # BOTTOM exactly at level 29/30 (a shelf edge at this depth)?
    bottom_level_t = np.asarray(br.z_coord.bottom_level)  # 0-indexed python bottom T-level per column
    tmask2d = np.asarray(g.tmask)[..., 0] > 0.5
    for lev in (27, 28, 29, 30, 31, 32):
        n_shelf = int(np.sum((bottom_level_t == lev) & tmask2d))
        print(f"  columns whose WET-COLUMN BOTTOM is exactly level {lev:2d} (jk {lev+1}): {n_shelf}")
    n_wet_total = int(tmask2d.sum())
    n_wet_at_29_or_deeper = int(np.sum((bottom_level_t >= 29) & tmask2d))
    print(f"  total wet T-columns: {n_wet_total}; columns reaching level >=29 (still wet AT level 29): {n_wet_at_29_or_deeper}")
    print("  (DINO's documented sill is ~625 m -- levels 27-32 span ~1140-2350 m,")
    print("  well below the sill; this census checks for any SECOND bathymetry")
    print("  feature -- ridge/Drake-sill parameters live in usrdef_nam, not")
    print("  reconstructed here since br.state.H_bathy is read directly from")
    print("  NEMO's own mesh_mask, not re-derived -- so this census uses the")
    print("  ACTUAL bathymetry NEMO ran with, not a legoESM reconstruction.)")

    # =========================================================================
    # TASK B STEP 2: spatial map of the level-29 ACTIVE-ONLY error.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK B STEP 2: spatial localisation of the level-29 error")
    print("=" * 78)
    lev29 = 29
    act29 = act_d[..., lev29] & wet2d_d
    err29 = err_d[..., lev29]
    err29_active = np.where(act29, err29, np.nan)
    abs29 = np.abs(err29_active)
    n_active_29 = int(act29.sum())
    thresh = np.nanpercentile(abs29, 90)
    n_above_90th_contrib = int(np.sum(abs29[act29] >= thresh))
    # Fraction of the total sum-of-squares at level 29 owned by the top-K
    # columns (K = 1%, 5%, 10% of active columns) -- concentrated-in-few-columns
    # vs basin-wide discriminator.
    flat = abs29[act29]
    sq = flat ** 2
    order = np.argsort(sq)[::-1]
    sq_sorted = sq[order]
    cum = np.cumsum(sq_sorted)
    total_sq = cum[-1] if len(cum) else 0.0
    frac_top1pct = float(cum[max(0, int(0.01 * len(sq_sorted)) - 1)] / total_sq) if total_sq > 0 else float("nan")
    frac_top5pct = float(cum[max(0, int(0.05 * len(sq_sorted)) - 1)] / total_sq) if total_sq > 0 else float("nan")
    frac_top10pct = float(cum[max(0, int(0.10 * len(sq_sorted)) - 1)] / total_sq) if total_sq > 0 else float("nan")
    print(f"  n_active columns at level 29: {n_active_29}")
    print(f"  fraction of level-29 sum-of-squared-error owned by:")
    print(f"    top  1% of active columns (by |err|): {frac_top1pct:.4f}")
    print(f"    top  5% of active columns (by |err|): {frac_top5pct:.4f}")
    print(f"    top 10% of active columns (by |err|): {frac_top10pct:.4f}")
    if frac_top1pct > 0.5:
        verdict_spatial = "CONCENTRATED in a handful of columns -- implicates a geometry edge case"
    elif frac_top10pct < 0.3:
        verdict_spatial = "BASIN-WIDE (roughly uniform contribution) -- implicates a formula term, not geometry"
    else:
        verdict_spatial = "INTERMEDIATE -- neither cleanly localised nor cleanly uniform"
    print(f"  VERDICT (spatial): {verdict_spatial}")

    # Cross-tab against periodic seam (i=0/near jpi) and meridional walls
    # (j=0/near jpj), and against bottom_level (shelf-edge proximity) --
    # print raw counts, no verdict baked in beyond the spatial fraction above.
    jj_idx, ii_idx = np.where(act29)
    if len(ii_idx):
        near_seam = int(np.sum((ii_idx <= 2) | (ii_idx >= act29.shape[1] - 3)))
        near_wall = int(np.sum((jj_idx <= 2) | (jj_idx >= act29.shape[0] - 3)))
        print(f"  of {len(ii_idx)} active level-29 columns: {near_seam} within 3 cells of the "
              f"zonal periodic seam; {near_wall} within 3 cells of a meridional wall")
        bl29 = bottom_level_t[:act29.shape[0], :act29.shape[1]][act29]
        near_shelf29 = int(np.sum(np.abs(bl29 - 29) <= 1))
        print(f"  of those, {near_shelf29} have their OWN column bottom_level within 1 of level 29"
              f" (shelf-edge proximity at this specific level)")
        # Where does the error-weighted centroid sit vs the domain?
        wsq = sq
        lat_c = float(np.average(jj_idx, weights=wsq)) if wsq.sum() > 0 else float("nan")
        lon_c = float(np.average(ii_idx, weights=wsq)) if wsq.sum() > 0 else float("nan")
        print(f"  error-weighted centroid (j,i) = ({lat_c:.1f}, {lon_c:.1f})  "
              f"of a domain (n_lat,n_lon)=({act29.shape[0]},{act29.shape[1]})")

    # =========================================================================
    # TASK B STEP 3: per-input decomposition AT LEVEL 29 ONLY.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK B STEP 3: decompose ZAD's own inputs at level 29, vs NEMO dumps")
    print("=" * 78)
    grid = br.geometry
    z_coord = br.z_coord
    T = br.state.T.data
    S = br.state.S.data
    eta = br.state.eta.data
    H_bathy = br.state.H_bathy.data
    mask = br.state.land_mask.data
    u_full = br.state.u.data
    v_full = br.state.v.data

    min_water_col = float(mc.min_water_column_m)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask
    _J, h_k, _rho_prime, _p_prime = _bc_geometry_and_density(
        eta_safe, H_bathy, z_coord, mc, T, S, mask, grid, mc.rho_0, mc.g,
    )
    if isinstance(z_coord, OceanPartialCellCoordinate):
        u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, grid)
    else:
        u_mask_3d = jnp.asarray(g.umask)
        v_mask_3d = jnp.asarray(g.vmask)
    umask_2d = br.state.u_mask.data
    vmask_2d = br.state.v_mask.data
    (h_u_lego, _h_v_lego, _flux_div_k, w, _u_prime, _v_prime) = (
        _bc_vertical_and_depthmean_velocity(
            h_k, u_full, v_full, u_mask_3d, v_mask_3d, grid, z_coord, umask_2d, vmask_2d,
        )
    )
    area_w = grid.area_T[..., jnp.newaxis] * w
    w_area_u = interp_cell_to_uface(area_w)
    face_area_u = (grid.dx_u * grid.dy_u)[..., jnp.newaxis]

    w_np = np.asarray(w)
    w_area_u_np = np.asarray(w_area_u)
    u_full_np = np.asarray(u_full)
    h_u_np = np.asarray(h_u_lego)
    area_T = np.asarray(grid.area_T)
    face_area_u_np = np.asarray(grid.dx_u * grid.dy_u)

    # NEMO-space per-input reference reconstructions (pure-NEMO formulas,
    # not legoESM's operator -- built independently, matching
    # zad_recurrence_walk.py's self-check-3 pattern):
    ssh_now = np.asarray(s.ssh)
    ssh_e = np.roll(ssh_now, -1, axis=1)
    e1e2t = e1t * e2t
    e1e2t_e = np.roll(e1e2t, -1, axis=1)
    e1e2u = e1u * e2t
    r3u = 0.5 * (e1e2t * ssh_now + e1e2t_e * ssh_e) / np.where(g.hu_0 > 0, g.hu_0, 1.0) / np.where(e1e2u > 0, e1e2u, 1.0)
    r3u = np.where(g.hu_0 > 0, r3u, 0.0)
    e3u_kmm_nemo_raw = g.e3u_0 * (1.0 + r3u[..., None] * g.umask)
    e3u_kmm_nemo = np.asarray(_u_east_to_face_periodic(e3u_kmm_nemo_raw))

    # Reconstruct NEMO's ww (T-point vertical velocity at w-faces) is NOT
    # directly dumped by this walk; use legoESM's own w (already validated
    # clean elsewhere, err_norm 5.98e-05, ww_inheritance_walk.py) as the
    # "expected" ww and compare its LEVEL-29 value specifically, since a
    # per-level regression at 29 in an otherwise-clean field would itself be
    # the signature searched for.
    n_lat_i = min(w_np.shape[0], umask2.shape[0])
    n_lon_i = min(w_np.shape[1], umask2.shape[1])

    def _level_stats(arr3, label, mask2, lev):
        a = np.asarray(arr3)
        n_lat_l = min(a.shape[0], mask2.shape[0])
        n_lon_l = min(a.shape[1], mask2.shape[1])
        m = mask2[:n_lat_l, :n_lon_l]
        vals = a[:n_lat_l, :n_lon_l, lev][m]
        vals = vals[np.isfinite(vals)]
        rms28 = float(np.sqrt(np.mean(a[:n_lat_l, :n_lon_l, lev - 1][m] ** 2))) if lev - 1 >= 0 else float("nan")
        rms = float(np.sqrt(np.mean(vals ** 2))) if len(vals) else float("nan")
        ratio = rms / rms28 if rms28 not in (0.0, float("nan")) and not np.isnan(rms28) else float("nan")
        print(f"  {label:40s}: RMS(lev28)={rms28:.4e}  RMS(lev29)={rms:.4e}  ratio(29/28)={ratio:.4f}")
        return rms28, rms, ratio

    # w/w_area_u/G live on the INTERFACE grid (nlev+1=37: k=0 surface .. k=36
    # bottom, both boundary interfaces architecturally zero, vertical.py:1240
    # -1258). u/h_u/vertadv live on the CELL-CENTER grid (nlev=36). "level 29"
    # in the err_by_level profile is CELL-CENTER index 29; the interface
    # inputs that feed cell 29's tendency are G[29] (top) and G[30] (bottom),
    # i.e. w_area_u/dudz interface-indices 29 and 30 -- NOT the same integer
    # index on the two grids. Report both neighbouring interfaces explicitly
    # rather than reusing a naive same-index slice (the bug the shape crash
    # just caught).
    #
    # INDEX-CONVENTION WARNING (found the hard way, self-checked below before
    # trusting anything downstream): legoESM's raw ``br.state.u``/``h_u_lego``
    # arrays carry ONE EXTRA leading u-column vs NEMO/mesh_mask's natural
    # index (the same offset ``_u_to_nemo`` strips via ``[:, 1:]`` for the
    # Krhs comparison everywhere else in this campaign) -- i.e.
    # ``h_u_np[j, i+1, k] == NEMO's u-face (j, i, k)``. Every per-column probe
    # below reads ``ii_shift = ii + 1`` for exactly this reason; indexing
    # ``h_u_np``/``u_full_np`` at NEMO's raw ``ii`` silently returns the WRONG
    # (western-neighbour) column -- verified: it produced an all-zero h_u/u
    # column at a genuinely wet NEMO u-face during this walk's development
    # and was caught by cross-checking against the raw restart ``un``/
    # ``umask``, which were nonzero there. Fixed here; the two hot columns
    # found in STEP 2 are the demonstration.
    print("  RMS magnitude of each ZAD INPUT: T-point w at T-mask, all else at u-face active mask.")
    tmask2 = np.asarray(g.tmask)[..., 0] > 0.5
    _level_stats(w_np, "w (T-point vertical velocity, interface-indexed)", tmask2, lev29)
    _level_stats(w_area_u_np, "w_area_u (u-face, interface idx 29)", umask2, lev29)
    _level_stats(w_area_u_np, "w_area_u (u-face, interface idx 30)", umask2, lev29 + 1)

    du_interior = u_full_np[..., :-1] - u_full_np[..., 1:]           # (..., nlev-1), interface k=1..nlev-1
    dudz_interface = np.pad(du_interior, ((0, 0), (0, 0), (1, 1)))    # (..., nlev+1), zero at k=0,nlev
    _level_stats(dudz_interface, "du/dk shear (u[k-1]-u[k]), interface idx 29", umask2, lev29)
    _level_stats(dudz_interface, "du/dk shear (u[k-1]-u[k]), interface idx 30", umask2, lev29 + 1)
    _level_stats(h_u_np, "h_u (legoESM min-rule divisor, cell idx 29)", umask2, lev29)
    _level_stats(e3u_kmm_nemo, "e3u(Kmm) pure-NEMO divisor, cell idx 29", umask2, lev29)

    # Assembled flux G[k] = 2*w_area_half[k]*(u[k-1]-u[k]) (dynzad.F90:97-100,
    # vertical.py:1244-1247) at the SAME two interfaces feeding cell 29.
    G_full = 2.0 * w_area_u_np * dudz_interface
    _level_stats(G_full, "G = 2*w_area_u*(u[k-1]-u[k]), interface idx 29 (top of cell 29)", umask2, lev29)
    _level_stats(G_full, "G = 2*w_area_u*(u[k-1]-u[k]), interface idx 30 (bottom of cell 29)", umask2, lev29 + 1)

    # ZAD Krhs itself, our side, at level 29 vs 28 (the quantity actually
    # being compared, for direct magnitude reference).
    _level_stats(vertadv_u_f, "vertadv_u (legoESM ZAD Krhs, our side), cell idx 29", umask2, lev29)
    _level_stats(nemo_zad_du, "NEMO ZAD Krhs dump (their side), cell idx 29", umask2, lev29)

    print("\n  Same table but restricted to level 29's ACTIVE-ONLY error hot cells")
    print("  (top 5% by |err| at level 29) vs the REST of level-29 active cells,")
    print("  to see whether the input jump is confined to the same hot columns:")
    act29_full = act_d[..., lev29] & wet2d_d
    err29_full = np.where(act29_full, np.abs(err_d[..., lev29]), np.nan)
    flat_e = err29_full[act29_full]
    if len(flat_e):
        thresh95 = np.nanpercentile(flat_e, 95)
        hot = act29_full & (np.where(act29_full, np.abs(err_d[..., lev29]), -1) >= thresh95)
        cold = act29_full & ~hot
        for arrname, arr in (
            ("w_area_u (iface 29)", w_area_u_np), ("dudz (iface 29)", dudz_interface),
            ("G=2*w_area_u*dudz (iface 29)", G_full),
            ("h_u (lego divisor)", h_u_np), ("e3u(Kmm) NEMO divisor", e3u_kmm_nemo),
        ):
            a = np.asarray(arr)
            n_lat_l = min(a.shape[0], hot.shape[0])
            n_lon_l = min(a.shape[1], hot.shape[1])
            hh = hot[:n_lat_l, :n_lon_l]
            cc = cold[:n_lat_l, :n_lon_l]
            v_hot = a[:n_lat_l, :n_lon_l, lev29][hh]
            v_cold = a[:n_lat_l, :n_lon_l, lev29][cc]
            rms_hot = float(np.sqrt(np.mean(v_hot ** 2))) if len(v_hot) else float("nan")
            rms_cold = float(np.sqrt(np.mean(v_cold ** 2))) if len(v_cold) else float("nan")
            print(f"    {arrname:26s}: RMS(hot n={len(v_hot):4d})={rms_hot:.4e}  "
                  f"RMS(cold n={len(v_cold):4d})={rms_cold:.4e}  "
                  f"hot/cold={rms_hot / rms_cold if rms_cold else float('nan'):.3f}")

    # =========================================================================
    # TASK B STEP 4: magnitude arithmetic -- STRADDLING-BOTTOM MECHANISM.
    #
    # STEP 2 found the level-29 error is owned almost entirely (>99% of the
    # sum-of-squares) by the ``bl_u == 29`` u-faces (the 9 shelf-edge
    # columns). At every such face, one T-neighbour's OWN bottom is level 29
    # and the OTHER's is level 30 (verified below: diff = +/-1 at every one
    # of the 9). NEMO's ``dyn_zad`` (``dynzad.F90:86``, ``DO_3D(0,0,0,0,
    # 1,jpk-2)``) has NO per-face umask guard, so at that face's OWN bottom
    # cell (jk=29, still INTERIOR to the fixed loop since jpk-2=34 > 29) it
    # unconditionally reads ``ww`` from BOTH T-neighbours at interface jk+1
    # =30 -- including the DEEPER neighbour, which is STILL WET there and
    # carries a genuine (non-garbage) nonzero ``ww``. legoESM's
    # ``face_active`` masks G at ANY interface bordering an inactive face
    # (``active_above*active_below``, vertical.py:1249-1252) using the
    # MIN-RULE u-face mask, so it correctly zeros interface 30 for this face
    # (the face itself is considered dead below level 29) and discards that
    # neighbour's real flux entirely.
    #
    # This predicts the FULL NEMO-dumped Krhs at the 3 dominant hot columns
    # (fresh zzWdzU at interface 30 + the CARRIED zWdzU produced at interface
    # 29) using ONLY quantities legoESM itself already computes (w, u, area,
    # e3u) -- no free parameters.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK B STEP 4: STRADDLING-BOTTOM MECHANISM -- magnitude arithmetic")
    print("=" * 78)
    umask_g = np.asarray(g.umask)
    bl_t = np.asarray(z_coord.bottom_level)     # T-point bottom level, python 0-idx
    bl_u_face = np.minimum(bl_t, np.roll(bl_t, -1, axis=1))
    jj_bl29, ii_bl29 = np.where(bl_u_face == lev29)
    print(f"  u-faces with bl_u==29 (the shelf-edge population from STEP 2): "
          f"{list(zip(jj_bl29.tolist(), ii_bl29.tolist()))}")
    diffs = [int(bl_t[jj, (ii + 1) % bl_t.shape[1]] - bl_t[jj, ii]) for jj, ii in zip(jj_bl29, ii_bl29)]
    print(f"  bottom_level(east neighbour) - bottom_level(self) at each: {diffs}")
    print("  (all |diff|==1 => every one of these faces straddles a genuine")
    print("  1-level bathymetry step -- not a coincidence, the defining property")
    print("  of this population.)")

    print("\n  Per-column magnitude prediction (fresh + carried term, no free params):")
    n_i = bl_t.shape[1]
    for jj, ii in zip(jj_bl29.tolist(), ii_bl29.tolist()):
        ii_shift = ii + 1                      # legoESM raw-column offset (see warning above)
        ie = (ii + 1) % n_i                    # NEMO east T-neighbour, natural index
        k_self = int(bl_t[jj, ii])
        if k_self != lev29:
            continue
        k_dead = k_self + 1                    # interface one level below this face's own bottom
        zzwfu_fresh = area_T[jj, ii] * w_np[jj, ii, k_dead] + area_T[jj, ie] * w_np[jj, ie, k_dead]
        du_fresh = u_full_np[jj, ii_shift, k_self]     # u[k_self]-u[k_self+1], u[k_self+1]=0 (masked)
        zzwdzu_fresh = zzwfu_fresh * du_fresh

        zzwfu_carry = area_T[jj, ii] * w_np[jj, ii, k_self] + area_T[jj, ie] * w_np[jj, ie, k_self]
        du_carry = u_full_np[jj, ii_shift, k_self - 1] - u_full_np[jj, ii_shift, k_self]
        zwdzu_carried = zzwfu_carry * du_carry

        e1e2u = float(face_area_u_np[jj, ii_shift])
        e3u_self = float(h_u_np[jj, ii_shift, k_self])
        pred = -0.25 / e1e2u / max(e3u_self, 1e-30) * (zzwdzu_fresh + zwdzu_carried)
        obs = float(nemo_zad_du[jj, ii, k_self]) if jj < nemo_zad_du.shape[0] and ii < nemo_zad_du.shape[1] else float("nan")
        ratio = pred / obs if obs not in (0.0,) and np.isfinite(obs) else float("nan")
        print(f"    (j={jj:3d},i={ii:3d}): fresh={zzwdzu_fresh:.4e}  carried={zwdzu_carried:.4e}  "
              f"predicted Krhs={pred:.4e}  NEMO dumped={obs:.4e}  ratio={ratio:.3f}")

    print("\n  Reference RMS ratio (S_act[29]/S_act[28])**0.5 = "
          f"{np.sqrt(S_act[29] / S_act[28]) if S_act[28] > 0 else float('nan'):.1f}x -- this is a "
          "9-column effect (of 9758 wet columns), so no PER-INPUT level-28-to-29")
    print("  RMS ratio (STEP 3, ~1.0-1.6x pooled) is expected to predict it: the")
    print("  jump is concentrated, not distributed, and STEP 3's pooled RMS")
    print("  necessarily dilutes a 9-column signal by ~1000x (9/9758).")

    # =========================================================================
    # TASK B STEP 5: GENERALISE -- does the SAME mechanism (own-seafloor
    # u-face straddling a bathymetry step) explain the error at EVERY level
    # 29-34, and what fraction of the ENTIRE active-only row does it own?
    # This is the decisive scope check: STEP 4 only proved 9 columns at one
    # level; a real cause must generalise to the whole "jump band" the task
    # named (levels 29-33) and bound how much of the row it actually owns.
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK B STEP 5: GENERALISED SCOPE -- same mechanism at every level,")
    print("fraction of the FULL active-only row explained")
    print("=" * 78)
    bl_u_face_d = bl_u_face[:n_lat_d, :n_lon_d]
    S_total_all = 0.0
    S_own_seafloor_all = 0.0
    for k in range(n_lev_d):
        act_k = wet2d_d & act_d[..., k]
        is_own_seafloor_k = (bl_u_face_d == k) & wet2d_d
        s_all = float(np.sum(err_d[..., k][act_k] ** 2))
        s_own = float(np.sum(err_d[..., k][is_own_seafloor_k & act_k] ** 2))
        S_total_all += s_all
        S_own_seafloor_all += s_own
        if 26 <= k <= 34:
            n_own = int((is_own_seafloor_k & act_k).sum())
            frac_k = s_own / s_all if s_all > 0 else float("nan")
            print(f"  lev {k:2d} (jk {k+1:2d}): n_own_seafloor_faces={n_own:5d}  "
                  f"S(own_seafloor)={s_own:.3e}  S(all active)={s_all:.3e}  frac={frac_k:.4f}")
    frac_total = S_own_seafloor_all / S_total_all if S_total_all > 0 else float("nan")
    print(f"\n  S_total (active-only, ALL levels)              = {S_total_all:.6e}")
    print(f"  S_own-seafloor-u-face (ALL levels)              = {S_own_seafloor_all:.6e}")
    print(f"  FRACTION OF THE ENTIRE ACTIVE-ONLY ROW EXPLAINED = {frac_total:.4f}")
    print("  (own-seafloor u-face = a u-face at exactly its OWN column's")
    print("  bottom_level -- i.e. the same jk=jpkm1-STYLE cell NEMO's fixed")
    print("  DO_3D(0,0,0,0,1,jpk-2) loop still treats as INTERIOR (since")
    print("  jpk-2=34 for every local seafloor at level<=34), reading a still-")
    print("  wet neighbour's ww unmasked -- this is NOT limited to level 29,")
    print("  it recurs at every shelf-edge column throughout the water column.)")

    print("\n" + "=" * 78)
    print("DONE -- see terminal output above for the report's numeric inputs.")
    print("=" * 78)
    return 0 if check1_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
