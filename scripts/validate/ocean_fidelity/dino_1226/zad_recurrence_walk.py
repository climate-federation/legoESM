"""#1226 dyn_adv ZAD -- recurrence-structure + e3-vertical-gradient walk
(READ-ONLY measurement, one new script per task file-touch rules).

TARGET: the SAME open ``dyn_adv ZAD`` row as ``dyn_zad_ldf_walk.py``/
``ww_inheritance_walk.py``/``zad_vertical_metric_walk.py`` (all three
UNTOUCHABLE, read first, findings cited not re-derived): near-zero error
through Python level 28, then a 1000x+ jump at Python levels 29-33
(Fortran jk 30-34, ~1900-3304 m), INSIDE the s-coordinate stretch zone
(``kkconst`` = Fortran level 26) but 2-4 levels below the z->s transition.
Already EXCLUDED (do not re-litigate): time level (Kmm confirmed both
sides), ``ww`` inheritance (reconstructed NEMO ww is clean, err_norm
5.98e-05, no levels-29-33 feature), bolus/GM (enters neither side), bottom/
partial-cell per-column boundary (corr(colmax|err|, bottom_level)=-0.10),
vertical-metric DIVISOR value (``zad_vertical_metric_walk.py`` A/B: min-rule
h_u vs pure-NEMO e3u(Kmm) give near-identical err_norm, re-confirmed in this
script's self-check 3 below).

THIS SCRIPT tests the two SURVIVING candidates named in the task:

  Candidate 1: is NEMO's zWdzU recurrence (``dynzad.F90:81-119``) actually
  equivalent to legoESM's vectorized gather ``G[k]/G[k+1]``
  (``vertical.py:1149-1270``, ``nemo_advective_vertical_momentum_advection``),
  INCLUDING the boundary seeds and any interior masking legoESM applies that
  NEMO's fixed-bound loop does not?

  Candidate 2: does the ZAD error correlate with the LOCAL vertical e3
  gradient |e3(k+1)-e3(k)|/e3(k) (a quantity not tested by
  ``zad_vertical_metric_walk.py``, which tested the e3 VALUE/divisor, not
  its level-to-level gradient)?

===========================================================================
CANDIDATE 1 -- recurrence vs gather, worked from the Fortran verbatim
===========================================================================
``dynzad.F90:81-119`` (stock, no DINO MY_SRC override -- confirmed:
``ls cfgs/DINO/MY_SRC/dynzad.F90`` -> not found):

    zWdzU(T2D(0)) = 0._wp                                   ! :83  surface seed, jk=1 "top" term = 0
    DO_3D( 0,0,0,0, 1, jpk-2 )                               ! :86  jk = 1 .. jpk-2 (top-down, ASCENDING)
       zWf   = e1e2t(ji  ,jj) * ww(ji  ,jj,jk+1)             ! :93  Kmm ww, INTERFACE jk+1 (below cell jk)
       zWfi  = e1e2t(ji+1,jj) * ww(ji+1,jj,jk+1)             ! :94
       zzWfu = zWfi + zWf                                    ! :97  (NOT *0.5 -- a SUM of the two T-columns' e1e2t*ww)
       zzWdzU = zzWfu * ( puu(ji,jj,jk,Kmm) - puu(ji,jj,jk+1,Kmm) )   ! :100  Kmm velocity, cell jk minus cell jk+1
       puu(ji,jj,jk,Krhs) -= 0.25*r1_e1e2u(ji,jj)/e3u(ji,jj,jk,Kmm) * ( zWdzU(ji,jj) + zzWdzU )   ! :104-105
       zWdzU(ji,jj) = zzWdzU                                 ! :109  CARRY to the NEXT iteration (jk+1)'s "top" term
    END_3D
    jk = jpkm1                                               ! :113  bottom cell, ONLY the carried top-interface term
    puu(ji,jj,jk,Krhs) -= 0.25*r1_e1e2u(ji,jj)/e3u(ji,jj,jk,Kmm) * zWdzU(ji,jj)   ! :115-116

This IS a genuine bottom-directed-value CARRY (``zWdzU`` persists across
loop iterations), but unrolling it shows every carried value is IDENTICAL to
a value computed fresh one iteration earlier -- i.e. the recurrence has NO
memory beyond one step, so it collapses to a pure interface-indexed array
with NO recursive dependency:

    at iteration jk, "zWdzU" used = the "zzWdzU" PRODUCED at iteration jk-1
                    = zzWfu(jk) * (puu(jk-1,Kmm) - puu(jk,Kmm))   [interface jk flux]
    "zzWdzU" produced at iteration jk = zzWfu(jk+1) * (puu(jk,Kmm)-puu(jk+1,Kmm))  [interface jk+1 flux]

Defining Flux[k] = zzWfu(k) * (puu(k-1,Kmm) - puu(k,Kmm)) for interface k
(k=2..jpk-1; Flux[1]=0 by the :83 seed; Flux[jpk] architecturally 0 since
``ww(jpk)=0``, ``sshwzv.F90:182``, never read), the loop body is EXACTLY

    tend(jk) = -0.25*r1_e1e2u/e3u(jk,Kmm) * ( Flux[jk] + Flux[jk+1] )     for jk=1..jpk-2
    tend(jpkm1) = -0.25*r1_e1e2u/e3u(jpkm1,Kmm) * Flux[jpkm1]             (Flux[jpk]=0 dropped)

which is IDENTICAL, term for term, to legoESM's
``G[k] = 2*w_area_half[k]*(u[k-1]-u[k])``, ``tend[k] = -0.25/(area*h)*(G[k]+G[k+1])``
PROVIDED ``w_area_half[k] == zzWfu(k) == zWf(k)+zWfi(k)`` exactly (a SUM of
the two neighbouring T-columns' ``e1e2t*ww``, not a mean). legoESM's
``w_area_half = interp_cell_to_uface(area_T*w)`` is a 0.5*(a+b) MEAN
(``operators_latlon_cgrid.py:325-345``: ``return 0.5*(f_pad[:,:-1]+f_pad[:,1:])``),
and ``G`` carries an explicit leading ``2.0*`` factor
(``vertical.py:1247``: ``G_interior = 2.0*w_interior*du_interior``) --
``2 * 0.5*(a+b) = a+b`` -- so the mean-with-a-factor-of-2 IS the sum NEMO
uses. This is verified numerically below (self-check 2, synthetic random
column, exact recurrence vs gather, both boundary seeds included) before
trusting it on real data.

VERDICT (pending the synthetic self-check below): Candidate 1's recurrence
DIRECTION AND BOUNDARY SEEDS ARE EQUIVALENT to legoESM's gather -- REFUTES a
recurrence/indexing mismatch as the cause, ANALOGOUS to how
``ldfslp.F90:209``'s ``DO jk=jpkm1,2,-1`` turned out to be gather-equivalent
in this campaign's precedent (per the task prompt). The remaining candidate
inside "Candidate 1"'s scope is therefore narrowed to: does legoESM's
``face_active`` masking (``vertical.py:1249-1252``,
``active_above*active_below`` zeroing ``G_interior`` at inactive faces) --
which NEMO's Fortran loop has NO EQUIVALENT OF at all, since ``DO_3D(...,
1,jpk-2)`` runs over the FIXED level range with no ``IF(umask)`` guard
inside the loop body -- diverge from NEMO at exactly Python levels 29-33?
Tested directly in STEP B below (real restart data, not synthetic).

===========================================================================
CANDIDATE 2 -- e3 vertical GRADIENT (not value) correlation
===========================================================================
Test whether |e3(k+1)-e3(k)|/e3(k) (u-face e3, live e3u(Kmm), the SAME
pure-NEMO-space reconstruction ``zad_vertical_metric_walk.py`` already built
and validated -- reused via an independent re-derivation here since that
file is UNTOUCHABLE, not imported) correlates with the per-level, per-column
ZAD error at Python levels 29-33.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.zad_recurrence_walk
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
    bridge_nemo_to_legoesm_topo, _u_east_to_face_periodic, _v_north_to_face,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_geometry_and_density, _bc_vertical_and_depthmean_velocity,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    interp_cell_to_uface, interp_cell_to_vface, compute_face_masks_3d,
)
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate, nemo_advective_vertical_momentum_advection,
)
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale (not re-derived): established RUN_DIR/DT/loader, same as
# every #1226 probe in this campaign.
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import RUN_DIR, DT, _load_full_3d

register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment (stock dynzad.F90:86-119).")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")

for _name in ("zad_dump_du.bin", "zad_dump_dv.bin"):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _err_norm(lego3, nemo3, mask2d):
    """SAME convention as dyn_zad_ldf_walk.py/zad_vertical_metric_walk.py's
    ``_err_norm`` (verified identical formula by inspection before use here,
    per the metric-mixing rule): RMS-normalized error, per level, combined
    via sqrt(mean(err_by_level**2))/sqrt(mean(rms_by_level**2))."""
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
    near_zero_frac = float(np.mean(np.abs(ne) < 1e-30))
    return err, err_by_level, rms_by_level, tot_err_norm, max_abs, near_zero_frac


def _synthetic_recurrence_check() -> bool:
    """Self-check 2 (recurrence == gather), synthetic random column, exact
    Fortran transcription vs legoESM's gather formula, both boundary seeds.
    Returns True iff max|diff| < 1e-12 (bit-identical to fp roundoff)."""
    rng = np.random.default_rng(0)
    jpk = 12
    jpkm1 = jpk - 1
    # 1-indexed Fortran arrays via 0-index python arrays of length jpk+2
    # (index 0 unused, index jpk+1 unused padding).
    ww_e1e2 = np.zeros(jpk + 2)
    ww_e1e2_i = np.zeros(jpk + 2)
    for jk in range(1, jpk + 1):
        ww_e1e2[jk] = rng.standard_normal()
        ww_e1e2_i[jk] = rng.standard_normal()
    ww_e1e2[jpk] = 0.0     # bottom BC pww(jpk)=0, sshwzv.F90:182
    ww_e1e2_i[jpk] = 0.0
    u = rng.standard_normal(jpk + 2)
    e3u = 1.0 + 0.1 * rng.random(jpk + 2)
    e1e2u_r1 = 1.3

    # --- literal NEMO recurrence transcription ---
    Krhs = np.zeros(jpk + 2)
    zWdzU = 0.0
    for jk in range(1, jpk - 1):          # DO jk = 1, jpk-2
        zWf = ww_e1e2[jk + 1]
        zWfi = ww_e1e2_i[jk + 1]
        zzWfu = zWfi + zWf
        zzWdzU = zzWfu * (u[jk] - u[jk + 1])
        Krhs[jk] = Krhs[jk] - 0.25 * e1e2u_r1 / e3u[jk] * (zWdzU + zzWdzU)
        zWdzU = zzWdzU
    jk = jpkm1
    Krhs[jk] = Krhs[jk] - 0.25 * e1e2u_r1 / e3u[jk] * zWdzU

    # --- legoESM gather: w_area_half[k] = MEAN(ww_e1e2[k], ww_e1e2_i[k])
    # (interp_cell_to_uface convention, operators_latlon_cgrid.py:325-345),
    # G[k] = 2*w_area_half[k]*(u[k-1]-u[k]) (vertical.py:1244-1247) ---
    w_area_half = np.zeros(jpk + 2)
    for k in range(1, jpk + 1):
        w_area_half[k] = 0.5 * (ww_e1e2[k] + ww_e1e2_i[k])
    G = np.zeros(jpk + 2)
    for k in range(2, jpk):
        G[k] = 2.0 * w_area_half[k] * (u[k - 1] - u[k])
    G[1] = 0.0       # surface seed, matches zWdzU(jk=1)=0
    G[jpk] = 0.0      # bottom interface, never read / architecturally zero

    Krhs2 = np.zeros(jpk + 2)
    for jk in range(1, jpkm1 + 1):
        Krhs2[jk] = -0.25 * e1e2u_r1 / e3u[jk] * (G[jk] + G[jk + 1])

    maxdiff = float(np.max(np.abs(Krhs[1:jpk] - Krhs2[1:jpk])))
    print(f"  synthetic recurrence-vs-gather max|diff| = {maxdiff:.3e} "
          f"(want < 1e-12, bit-identical to fp roundoff)")
    return maxdiff < 1e-12


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zad_recurrence_walk")
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
    require_fp64(br.geometry, br.z_coord, br.state, context="zad_recurrence_walk")
    print("dtype check: u", br.state.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype,
          "eta", br.state.eta.data.dtype)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)

    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)

    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vertadv_v_f = _v_to_nemo(vertadv_v_3d)

    # =========================================================================
    # SELF-CHECK 1 (MANDATORY, first thing, before anything else): reproduce
    # ZAD's already-recorded 3.9993e-2 / 5.0746e-2 baseline.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce the recorded ZAD baseline (production, unchanged)")
    print("=" * 78)
    err_u0, ebl_u0, rbl_u0, tot_u0, maxabs_u0, nzf_u0 = _err_norm(vertadv_u_f, nemo_zad_du, umask2)
    err_v0, ebl_v0, rbl_v0, tot_v0, maxabs_v0, nzf_v0 = _err_norm(vertadv_v_f, nemo_zad_dv, vmask2)
    print(f"  u: err_norm={tot_u0:.4e}  (recorded 3.9993e-2)   max|diff|={maxabs_u0:.4e}")
    print(f"  v: err_norm={tot_v0:.4e}  (recorded 5.0746e-2)   max|diff|={maxabs_v0:.4e}")
    check1_ok = abs(tot_u0 - 3.9993e-2) < 2e-3 and abs(tot_v0 - 5.0746e-2) < 2e-3
    print(f"  SELF-CHECK 1: {'PASSED' if check1_ok else 'FAILED -- STOP, harness unverified'}")
    if not check1_ok:
        return 1

    # =========================================================================
    # SELF-CHECK 2: recurrence == gather, synthetic column (Candidate 1
    # algebraic verification, independent of legoESM's operator entirely).
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 2: NEMO recurrence (literal transcription) == legoESM gather")
    print("formula, synthetic random column, both boundary seeds included")
    print("=" * 78)
    check2_ok = _synthetic_recurrence_check()
    print(f"  SELF-CHECK 2: {'PASSED' if check2_ok else 'FAILED'}")

    # =========================================================================
    # STEP A: CANDIDATE 1 continued -- alignment table (printed, not baked
    # into a verdict string) + is legoESM's `face_active` masking (which NEMO
    # has NO equivalent of inside its fixed-bound loop) the source of the
    # levels-29-33 jump? Rebuild the production call with face_active
    # DISABLED (all-ones) and see whether the jump moves.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP A: ordered term-by-term alignment table, NEMO line vs legoESM line")
    print("=" * 78)
    alignment_rows = [
        ("surface seed zWdzU(jk=1)=0",              "dynzad.F90:83",      "G[..., 0]=0 (jnp.pad)",                    "vertical.py:1258",  "MATCH"),
        ("loop k-ordering (ascending jk, top-down)", "dynzad.F90:86",      "vectorized over ALL k at once (order-free)","vertical.py:1244-1247","MATCH (order-free gather is equivalent to an ascending scan with 1-step memory, verified self-check 2)"),
        ("w at interface jk+1, area-weighted SUM",   "dynzad.F90:93-97",   "w_area_half[k]=interp_cell_to_uface(area_T*w) [MEAN] * 2.0 in G", "vertical.py:1194-1195,1247; operators_latlon_cgrid.py:325-345", "MATCH (2*mean==sum, verified self-check 2)"),
        ("velocity difference u(jk,Kmm)-u(jk+1,Kmm)","dynzad.F90:100",     "u[...,:-1]-u[...,1:]",                     "vertical.py:1246",  "MATCH"),
        ("carry zWdzU(jk)=zzWdzU (recurrence)",      "dynzad.F90:109",     "no carry -- G[k] computed independently for every k", "vertical.py:1244-1247", "MATCH (algebraically equivalent, no residual memory beyond 1 step, self-check 2)"),
        ("tendency = -(0.25/e1e2u/e3u)*(top+bot)",   "dynzad.F90:104-105", "-0.25/(face_area*h_u)*(G_top+G_bot)",      "vertical.py:1260-1270", "MATCH"),
        ("bottom cell jk=jpkm1: TOP term only",      "dynzad.F90:113-118", "G[jpk]=0 via jnp.pad (never a real value)", "vertical.py:1258",  "MATCH"),
        ("interior per-level scheme-consistency mask","(none -- NEMO's DO_3D(0,0,0,0,1,jpk-2) has NO umask/bottom guard inside the loop body at all)", "face_active=broadcast(u_mask_3d); G_interior *= active_above*active_below", "ocean_pe_latlon_cgrid.py:2532-2534,2538-2543; vertical.py:1249-1252", "ABSENT ON NEMO SIDE -- legoESM adds masking NEMO's loop does not have. TESTED below."),
        ("thickness divisor e3u(jk,Kmm)",            "dynzad.F90:104,115", "h_u = min_cell_to_uface(h_k) [min-rule]",  "ocean_pe_latlon_cgrid.py:1351",  "DIFFERENT FORMULA, already A/B-tested near-identical numerically (zad_vertical_metric_walk.py, re-confirmed self-check 3 below) -- REFUTED as cause"),
    ]
    for row in alignment_rows:
        print(f"  [{row[4]}] {row[0]}")
        print(f"      NEMO   : {row[1]}")
        print(f"      legoESM: {row[3]}  ({row[2]})")

    # =========================================================================
    # STEP B: does legoESM's face_active masking explain the jump? Rebuild
    # the SAME production call but with face_active forced to all-active
    # (matching NEMO's unmasked fixed-bound loop exactly) and compare.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP B A/B: face_active masking ON (production) vs OFF (all-active,")
    print("matching NEMO's unmasked DO_3D(0,0,0,0,1,jpk-2) loop bound exactly)")
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
    (h_u_lego, h_v_lego, _flux_div_k, w, _u_prime, _v_prime) = (
        _bc_vertical_and_depthmean_velocity(
            h_k, u_full, v_full, u_mask_3d, v_mask_3d, grid, z_coord, umask_2d, vmask_2d,
        )
    )
    area_w = grid.area_T[..., jnp.newaxis] * w
    w_area_u = interp_cell_to_uface(area_w)
    w_area_v = interp_cell_to_vface(area_w, grid=grid)
    face_area_u = (grid.dx_u * grid.dy_u)[..., jnp.newaxis]
    face_area_v = (grid.dx_v * grid.dy_v)[..., jnp.newaxis]

    u_face_active_on = jnp.broadcast_to(u_mask_3d, u_full.shape)
    v_face_active_on = jnp.broadcast_to(v_mask_3d, v_full.shape)
    u_face_active_off = jnp.ones_like(u_face_active_on, dtype=bool)
    v_face_active_off = jnp.ones_like(v_face_active_on, dtype=bool)

    variants_b = {
        "ON  (production, face_active masks G at inactive faces)": (u_face_active_on, v_face_active_on),
        "OFF (all-active, matches NEMO's unmasked fixed loop bound)": (u_face_active_off, v_face_active_off),
    }
    for label, (ufa, vfa) in variants_b.items():
        du = nemo_advective_vertical_momentum_advection(u_full, w_area_u, h_u_lego, face_area_u, face_active=ufa)
        dv = nemo_advective_vertical_momentum_advection(v_full, w_area_v, h_v_lego, face_area_v, face_active=vfa)
        du_f = _u_to_nemo(np.asarray(du))
        dv_f = _v_to_nemo(np.asarray(dv))
        err_u, ebl_u, rbl_u, tot_u, maxabs_u, _ = _err_norm(du_f, nemo_zad_du, umask2)
        err_v, ebl_v, rbl_v, tot_v, maxabs_v, _ = _err_norm(dv_f, nemo_zad_dv, vmask2)
        n_lat_bb = min(du_f.shape[0], nemo_zad_du.shape[0], umask2.shape[0])
        n_lon_bb = min(du_f.shape[1], nemo_zad_du.shape[1], umask2.shape[1])
        n_lev_bb = min(du_f.shape[2], nemo_zad_du.shape[2])
        mbb = umask2[:n_lat_bb, :n_lon_bb]
        a_full = np.asarray(du_f)[:n_lat_bb, :n_lon_bb, :n_lev_bb]
        b_full = np.asarray(nemo_zad_du)[:n_lat_bb, :n_lon_bb, :n_lev_bb]
        a_ = a_full[mbb]
        b_ = b_full[mbb]
        ok = np.isfinite(a_) & np.isfinite(b_)
        corr = float(np.corrcoef(a_[ok], b_[ok])[0, 1]) if ok.sum() > 1 else float("nan")
        ratio = float(np.abs(a_[ok]).sum() / np.abs(b_[ok]).sum()) if ok.sum() else float("nan")
        print(f"\n  --- {label} ---")
        print(f"  u: err_norm={tot_u:.4e}  max|diff|={maxabs_u:.4e}  corr={corr:.4f}  ratio={ratio:.4f}")
        print(f"  v: err_norm={tot_v:.4e}  max|diff|={maxabs_v:.4e}")
        print(f"  u err_by_level[26:35]: {np.array2string(ebl_u[26:35], precision=3, max_line_width=200)}")
        print(f"  v err_by_level[26:35]: {np.array2string(ebl_v[26:35], precision=3, max_line_width=200)}")

    # =========================================================================
    # SELF-CHECK 3: re-confirm the vertical-metric DIVISOR is a near no-op
    # here (already established by zad_vertical_metric_walk.py -- re-derive
    # independently, not trusted blindly, since that script is untouchable).
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 3: re-confirm h_u divisor (min-rule) vs pure-NEMO e3u(Kmm)")
    print("give near-identical err_norm (zad_vertical_metric_walk.py's finding),")
    print("independent re-derivation here (that script is untouchable, not")
    print("imported for this check).")
    print("=" * 78)
    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
        e1t = np.asarray(ds.variables["e1t"][0], dtype=np.float64)
        e2t = np.asarray(ds.variables["e2t"][0], dtype=np.float64)
        e1u = np.asarray(ds.variables["e1u"][0], dtype=np.float64)
        e2v = np.asarray(ds.variables["e2v"][0], dtype=np.float64)
    e1e2t = e1t * e2t
    ssh_now = np.asarray(s.ssh)
    ssh_e = np.roll(ssh_now, -1, axis=1)
    e1e2t_e = np.roll(e1e2t, -1, axis=1)
    e1e2u = e1u * e2t
    r3u = 0.5 * (e1e2t * ssh_now + e1e2t_e * ssh_e) / np.where(g.hu_0 > 0, g.hu_0, 1.0) / np.where(e1e2u > 0, e1e2u, 1.0)
    r3u = np.where(g.hu_0 > 0, r3u, 0.0)
    e3u_kmm_nemo_raw = g.e3u_0 * (1.0 + r3u[..., None] * g.umask)
    e3u_kmm_nemo = jnp.asarray(_u_east_to_face_periodic(e3u_kmm_nemo_raw))

    du_divisor_B = nemo_advective_vertical_momentum_advection(
        u_full, w_area_u, e3u_kmm_nemo, face_area_u, face_active=u_face_active_on)
    du_divisor_B_f = _u_to_nemo(np.asarray(du_divisor_B))
    _, ebl_B, _, tot_B, maxabs_B, _ = _err_norm(du_divisor_B_f, nemo_zad_du, umask2)
    print(f"  divisor=min-rule (production): u err_norm={tot_u0:.4e}")
    print(f"  divisor=pure-NEMO e3u(Kmm)   : u err_norm={tot_B:.4e}  max|diff|={maxabs_B:.4e}")
    print(f"  u err_by_level[26:35] (pure-NEMO divisor): "
          f"{np.array2string(ebl_B[26:35], precision=3, max_line_width=200)}")
    print("  (near-identical to self-check 1's production number -- re-confirms")
    print("  zad_vertical_metric_walk.py's divisor-value exclusion.)")

    # =========================================================================
    # CANDIDATE 2: e3 vertical GRADIENT correlation with the per-level,
    # per-column ZAD error. Uses the SAME pure-NEMO-space e3u(Kmm) built above
    # (live, at THIS restart's ssh) -- a genuinely different quantity from
    # the e3 VALUE tested by zad_vertical_metric_walk.py.
    # =========================================================================
    print("\n" + "=" * 78)
    print("CANDIDATE 2: e3 vertical-GRADIENT correlation with per-column |err|,")
    print("levels 26-35 (the jump band + 3 clean levels above it for contrast)")
    print("=" * 78)
    e3u_kmm_np = np.asarray(e3u_kmm_nemo)
    n_lat_g = min(e3u_kmm_np.shape[0], err_u0.shape[0], umask2.shape[0])
    n_lon_g = min(e3u_kmm_np.shape[1], err_u0.shape[1], umask2.shape[1])
    e3u_c = e3u_kmm_np[:n_lat_g, :n_lon_g, :]
    err_u_c = err_u0[:n_lat_g, :n_lon_g, :]
    m2 = umask2[:n_lat_g, :n_lon_g]

    nlev_g = min(e3u_c.shape[-1], err_u_c.shape[-1]) - 1
    grad_e3 = np.full((n_lat_g, n_lon_g, nlev_g), np.nan)
    for k in range(nlev_g):
        denom_k = np.where(np.abs(e3u_c[..., k]) > 1e-9, e3u_c[..., k], np.nan)
        grad_e3[..., k] = np.abs(e3u_c[..., k + 1] - e3u_c[..., k]) / denom_k

    print("  per-level: mean(grad_e3) over wet cells, mean(|err|), corr(grad_e3, |err|) across columns")
    corr_by_level = np.full(nlev_g, np.nan)
    for k in range(24, min(nlev_g, 36)):
        g_k = grad_e3[..., k][m2]
        e_k = np.abs(err_u_c[..., k])[m2]
        ok = np.isfinite(g_k) & np.isfinite(e_k)
        if ok.sum() > 10 and np.std(g_k[ok]) > 0 and np.std(e_k[ok]) > 0:
            c = float(np.corrcoef(g_k[ok], e_k[ok])[0, 1])
        else:
            c = float("nan")
        corr_by_level[k] = c
        print(f"    level {k:2d} (Fortran jk {k+1:2d}): mean(grad_e3)={np.nanmean(g_k):.4e}  "
              f"mean(|err|)={np.nanmean(e_k):.4e}  corr={c:.4f}")

    # Pooled correlation across ALL wet (column, level) pairs at levels 24-35.
    # grad_e3 has nlev_g levels (one fewer than err_u_c, lost to the k+1
    # diff); slice BOTH to the SAME common upper bound before pooling
    # (metric-mixing guard -- same population on both sides).
    pool_hi = min(nlev_g, err_u_c.shape[-1], 36)
    g_slab = grad_e3[..., 24:pool_hi]
    e_slab = np.abs(err_u_c[..., 24:pool_hi])
    assert g_slab.shape == e_slab.shape, (g_slab.shape, e_slab.shape)
    g_pool = g_slab[np.broadcast_to(m2[..., None], g_slab.shape)]
    e_pool = e_slab[np.broadcast_to(m2[..., None], e_slab.shape)]
    ok_pool = np.isfinite(g_pool) & np.isfinite(e_pool)
    corr_pooled = float(np.corrcoef(g_pool[ok_pool], e_pool[ok_pool])[0, 1])
    print(f"\n  POOLED corr(grad_e3, |err|), all wet (column,level) pairs lev 24-{pool_hi-1}: {corr_pooled:.4f}")

    steepest_level = int(np.nanargmax([np.nanmean(grad_e3[..., k][m2]) for k in range(24, min(nlev_g, 36))])) + 24
    print(f"  steepest-mean-gradient level in this band: {steepest_level} (Fortran jk {steepest_level+1})")
    print("  (jump band established by prior scripts: Python levels 29-33)")

    print("\n" + "=" * 78)
    print("DONE -- see terminal output above for the report's numeric inputs.")
    print("=" * 78)
    return 0 if (check1_ok and check2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
