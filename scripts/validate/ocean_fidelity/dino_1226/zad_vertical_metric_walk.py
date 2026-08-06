"""#1226 dyn_adv ZAD -- vertical-METRIC A/B walk (READ-ONLY measurement).

TARGET: ``dyn_adv ZAD`` (NEMO ``dynzad.F90``), the campaign's highest-leverage
open row (also feeds ``zv_frc``/barotropic v). Everything EXCLUDED already
(time level, bottom/partial-cell, ``ww`` inheritance, bolus) is documented in
``dyn_zad_ldf_walk.py``/``ww_inheritance_walk.py`` and NOT re-chased here.

THE LEAD (from the task): the ZAD error is near-zero through Python level 28
then jumps 1000x+ at Python levels 29-33 (Fortran jk 30-34, ~1900-3304 m) --
INSIDE the s-coordinate stretch zone (``kkconst`` = Fortran level 26,
``gdepw_1d[25]=982.4 m``) but 2-4 levels BELOW the z->s transition, while
``ww`` itself stays clean (err_norm 5.98e-05, no levels-29-33 feature). This
script tests whether the VERTICAL-METRIC (thickness) divisor inside
``dyn_zad`` -- not ``ww`` -- generates the jump.

===========================================================================
STEP 1: dynzad.F90 line-by-line, thickness array + time level at EVERY
division point (task requirement).
===========================================================================

Stock ``src/OCE/DYN/dynzad.F90`` (DINO has NO MY_SRC override -- checked:
``cfgs/DINO/MY_SRC/`` contains no ``dynzad.F90``). Only TWO division points
exist in the whole routine, both by the SAME array:

    dynzad.F90:104  puu(...,Krhs) -= 0.25 * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm) * (...)   [interior, jk=1..jpk-2]
    dynzad.F90:106  pvv(...,Krhs) -= 0.25 * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm) * (...)
    dynzad.F90:115  puu(...,Krhs) -= 0.25 * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm) * zWdzU  [bottom, jk=jpkm1]
    dynzad.F90:117  pvv(...,Krhs) -= 0.25 * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm) * zWdzV

``e3u``/``e3v`` are ``domzgr_substitute.h90`` macros. DINO's actual build
uses ``key_qco key_vco_3d`` (confirmed from the PREPROCESSED
``cfgs/DINO/BLD/ppsrc/nemo/dynzad.f90``, not just the cpp key list -- the
expanded macro is quoted verbatim below), so:

    domzgr_substitute.h90:127  e3u(i,j,k,t) = E3u_0(i,j,k) * (1 + r3u(i,j,t)*umask(i,j,k))   [key_qco Tmsk macro]
    domzgr_substitute.h90:95   E3u_0(i,j,k) = e3u_3d(i,j,k)                                   [key_vco_3d]

confirmed by the preprocessed source (``BLD/ppsrc/nemo/dynzad.f90:104``):

    puu(...,Krhs) -= 0.25 * r1_e1e2u(ji,jj) / (e3u_3d(ji,jj,jk) * (1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk))) * (...)

So the divisor is **LIVE**: a STATIC per-level 3-D reference field
``e3u_3d(i,j,k)`` (set ONCE at ``usrdef_zgr``/``domzgr`` init, never
recomputed per step) multiplied by a per-COLUMN (not per-level) LIVE stretch
factor ``(1 + r3u(i,j,Kmm))`` evaluated at Kmm ("now").

===========================================================================
STEP 1b: how ``e3u_3d`` (static) and ``r3u`` (live) are ACTUALLY built --
this is where the divergence from legoESM lives.
===========================================================================

RUN_GDB namelist (``cfgs/DINO/RUN_GDB/namelist_cfg:70-72``):
``ln_zco_nam=.true., ln_zps_nam=.false., ln_sco_nam=.false.`` -- RUN_GDB is
PURE Z-COORDINATE (``ld_zco`` branch), NOT an s-coordinate run. But DINO's
``ld_zco`` branch (``cfgs/DINO/MY_SRC/usrdef_zgr.F90:108-118``) still calls
the SAME ``zgr_sco_mi96`` tanh-stretch builder used by the s-coordinate
branch, passing a UNIFORM flat bathymetry (``zflat(:,:) = zHmax``,
line 111). The tanh stretch (``mi96_1d``, ``kkconst`` transition at Fortran
level 26) is therefore baked into the 1-D REFERENCE ladder (``pe3t_1d``) at
EVERY column identically -- there is no horizontal bathymetry variation
feeding the z-ladder itself in RUN_GDB.

``zgr_sco_mi96`` builds ``pe3u``/``pe3v`` from ``pe3t`` via
``e3tw_to_other_e3`` (``cfgs/DINO/MY_SRC/zgr_lib.F90:230-234``, comment at
:213-215 "s-coordinate: scale factors are simple t-level (w-level) averaging
from e3t (e3w) neighbours"):

    zgr_lib.F90:231   pe3u(ji,jj,jk) = 0.50 * ( pe3t(ji,jj,jk) + pe3t(ji+1,jj,jk) )

Since ``pe3t`` is horizontally UNIFORM in RUN_GDB (flat ``zHmax`` bathy for
the ladder), ``pe3u == pe3t`` to roundoff -- matching the walk script's
recorded ``e3t_0==e3u_0==e3v_0==e3f_0`` finding. So the STATIC reference
``e3u_3d`` is NOT where legoESM's ZAD divisor diverges from NEMO's -- both
sides see the same per-level uniform static ladder there.

The divergence is in the LIVE stretch ``r3u(Kmm)``. NEMO
(``domqco.F90:159-167``, ``dom_qco_r3c``, stock):

    domqco.F90:160  pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)                        ! T-point: LOCAL column ratio
    domqco.F90:166  pr3u(ji,jj) = 0.5 * ( e1e2t(ji,jj)*pssh(ji,jj) + e1e2t(ji+1,jj)*pssh(ji+1,jj) )
                                  * r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)                   ! U-point: AREA-WEIGHTED ssh MEAN / hu_0

``hu_0`` (``domain.F90:140,145``, stock): ``hu_0(i,j) = sum_k( e3u_0(i,j,k) *
umask(i,j,k) )`` -- ONE column-integrated static depth, built from the
ALREADY-MEANED ``e3u_0``. ``r3u`` is therefore ONE SCALAR PER COLUMN (not
per level k), applied identically to every wet level of that u-column via
the ``(1+r3u)`` factor on the per-level STATIC ``e3u_3d(i,j,k)``.

legoESM's chain (``ocean_pe_latlon_cgrid.py:1351/2462``,
``vertical.py:720-774``, ``latlon_cgrid_operators.py:277-306``):

    h_k(i,j,k)   = h_partial(i,j,k) * (eta(i,j)+H_bathy(i,j)) / H_bathy(i,j)   [T-point, PER LEVEL, full-step OceanPartialCellCoordinate]
    h_u(i,j,k)   = min( h_k(i,j-west,k), h_k(i,j,k) )                          [[min_cell_to_uface]] -- MIN-RULE, PER LEVEL

Two structurally different formulas at the u-face:
  - NEMO: MEAN the ssh of the two T-neighbours (weighted by cell area), then
    divide by the U-POINT'S OWN column-total static depth ``hu_0`` (itself
    built from mean-of-neighbour ``e3u_0``, summed over the U-FACE'S OWN
    ``umask``-active levels) -- ONE ratio broadcast to every k.
  - legoESM: take the per-level, per-COLUMN Jacobian-scaled T-thickness at
    EACH neighbour (using THAT column's own H_bathy in the ratio), THEN take
    the MIN of the two, independently AT EVERY LEVEL k.

These are bit-identical ONLY when ``H_bathy`` (and hence the per-column
Jacobian ratio) is the SAME on both sides of every face -- i.e. a perfectly
flat-bottom domain. DINO's ``nn_botcase=1`` bowl-shaped bathymetry
(``usrdef_zgr.F90:225-360``) means ``H_bathy`` genuinely varies row-to-row
and column-to-column; at a staircase step in the bowl (adjacent T-columns
with DIFFERENT ``bottom_level``/H_bathy), the MIN-rule and the
column-averaged-ratio-times-static-mean formula predict different u-face
thicknesses, growing sharply wherever the STEP is large -- which is exactly
what a bowl-shaped bathymetry produces at mid-depth (not at the surface,
where the column is uniformly wet, and not right at the seafloor of any
SINGLE column, matching the already-measured "peak sits 2-4 levels above the
column's own bottom_level, corr=-0.10" exclusion).

===========================================================================
STEP 2: legoESM caller alignment table (:2523-2548) -- see the printed table
in main() for the per-row alignment; short version: G/w-side transcription
(algebra + w area-weighting) MATCHES NEMO exactly (already established, ROW 2
exclusion); the ONLY term this script varies is the ``h_u``/``h_v`` divisor.
===========================================================================

===========================================================================
STEP 3/4: A/B the divisor (this script). Self-checks: (1) reproduce ZAD's
recorded baseline err_norm 3.9993e-2/5.0746e-2 with the UNCHANGED production
path before touching anything; (2) reproduce the recorded near-zero levels
0-28 / 1000x-jump levels 29-33 profile. Then swap ONLY the ``h_u``/``h_v``
argument fed to ``nemo_advective_vertical_momentum_advection`` between (A)
legoESM's production ``min_cell_to_uface(h_k)`` and (B) a pure-NEMO-space
reconstruction of ``e3u(Kmm)``/``e3v(Kmm)`` built from THIS restart's own
``ssh``/``e3u_0``/``hu_0`` (all already loaded by ``nemo_io.read_nemo_mesh_
mask`` -- reused, not re-derived, per the #1226 item-2 comment already in
that reader). Report per-level err_norm for both variants + the magnitude
arithmetic predicting the jump from ``(h_u_minrule - e3u_kmm_nemo) /
e3u_kmm_nemo`` at levels 29-34.

MEASUREMENT ONLY. Does not touch ``fidelity_bar_gate.py`` or any production
module. New script per task instructions.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zad_vertical_metric_walk.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface, min_cell_to_vface, interp_cell_to_uface, interp_cell_to_vface,
    compute_face_masks_3d,
)
from legoesm.ocean.vertical import OceanPartialCellCoordinate
from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale (not re-derived): established loaders + RUN_DIR/DT/helpers.
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
    """Same convention as dyn_zad_ldf_walk.py's ``_err_norm`` (RMS-normalized,
    per-level then combined), reused for direct comparability with the
    recorded 3.9993e-2/5.0746e-2 baseline."""
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


def _corr_ratio(a, b, mask):
    a = np.asarray(a)[mask].ravel()
    b = np.asarray(b)[mask].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(a, b)[0, 1])
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    ratio = rms_a / rms_b if rms_b > 0 else float("nan")
    return corr, ratio


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zad_vertical_metric_walk")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both' for this walk)")
    assert e3t_mode == "both", "run with LEGOESM_NEMO_E3T=both (task rule)"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    print(f"restart path: {os.path.join(RUN_DIR, 'DINO_00057600_restart.nc')}  step kt=57601")

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="zad_vertical_metric_walk")
    print("dtype check: u", br.state.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype,
          "eta", br.state.eta.data.dtype)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)

    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)

    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vertadv_v_f = _v_to_nemo(vertadv_v_3d)

    # =====================================================================
    # SELF-CHECK 1: reproduce the recorded ZAD baseline BEFORE changing
    # anything (task rule: reproduce 3.9993e-2 / 5.0746e-2 first).
    # =====================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce the recorded ZAD baseline (production, unchanged)")
    print("=" * 78)
    err_u0, ebl_u0, rbl_u0, tot_u0, maxabs_u0, nzf_u0 = _err_norm(vertadv_u_f, nemo_zad_du, umask2)
    err_v0, ebl_v0, rbl_v0, tot_v0, maxabs_v0, nzf_v0 = _err_norm(vertadv_v_f, nemo_zad_dv, vmask2)
    print(f"  u: err_norm={tot_u0:.4e}  (recorded 3.9993e-2)   max|diff|={maxabs_u0:.4e}")
    print(f"  v: err_norm={tot_v0:.4e}  (recorded 5.0746e-2)   max|diff|={maxabs_v0:.4e}")
    assert abs(tot_u0 - 3.9993e-2) < 2e-3, f"u baseline mismatch: {tot_u0}"
    assert abs(tot_v0 - 5.0746e-2) < 2e-3, f"v baseline mismatch: {tot_v0}"
    print("  SELF-CHECK 1 PASSED: baseline reproduced.")

    print("\n" + "=" * 78)
    print("SELF-CHECK 2: reproduce the recorded per-level jump at Python lev 29-33")
    print("=" * 78)
    print(f"  u err_by_level[20:36]: {np.array2string(ebl_u0[20:36], precision=3, max_line_width=200)}")
    print(f"  v err_by_level[20:36]: {np.array2string(ebl_v0[20:36], precision=3, max_line_width=200)}")
    jump_ratio_u = float(np.mean(ebl_u0[29:34]) / (np.mean(ebl_u0[:28]) + 1e-30))
    jump_ratio_v = float(np.mean(ebl_v0[29:34]) / (np.mean(ebl_v0[:28]) + 1e-30))
    print(f"  jump ratio (mean err[29:34] / mean err[0:28]): u={jump_ratio_u:.1f}  v={jump_ratio_v:.1f}")
    assert jump_ratio_u > 100 and jump_ratio_v > 100, "expected >>1000x jump not reproduced"
    print("  SELF-CHECK 2 PASSED: levels-29-33 jump reproduced.")

    # =====================================================================
    # A/B: swap the h_u/h_v divisor.
    #   (A) legoESM production: min_cell_to_uface/vface(h_k)  [already computed inside diag]
    #   (B) pure-NEMO-space reconstruction: e3u(Kmm) = e3u_0 * (1 + r3u(Kmm))
    #       r3u = 0.5*(e1e2t_w*ssh_w + e1e2t_e*ssh_e) * r1_hu_0 * r1_e1e2u
    #       (domqco.F90:159-167, dom_qco_r3c, stock -- exact transcription)
    # Both variants reuse the SAME G (w-side, u_full) already validated as
    # matching NEMO exactly (ROW 2 exclusion) -- only h_u/h_v differs.
    # =====================================================================
    print("\n" + "=" * 78)
    print("A/B: h_u/h_v divisor -- legoESM min-rule vs NEMO e3u(Kmm)/e3v(Kmm)")
    print("=" * 78)

    if g.e3u_0 is None or g.hu_0 is None or g.e3v_0 is None or g.hv_0 is None:
        raise RuntimeError(
            "mesh_mask.nc is missing e3u_0/e3v_0 -- cannot build the NEMO-space "
            "e3u(Kmm)/e3v(Kmm) reconstruction (nemo_io.read_nemo_mesh_mask "
            "already derives hu_0/hv_0 from these when present)."
        )

    # Recompute the exact quantities the production call site builds, so the
    # ONLY thing that differs between variant A and B is h_u/h_v. Pull u_full/
    # v_full, w, mask, grid from the state/model the same way
    # _bc_vertical_momentum_advection does (ocean_pe_latlon_cgrid.py:2536-2548).
    grid = br.geometry
    z_coord = br.z_coord
    T = br.state.T.data
    S = br.state.S.data
    eta = br.state.eta.data
    H_bathy = br.state.H_bathy.data
    mask = br.state.land_mask.data
    u_full = br.state.u.data
    v_full = br.state.v.data

    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_geometry_and_density, _bc_vertical_and_depthmean_velocity,
    )
    min_water_col = float(mc.min_water_column_m)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask
    _J, h_k, _rho_prime, _p_prime = _bc_geometry_and_density(
        eta_safe, H_bathy, z_coord, mc, T, S, mask, grid, mc.rho_0, mc.g,
    )
    # Face masks: production (:3915-3919) uses compute_face_masks_3d for a
    # partial/full-step OceanPartialCellCoordinate (per-level face wetness,
    # handles differing bottom_level across a face) rather than the plain 2-D
    # broadcast -- match that exactly since z_coord here IS full-step.
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
    u_face_active = jnp.broadcast_to(u_mask_3d, u_full.shape)
    v_face_active = jnp.broadcast_to(v_mask_3d, v_full.shape)

    # --- Variant A: legoESM production divisor (min-rule on the live h_k). ---
    diag_zad_u_A = nemo_advective_vertical_momentum_advection(
        u_full, w_area_u, h_u_lego, face_area_u, face_active=u_face_active)
    diag_zad_v_A = nemo_advective_vertical_momentum_advection(
        v_full, w_area_v, h_v_lego, face_area_v, face_active=v_face_active)

    # --- Variant B: pure-NEMO-space e3u(Kmm)/e3v(Kmm) reconstruction. ---
    # domqco.F90:159-167 dom_qco_r3c, stock (no DINO override):
    #   pr3u(i,j) = 0.5*(e1e2t(i,j)*ssh(i,j) + e1e2t(i+1,j)*ssh(i+1,j)) * r1_hu_0(i,j) * r1_e1e2u(i,j)
    #   e3u(i,j,k,Kmm) = e3u_0(i,j,k) * (1 + r3u(i,j)*umask(i,j,k))
    # All built in RAW NEMO T-grid space (52-wide, no periodic wrap) using
    # ONLY NemoGrid fields (e1t/e2t/e1u/e3u_0/hu_0/umask -- all already loaded
    # by nemo_io.read_nemo_mesh_mask, none re-derived) + this restart's own
    # ssh (Kmm/"now", read_nemo_restart docstring), then wrapped to legoESM's
    # face convention with the SAME periodic u-face mapper the bridge uses
    # for the velocity itself (bridge_nemo_to_legoesm_topo -> _u_east_to_face
    # _periodic), so variant B's e3u sits at EXACTLY the same face index as
    # variant A's h_u_lego and as u_full.
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face_periodic, _v_north_to_face,
    )
    ssh_now = np.asarray(s.ssh)   # restart sshn = Kmm/"now" (read_nemo_restart docstring), RAW NEMO shape
    e1t = g.e1t
    e2t = g.e2t
    e1e2t = e1t * e2t
    # NEMO e1e2u = e1u*e2u; DINO regular lon-lat has e2u==e2t at the same row
    # (the SAME coincidence already documented+relied on in
    # nemo_advective_vertical_momentum_advection's own docstring) -- e2u is
    # not carried in NemoGrid, so use e2t (exact on this grid, not assumed:
    # e2 depends only on latitude, independent of the u/t distinction in
    # longitude).
    e1u = g.e1u
    e1e2u = e1u * e2t
    e1e2v = e1t * g.e2v

    ssh_e = np.roll(ssh_now, -1, axis=1)          # east neighbour (periodic i, matches DINO ln_Iperio)
    e1e2t_e = np.roll(e1e2t, -1, axis=1)
    r3u = 0.5 * (e1e2t * ssh_now + e1e2t_e * ssh_e) / np.where(g.hu_0 > 0, g.hu_0, 1.0) / np.where(e1e2u > 0, e1e2u, 1.0)
    r3u = np.where(g.hu_0 > 0, r3u, 0.0)

    ssh_n = np.pad(ssh_now, ((0, 1), (0, 0)), mode="edge")[1:, :]   # north neighbour (wall at pole rows; edge-pad is inert there since vmask=0)
    e1e2t_n = np.pad(e1e2t, ((0, 1), (0, 0)), mode="edge")[1:, :]
    r3v = 0.5 * (e1e2t * ssh_now + e1e2t_n * ssh_n) / np.where(g.hv_0 > 0, g.hv_0, 1.0) / np.where(e1e2v > 0, e1e2v, 1.0)
    r3v = np.where(g.hv_0 > 0, r3v, 0.0)

    e3u_kmm_nemo_raw = g.e3u_0 * (1.0 + r3u[..., None] * g.umask)   # RAW NEMO shape (n_lat, n_lon, nlev)
    e3v_kmm_nemo_raw = g.e3v_0 * (1.0 + r3v[..., None] * g.vmask)

    e3u_kmm_nemo = _u_east_to_face_periodic(e3u_kmm_nemo_raw)   # -> legoESM face shape, matches h_u_lego
    e3v_kmm_nemo = _v_north_to_face(e3v_kmm_nemo_raw)

    diag_zad_u_B = nemo_advective_vertical_momentum_advection(
        u_full, w_area_u, jnp.asarray(e3u_kmm_nemo), face_area_u, face_active=u_face_active)
    diag_zad_v_B = nemo_advective_vertical_momentum_advection(
        v_full, w_area_v, jnp.asarray(e3v_kmm_nemo), face_area_v, face_active=v_face_active)

    variants = {
        "A: legoESM min-rule h_u/h_v (production)": (diag_zad_u_A, diag_zad_v_A),
        "B: NEMO e3u(Kmm)/e3v(Kmm) reconstruction": (diag_zad_u_B, diag_zad_v_B),
    }
    for label, (du, dv) in variants.items():
        du_f = _u_to_nemo(np.asarray(du))
        dv_f = _v_to_nemo(np.asarray(dv))
        err_u, ebl_u, rbl_u, tot_u, maxabs_u, _ = _err_norm(du_f, nemo_zad_du, umask2)
        err_v, ebl_v, rbl_v, tot_v, maxabs_v, _ = _err_norm(dv_f, nemo_zad_dv, vmask2)
        n_lat_m = min(du_f.shape[0], nemo_zad_du.shape[0], umask2.shape[0])
        n_lon_m = min(du_f.shape[1], nemo_zad_du.shape[1], umask2.shape[1])
        n_lev_m = min(du_f.shape[2], nemo_zad_du.shape[2])
        mask3d_u = np.broadcast_to(
            umask2[:n_lat_m, :n_lon_m, None], (n_lat_m, n_lon_m, n_lev_m))
        corr_u, ratio_u = _corr_ratio(
            du_f[:n_lat_m, :n_lon_m, :n_lev_m],
            nemo_zad_du[:n_lat_m, :n_lon_m, :n_lev_m], mask3d_u)
        print(f"\n  --- {label} ---")
        print(f"  u: err_norm={tot_u:.4e}  max|diff|={maxabs_u:.4e}  corr={corr_u:.4f}  ratio={ratio_u:.4f}")
        print(f"  v: err_norm={tot_v:.4e}  max|diff|={maxabs_v:.4e}")
        print(f"  u err_by_level[20:36]: {np.array2string(ebl_u[20:36], precision=3, max_line_width=200)}")
        print(f"  v err_by_level[20:36]: {np.array2string(ebl_v[20:36], precision=3, max_line_width=200)}")
        jr_u = float(np.mean(ebl_u[29:34]) / (np.mean(ebl_u[:28]) + 1e-30))
        jr_v = float(np.mean(ebl_v[29:34]) / (np.mean(ebl_v[:28]) + 1e-30))
        print(f"  jump ratio (29:34 / 0:28): u={jr_u:.2f}  v={jr_v:.2f}")

    # =====================================================================
    # STEP 4: magnitude arithmetic -- does (h_u_minrule - e3u_kmm_nemo)/e3u_kmm_nemo
    # at levels 29-34 PREDICT the observed error jump?
    # =====================================================================
    print("\n" + "=" * 78)
    print("MAGNITUDE CHECK: relative divisor mismatch vs observed error jump, lev 29-34")
    print("=" * 78)
    h_u_lego_np = np.asarray(h_u_lego)
    e3u_kmm_nemo_np = np.asarray(e3u_kmm_nemo)
    n_lat_c = min(h_u_lego_np.shape[0], e3u_kmm_nemo_np.shape[0])
    n_lon_c = min(h_u_lego_np.shape[1], e3u_kmm_nemo_np.shape[1])
    h_u_c = h_u_lego_np[:n_lat_c, :n_lon_c, :]
    e3u_c = e3u_kmm_nemo_np[:n_lat_c, :n_lon_c, :]
    umask_face_2d = np.asarray(umask_2d)
    m_c = (umask_face_2d[:n_lat_c, :n_lon_c] > 0.5)
    for lev in range(26, 36):
        denom = np.where(np.abs(e3u_c[..., lev]) > 1e-9, e3u_c[..., lev], np.nan)
        rel = (h_u_c[..., lev] - e3u_c[..., lev]) / denom
        rel_wet = rel[m_c]
        rel_wet = rel_wet[np.isfinite(rel_wet)]
        if rel_wet.size:
            print(f"  Python level {lev:2d} (Fortran jk {lev+1:2d}): "
                  f"mean|rel divisor mismatch|={np.mean(np.abs(rel_wet)):.4e}  "
                  f"max|rel|={np.max(np.abs(rel_wet)):.4e}  "
                  f"frac(|rel|>1e-3)={np.mean(np.abs(rel_wet) > 1e-3):.4f}")
        else:
            print(f"  Python level {lev:2d}: no wet cells in overlap")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
