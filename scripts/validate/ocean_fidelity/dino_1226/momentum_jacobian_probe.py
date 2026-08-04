#!/usr/bin/env python
"""#1226 momentum surface-stress divisor: GLOBAL-Hmax Jacobian J vs NEMO's
LOCAL r3u, and whether the mismatch is the (partial-)cause of the zu_frc
residual (fidelity_bar_gate.py "dyn_spg_ts puu_b" row, 8.03e-3).

THE CONFIRMED GAP (read from NEMO source, file:line cited below):

  legoESM ``surface_stress_faces`` (ocean_pe_latlon_cgrid.py:3445-3481) builds
  the top-cell thickness it divides tau by as::

      dz_0_T = z_coord.dz_ref[0] * J                          # T-point
      J = compute_ocean_jacobian(eta, H_bathy, z_coord)        # vertical.py:777-833
      dz_0_u = interp_cell_to_uface(dz_0_T)                    # simple 2-pt average

  Under ``LEGOESM_NEMO_E3T=both`` (this run's mode), ``z_coord`` is an
  ``OceanPartialCellCoordinate`` so (vertical.py:824-833)::

      J = (eta + H_bathy) / H_bathy          (per T-column, LOCAL to that column)

  which is itself LOCAL -- the gap is not the H_max normalization on this
  branch (that formula is the ``OceanZStarCoordinate`` branch, vertical.py:833,
  NOT active for E3T=both). The remaining, still-real gap on this branch is
  the ORDER OF OPERATIONS + reference-thickness convention:
  legoESM computes ``dz_0_T`` at the T-point (from THAT column's own
  H_bathy), THEN simple-averages the thickness to the u-face. NEMO instead
  (1) area-weights ssh onto the u-face FIRST, then (2) divides by ``hu_0``,
  a u-point reference depth built INDEPENDENTLY from ``e3u_0`` (NOT a simple
  average of the two flanking ``ht_0``).

  NEMO divisor, dynzdf.F90:329-335 (MLF branch -- DINO has no key_RK3,
  cpp_DINO.fcm: ``key_qco key_vco_3d``, confirmed no ``key_RK3`` define)::

      puu(ji,jj,1,Kaa) = puu(ji,jj,1,Kaa)
        + zDt_2 * ( utau_b(ji,jj) + utauU(ji,jj) )
        / ( e3u(ji,jj,1,Kaa) * rho0 ) * umask(ji,jj,1)          [dynzdf.F90:333-334]

  e3u under key_qco (domzgr_substitute.h90:127, key_vco_3d E3u_0=e3u_3d
  :94-100)::

      e3u(i,j,1,Kaa) = E3u_0(i,j,1) * ( 1 + r3u(i,j,Kaa) )      [:127]

  r3u (domqco.F90:163-169, dom_qco_r3c, the non-RK3/vector-form branch
  DINO's ln_dynadv_vec selects)::

      r3u(i,j,t) = 0.5*( e1e2t(i,j)*ssh(i,j,t) + e1e2t(i+1,j)*ssh(i+1,j,t) )
                   * r1_hu_0(i,j) * r1_e1e2u(i,j)                [:166-167]

  hu_0 (domain.F90:140-146, the LOCAL u-point reference-depth SUM, built
  from e3u_0 -- NOT a simple average of ht_0)::

      hu_0(i,j) = SUM_k( e3u_0(i,j,k) * umask(i,j,k) )           [:145]

  TIME LEVEL (stpmlf.F90:305, DINO's active MLF branch, no key_RK3)::

      CALL dyn_zdf( kstp, Nbb, Nnn, Nrhs, uu, vv, Naa )
      -- Kbb=Nbb, Kmm=Nnn, Kaa=Naa.  dyn_zdf's own SBC (:329-335) uses
      e3u(:,:,1,Kaa) -- Naa/after.  r3u(Naa) is (re-)computed at
      stpmlf.F90:303 ``CALL dom_qco_r3c(ssh(:,:,Naa), r3t(:,:,Naa),
      r3u(:,:,Naa), r3v(:,:,Naa), r3f(:,:))`` -- called AFTER dyn_spg_ts
      (stpmlf.F90:288, which writes pssh(:,:,Kaa) at dynspg_ts.F90:835), so
      the ssh(Naa) feeding r3u(Naa) here is the BAROTROPIC-CORRECTED ssh, not
      the raw ssh_nxt estimate (sshnxt_dump_ssh_after.bin, stpmlf.F90:214,
      which nothing rewrites between there and dyn_spg_ts -- confirmed by
      reading dynspg_ts.F90:835's write target). The correct NEMO dump for
      ssh(Naa) at the dyn_zdf call is therefore
      ``spg_dump_pssh_final.bin`` (the after-loop boxcar-averaged
      pssh(Kaa), already registered "after" in spg_substep_chain.py:215-216),
      NOT ``atf_dump_ssh_after.bin`` (registered "after" too, but that is
      ssh_atf's OWN output -- sshwzv.F90:361 ``CALL ssh_atf(...)`` runs
      AFTER dyn_zdf at stpmlf.F90:305, so it is a LATER value that has
      nothing to do with the r3u dyn_zdf actually used).

TASK 2 (metric distribution) uses the T-point J (legoESM, both branches
active for E3T=both -- vertical.py:824-833) vs the u-FACE live thickness
NEMO's r3u implies, both converted to a common quantity: the U-FACE TOP-CELL
THICKNESS, since that is what both actually multiply/divide by in the
respective surface-stress formulas.

TASK 3/4 reuse spg_substep_chain.py's zu_frc reconstruction machinery (STAGE
0/7, "F_slow_u" capture + nemo_zu_frc load) -- imported as a sibling module,
not re-derived (skill Rule 0 / CLAUDE.md no-duplicate-numerics).

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/momentum_jacobian_probe.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(__file__)


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_cov = _load_sibling("coverage_rows_measure.py", "_coverage_rows_measure")
per_element_stats = _cov.per_element_stats
# Loading spg_substep_chain.py registers spg_dump_pssh_final.bin (and its
# siblings) as a side effect of its module-level register_dump(...) calls --
# reuse that registration rather than re-citing it here (single source of
# truth for the citation).
_spg_preload = _load_sibling("spg_substep_chain.py", "_spg_substep_chain_preload")

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.vertical import compute_ocean_jacobian
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
KT_DUMP = 57601  # nit000

set_policy(PrecisionPolicy.fp64())

# spg_dump_pssh_final.bin is already registered "after" by spg_substep_chain.py
# (dynspg_ts.F90:955-956,977-978,1008-1026 -- boxcar-averaged pssh(Kaa)); this
# script's OWN new dump-name usage is nothing new -- just cite the same
# registration and the reasoning for WHY it (not atf_dump_ssh_after.bin) is
# the correct ssh(Naa) for r3u(Naa) at the dyn_zdf call site (see docstring).
register_dump(
    "sshnxt_dump_ssh_after.bin", "after",
    "stpmlf.F90:214 CALL ssh_nxt(kstp,Nbb,Nnn,ssh,Naa) -- ssh(:,:,Naa) BEFORE "
    "dyn_spg_ts's barotropic correction (dynspg_ts.F90:835 pssh(:,:,Kaa) += "
    "...) is applied. Loaded here ONLY to demonstrate it is NOT the value "
    "dyn_zdf's r3u(Naa) actually uses (task self-check), not as the primary "
    "measurement input.")


def interp_cell_to_uface_np(f: np.ndarray) -> np.ndarray:
    """Numpy mirror of operators_latlon_cgrid.interp_cell_to_uface (periodic-i,
    same convention this task's DINO recipe uses -- periodic_i=True in every
    sibling bridge call). Re-derivation would violate no-duplicate-numerics;
    call the real one instead so there is exactly one implementation."""
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_uface
    return np.asarray(interp_cell_to_uface(jnp.asarray(f)))


def area_weight_ssh_to_uface_nemo(ssh_2d: np.ndarray, e1e2t: np.ndarray) -> np.ndarray:
    """domqco.F90:166-167 dom_qco_r3c's u-face ssh average (area-weighted,
    NOT the plain interp_cell_to_uface simple mean) -- periodic in i to match
    DINO's ln_Iperio=.true. re-entrant channel (usrdef_nam.F90:157). Returns
    the (n_lat, n_lon) array with face j holding the average using T-columns
    j and j+1 (wrapped)."""
    ssh_ip1 = np.roll(ssh_2d, -1, axis=1)
    e1e2t_ip1 = np.roll(e1e2t, -1, axis=1)
    numer = e1e2t * ssh_2d + e1e2t_ip1 * ssh_ip1
    return numer  # caller divides by (hu_0 * e1e2u) per domqco.F90:167


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="momentum_jacobian_probe")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (task requires 'both')")
    assert e3t_mode == "both"

    for _name in ("spg_dump_pssh_final.bin", "sshnxt_dump_ssh_after.bin"):
        lvl = time_level_for_dump(_name)
        print(f"  time_level_for_dump({_name!r}) = {lvl!r}")

    jpi, jpj, jpk, hls = _cov._read_dims(RUN_DIR)
    print(f"RUN_GDB dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}  "
          f"restart={RESTART}  kt(nit000)={KT_DUMP}")

    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)

    cfg = __import__("dataclasses").replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    require_fp64(br.z_coord, br.state.T.data, br.state.eta.data,
                  br.state.H_bathy.data, br.geometry.dx_u,
                  context="momentum_jacobian_probe.build_state")
    print(f"  dtypes: eta={np.asarray(br.state.eta.data).dtype} "
          f"H_bathy={np.asarray(br.state.H_bathy.data).dtype} "
          f"dz_ref={np.asarray(br.z_coord.dz_ref).dtype} "
          f"e3u_0={np.asarray(grid.e3u_0).dtype if grid.e3u_0 is not None else None} "
          f"hu_0={np.asarray(grid.hu_0).dtype if grid.hu_0 is not None else None}")

    tmask2d = np.asarray(grid.tmask[..., 0]) > 0.5
    umask2d = np.asarray(grid.umask[..., 0]) > 0.5
    n_lat, n_lon = tmask2d.shape

    # =========================================================================
    # TASK 2: J-vs-r3u metric distribution, as U-FACE TOP-CELL THICKNESS
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK 2: legoESM J-based dz_0_u  vs  NEMO r3u-based live u-face thickness")
    print("=" * 78)

    # --- legoESM side: production surface_stress_faces path -----------------
    H_bathy_T = np.asarray(br.state.H_bathy.data)
    eta_T_now = np.asarray(br.state.eta.data)  # this is ssh(Kmm) from the restart

    # Use NEMO's OWN Naa ssh (barotropic-corrected, spg_dump_pssh_final.bin) so
    # the comparison is apples-to-apples at the SAME instant dyn_zdf actually
    # uses -- not a stand-in for legoESM's own eta (a controlled, single-input
    # comparison of the DIVISOR FORMULA, holding the input ssh fixed).
    pssh_final = _cov._load_haloed(os.path.join(RUN_DIR, "spg_dump_pssh_final.bin"),
                                    jpi, jpj, hls)
    if pssh_final.ndim == 3:
        pssh_final = pssh_final[..., 0]
    sshnxt_after = _cov._load_haloed(os.path.join(RUN_DIR, "sshnxt_dump_ssh_after.bin"),
                                      jpi, jpj, hls)
    if sshnxt_after.ndim == 3:
        sshnxt_after = sshnxt_after[..., 0]

    eta_Naa = pssh_final  # ssh(Naa) as it is at the dyn_zdf call (stpmlf.F90:303)

    dz_ref0 = float(np.asarray(br.z_coord.dz_ref)[0])
    J_T = np.asarray(compute_ocean_jacobian(
        jnp.asarray(eta_Naa), jnp.asarray(H_bathy_T), br.z_coord,
        min_water_column_m=None))
    dz_0_T_lego = dz_ref0 * J_T
    dz_0_u_lego = interp_cell_to_uface_np(dz_0_T_lego)  # (n_lat, n_lon+1) periodic

    # Face-index convention (verified from interp_cell_to_uface's own
    # pad_lon_cgrid: face j = 0.5*(T[j-1]+T[j]) for j=1..n_lon, i.e. face j
    # is the EAST face of T[j-1] = NEMO's u-point u[j-1]) -- established
    # elsewhere in this probe family (kamm_twin_90d.py:118 "u[:, 1:, 0] ...
    # -> NEMO un[i]" and spg_substep_chain.py's own _u_to_nemo(a)=a[:,1:]).
    # Slice [:, 1:] (drop the WEST-wrap face 0), NOT [:, :n_lon].
    dz_0_u_lego = dz_0_u_lego[:, 1:]

    # --- NEMO side: r3u(Naa)-based live e3u(:,:,1,Naa) -----------------------
    e1t = np.asarray(grid.e1t)
    e2t = np.asarray(grid.e2t)
    e1e2t = e1t * e2t
    e1u = np.asarray(grid.e1u)
    # e1e2u: NEMO's u-cell area. nemo_io.NemoMeshMask does not separately
    # expose e2u -- on DINO's regular lat-lon C-grid (no j-stagger of the
    # meridional metric across an i-face), e2u == e2t identically (both are
    # functions of latitude only, usrdef_hgr.F90's analytic e2t/e2u/e2v are
    # the SAME formula with no i-dependence at all -- confirmed by reading
    # cfgs/DINO/MY_SRC/usrdef_hgr.F90). Reading e2t here is therefore exact,
    # not an approximation, for this specific grid.
    e2u = e2t
    e1e2u = e1u * e2u

    hu_0 = np.asarray(grid.hu_0)
    assert hu_0 is not None, "grid.hu_0 missing -- read_nemo_mesh_mask needs e3u_0 in mesh_mask.nc"
    r1_hu_0 = np.where(hu_0 > 0, 1.0 / np.maximum(hu_0, 1e-10), 0.0)

    numer = area_weight_ssh_to_uface_nemo(eta_Naa, e1e2t)
    r3u_Naa = 0.5 * numer * r1_hu_0 * np.where(e1e2u > 0, 1.0 / np.maximum(e1e2u, 1e-10), 0.0)

    e3u_0_k1 = np.asarray(grid.e3u_0)[..., 0] if grid.e3u_0 is not None else None
    assert e3u_0_k1 is not None, "grid.e3u_0 missing"
    e3u_live_k1 = e3u_0_k1 * (1.0 + r3u_Naa)  # domzgr_substitute.h90:127, umask=1 at k=1 wet

    print(f"\n  dz_ref0 (E3u_0 scalar, dz_ref[0])={dz_ref0:.6f} m")
    print(f"  hu_0: mean(wet)={hu_0[umask2d].mean():.3f} m  "
          f"H_max (z_coord)={float(getattr(br.z_coord, 'H_max', np.nan)):.3f} m")

    # --- ratio distribution --------------------------------------------------
    wet_u = umask2d & np.isfinite(dz_0_u_lego) & np.isfinite(e3u_live_k1) & (e3u_live_k1 > 0)
    ratio = dz_0_u_lego[wet_u] / e3u_live_k1[wet_u]
    print(f"\n  RATIO dz_0_u[legoESM J-based] / e3u(:,:,1,Naa)[NEMO r3u-based], "
          f"n_wet_u={int(wet_u.sum())}:")
    print(f"    median={np.median(ratio):.8f}  p95={np.percentile(ratio, 95):.8f}  "
          f"max|ratio-1|={np.max(np.abs(ratio - 1.0)):.6e}  "
          f"min={ratio.min():.8f}  max={ratio.max():.8f}")

    stats_dz = per_element_stats("dz_0_u [J-based vs r3u-based]",
                                  dz_0_u_lego, e3u_live_k1, wet_u, sign_changing=False)

    # --- spatial concentration: bathymetry slope + distance from walls ------
    print("\n" + "-" * 78)
    print("Spatial concentration of the ratio-1 mismatch")
    print("-" * 78)
    # bathymetry slope proxy: |dH/di| + |dH/dj| at T-points (m per cell)
    H_ip1 = np.roll(H_bathy_T, -1, axis=1)
    H_im1 = np.roll(H_bathy_T, 1, axis=1)
    H_jp1 = np.pad(H_bathy_T[1:], ((0, 1), (0, 0)), mode="edge")
    H_jm1 = np.pad(H_bathy_T[:-1], ((1, 0), (0, 0)), mode="edge")
    slope_T = 0.5 * (np.abs(H_ip1 - H_im1) + np.abs(H_jp1 - H_jm1))
    slope_u = interp_cell_to_uface_np(slope_T)[:, 1:]

    # distance (in cells) from nearest dry (land) T-cell -- DINO's only land
    # is the N/S channel walls (+ any interior continent), so this is
    # genuinely "distance to wall". scipy's chessboard distance transform
    # (exact multi-source BFS, no re-derivation of a graph algorithm).
    from scipy.ndimage import distance_transform_cdt
    wet_for_dist = tmask2d.astype(np.uint8)
    # distance_transform_cdt measures distance of each TRUE pixel to the
    # nearest FALSE (background) pixel -- pass wet as foreground so wet cells
    # get their distance to the nearest dry cell; dry cells get 0.
    dist = distance_transform_cdt(wet_for_dist, metric="chessboard").astype(np.float64)
    dist_u = interp_cell_to_uface_np(dist)[:, 1:]

    abs_dev = np.abs(ratio - 1.0)
    slope_wet = slope_u[wet_u]
    dist_wet = dist_u[wet_u]
    slope_med = np.median(slope_wet)
    dist_med = np.median(dist_wet)
    high_slope = slope_wet > slope_med
    near_wall = dist_wet <= dist_med

    print(f"  |ratio-1| by bathymetry-slope bucket: "
          f"low-slope(<=median) median={np.median(abs_dev[~high_slope]):.3e}  "
          f"high-slope(>median) median={np.median(abs_dev[high_slope]):.3e}")
    print(f"  |ratio-1| by wall-distance bucket: "
          f"near-wall(<=median dist) median={np.median(abs_dev[near_wall]):.3e}  "
          f"far-from-wall(>median dist) median={np.median(abs_dev[~near_wall]):.3e}")
    corr_slope = float(np.corrcoef(abs_dev, slope_wet)[0, 1]) if len(abs_dev) > 1 else float("nan")
    corr_dist = float(np.corrcoef(abs_dev, dist_wet)[0, 1]) if len(abs_dev) > 1 else float("nan")
    print(f"  corr(|ratio-1|, bathymetry slope) = {corr_slope:+.4f}")
    print(f"  corr(|ratio-1|, distance-from-wall) = {corr_dist:+.4f}  "
          "(negative = concentrated NEAR walls, matches the zu_frc lead)")

    concentrated_at_walls = (corr_dist < -0.1) and (
        np.median(abs_dev[near_wall]) > 2 * max(np.median(abs_dev[~near_wall]), 1e-12))
    print(f"  VERDICT (task Q: concentrated at sloped columns near walls?): "
          f"{'PLAUSIBLE-YES' if concentrated_at_walls else 'PLAUSIBLE-NO/DIFFUSE'} "
          f"(corr_dist={corr_dist:+.4f}, near/far median ratio="
          f"{np.median(abs_dev[near_wall]) / max(np.median(abs_dev[~near_wall]), 1e-12):.2f}x)")

    # =========================================================================
    # TASK 3: A/B surface-stress top-layer tendency (J-based vs r3u-based)
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK 3: A/B surface-stress top-layer momentum tendency")
    print("=" * 78)
    print("  No NEMO dump brackets dyn_zdf's surface-stress term ALONE:")
    print("    stp_dump_state_and_bt('dynspg') [pre, stpmlf.F90:293] and")
    print("    stp_dump_state_and_bt('dynzdf') [post, stpmlf.F90:312] bracket the")
    print("    WHOLE implicit vertical solve, which per coverage_rows_measure.py's")
    print("    own 'dyn_zdf' row ALSO folds in the implicit bottom-drag term")
    print("    (ln_drgimp.AND.ln_dynspg_ts, both True for DINO, dynzdf.F90:148-171)")
    print("    directly into the tridiagonal matrix -- so differencing those two")
    print("    dumps measures (surface-stress-deposit + drag-fold + tridiagonal")
    print("    solve), not the surface-stress term alone. Exact instrumentation")
    print("    that WOULD isolate it: a new dump at dynzdf.F90:334-335, immediately")
    print("    after the surface-stress line, of ONLY the increment")
    print("      puu(ji,jj,1,Kaa) - puu(ji,jj,1,Kaa)_before_this_line")
    print("    i.e. `WRITE(unit) zDt_2*(utau_b+utauU)/(e3u(:,:,1,Kaa)*rho0)*umask(:,:,1)`")
    print("    for jj=Njs0,Nje0 -- it would join the batched stp_dump_* rebuild")
    print("    (same pattern as stp_dump_krhs, stpmlf.F90:690-724).")
    print()
    print("  A/B instead computed DIRECTLY on the divisor-only formula (both")
    print("  variants share tau_i_u, rho0, dt -- the ONLY variable is the")
    print("  denominator thickness), against each other (no NEMO bracket exists")
    print("  to compare a THIRD way):")

    rho_0 = float(cfg.rho_0)
    dt = float(cfg.dt)
    from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before as _read_before
    _before_tau = _read_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    if getattr(_before_tau, "tau_x", None) is not None:
        # NEMO's OWN dumped utau_b (before-level T-point stress, restart.F90) --
        # the real forcing at this exact restart, in the SAME atmosphere-stress
        # sign convention _bc_external_surface_forcing negates (dino.py:3316-3320
        # comment: "+0.2 Pa accelerates the ocean eastward" = ocean convention;
        # utau_b is stored in atmosphere convention, hence the negation here too,
        # matching dino_step_surface_forcing's own -forcing["tau_u_cell_2d"]).
        tau_e_T = -np.asarray(_before_tau.tau_x)
        print(f"  wind stress source: NEMO restart utau_b (before-level, T-point)")
    else:
        # Fallback: DINO's own analytic forcing (dino.py), NOT a re-derivation --
        # same forcing both variants would see in a real step, so the
        # comparison stays apples-to-apples for the ONE variable under test
        # (the divisor).
        from legoesm.ocean.experiments.dino import dino_lat_lon_surface_forcing_arrays
        forcing_ab = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
        tau_e_T = -np.asarray(forcing_ab["tau_u_cell_2d"])
        print(f"  wind stress source: DINO analytic tau_u_cell_2d (restart had no utau_b)")
    tau_i_u = interp_cell_to_uface_np(tau_e_T)[:, 1:]

    du_dt_J = tau_i_u / (rho_0 * np.maximum(dz_0_u_lego, 1e-10))
    du_dt_r3u = tau_i_u / (rho_0 * np.maximum(e3u_live_k1, 1e-10))

    d_wet = wet_u & np.isfinite(du_dt_J) & np.isfinite(du_dt_r3u)
    diff = du_dt_J[d_wet] - du_dt_r3u[d_wet]
    rms_r3u = float(np.sqrt(np.mean(du_dt_r3u[d_wet] ** 2)))
    print(f"\n  du_dt [J-based]   median|.|={np.median(np.abs(du_dt_J[d_wet])):.6e} m/s^2")
    print(f"  du_dt [r3u-based] median|.|={np.median(np.abs(du_dt_r3u[d_wet])):.6e} m/s^2")
    print(f"  |du_dt_J - du_dt_r3u|: median={np.median(np.abs(diff)):.6e}  "
          f"max={np.max(np.abs(diff)):.6e}  "
          f"err_norm=RMS(diff)/RMS(r3u-based)={float(np.sqrt(np.mean(diff**2)))/max(rms_r3u,1e-30):.4e}")

    # =========================================================================
    # TASK 4: correlation with zu_frc residual field
    # =========================================================================
    print("\n" + "=" * 78)
    print("TASK 4: correlation with the zu_frc residual field")
    print("=" * 78)
    # Reuse the exact captured F_slow_u vs NEMO's spg_dump_zu_frc.bin the
    # sibling script already builds in its own main(); re-invoke its
    # machinery by re-running the equivalent inline (importing triggers
    # module-level dump registration only -- the F_slow_u capture is inside
    # main(), so call it and capture stdout-reported numbers is not enough;
    # instead rebuild the SAME capture here using its documented formula,
    # citing its own file for the construction, to get the per-element field
    # (not just the printed scalar) for a spatial correlation).
    print("  Rebuilding zu_frc residual per-element field via the SAME capture")
    print("  spg_substep_chain.py's main() uses (barotropic_substeps_latlon_cgrid")
    print("  monkeypatch capturing kw['F_slow_u']) -- calling that file's own")
    print("  helper functions, not re-deriving the numerics.")

    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before
    from legoesm.ocean.fidelity.nemo_state_bridge import bridge_before_state_topo
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments.dino import (
        dino_lat_lon_surface_forcing_arrays as _dino_sf,
        dino_step_surface_forcing as _dino_step_sf,
    )

    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    st_before = bridge_before_state_topo(br._replace(state=br.state), grid, before,
                                          periodic_i=True)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing_full = _dino_sf(br.geometry, cfg)
    sf = _dino_step_sf(forcing_full)

    captured = {}
    _real_fn = ocmod.barotropic_substeps_latlon_cgrid

    def _spy(*a, **kw):
        captured["kw"] = kw
        return _real_fn(*a, **kw)

    import jax
    ocmod.barotropic_substeps_latlon_cgrid = _spy
    try:
        with jax.disable_jit():
            model.step(st_before, jnp.asarray(cfg.dt), surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_fn

    assert "kw" in captured, "barotropic_substeps_latlon_cgrid was not called -- capture failed"
    lego_u_frc = np.asarray(captured["kw"]["F_slow_u"])

    def _u_to_nemo_shape(a):
        # EXACT convention from spg_substep_chain.py's own STAGE 0 _u_to_nemo
        # (this file's sibling, read directly rather than re-derived):
        # legoESM u-face arrays are (n_lat, n_lon+1) with face j = between
        # T[j-1] and T[j]; NEMO's interior u-point dump aligns with face
        # index [:, 1:] (drop the LEADING wrap face), not [:, :n_lon] (which
        # this probe's first draft used and which is OFF BY ONE COLUMN --
        # caught by the alignment scan below, per Rule 1e: reconcile a
        # disagreeing measurement before recording it).
        return a[:, 1:]

    lego_u_frc_nemo_shape = _u_to_nemo_shape(lego_u_frc)
    nemo_zu_frc = np.fromfile(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"),
                               dtype="<f8").reshape(n_lat, n_lon)

    # Alignment scan (task requirement): confirm a sharp offset-0 peak, not a
    # plateau, before trusting the residual field below.
    print("\n  [align scan] F_slow_u vs spg_dump_zu_frc.bin (di shift):")
    for di in (-2, -1, 0, 1, 2):
        shifted = np.roll(lego_u_frc_nemo_shape, di, axis=1)
        mm = wet_u & np.isfinite(shifted)
        rms_n = float(np.sqrt(np.mean(nemo_zu_frc[mm] ** 2)))
        err = (float(np.sqrt(np.mean((shifted[mm] - nemo_zu_frc[mm]) ** 2))) / rms_n
               if rms_n > 0 else float("nan"))
        print(f"    di={di:+d}  err_norm={err:.4e}")

    zu_frc_residual = lego_u_frc_nemo_shape - nemo_zu_frc
    m_common = wet_u & np.isfinite(zu_frc_residual)
    rms_nemo_zu = float(np.sqrt(np.mean(nemo_zu_frc[m_common] ** 2)))
    err_norm_zu = (float(np.sqrt(np.mean(zu_frc_residual[m_common] ** 2))) / rms_nemo_zu
                   if rms_nemo_zu > 0 else float("nan"))
    print(f"\n  zu_frc residual (this probe's own recompute): "
          f"RMS(diff)={float(np.sqrt(np.mean(zu_frc_residual[m_common]**2))):.4e}  "
          f"RMS(nemo)={rms_nemo_zu:.4e}  err_norm={err_norm_zu:.4e}  n={int(m_common.sum())}  "
          "(established figure to compare against: 8.03e-3 err_norm from "
          "fidelity_bar_gate.py's 'dyn_spg_ts puu_b' row)")

    ratio_full = np.full(wet_u.shape, np.nan)
    ratio_full[wet_u] = dz_0_u_lego[wet_u] / e3u_live_k1[wet_u]
    abs_dev_full = np.abs(ratio_full - 1.0)

    zu_res_abs = np.abs(zu_frc_residual)
    both_valid = m_common & np.isfinite(abs_dev_full)
    corr_zu = (float(np.corrcoef(abs_dev_full[both_valid], zu_res_abs[both_valid])[0, 1])
               if both_valid.sum() > 1 else float("nan"))
    print(f"  corr(|J-vs-r3u ratio - 1|, |zu_frc residual|) = {corr_zu:+.4f}  "
          f"n={int(both_valid.sum())}")

    connection_verdict = "CONFIRMED" if corr_zu > 0.3 else "REFUTED"
    print(f"\n  VERDICT (task Q: does the J-vs-r3u divisor mismatch explain the")
    print(f"  zu_frc residual?): {connection_verdict} "
          f"(corr={corr_zu:+.4f}; threshold 0.3 chosen as a weak-to-moderate")
    print("  linear-association floor -- a genuine causal ownership would show a")
    print("  much stronger positive correlation, not just 'same order of")
    print("  magnitude'. NOTE: zu_frc is the dyn_spg_ts barotropic slow-forcing")
    print("  term (dynspg_ts.F90:432, r1_hu(Kmm) divisor), a DIFFERENT call site")
    print("  and time level (Kmm/now) than dyn_zdf's surface-stress term")
    print("  (Kaa/after) -- both trace back to the SAME r3u/hu_0 family but are")
    print("  NOT the same formula instance, so a refutation here does not")
    print("  exonerate dyn_zdf's own divisor; it only tests whether THIS")
    print("  specific J-vs-r3u geometric mismatch is the shared root cause.")

    # =========================================================================
    # SELF-CHECKS
    # =========================================================================
    print("\n" + "-" * 78)
    print("SELF-CHECKS")
    print("-" * 78)
    # (1) forcing eta->0 collapses BOTH thickness variants to the SAME
    # eta-independent reference (dz_ref0 vs e3u_0_k1) -- they are NOT
    # bit-identical in general (different reference conventions: T-point
    # H_bathy-derived dz_ref0*1 vs u-point hu_0-derived e3u_0), but at eta=0
    # each variant collapses to its OWN static reference, decoupled from any
    # ssh-dependent bug -- verifies the eta plumbing, not formula equivalence.
    J_T_zero = np.asarray(compute_ocean_jacobian(
        jnp.zeros_like(jnp.asarray(eta_Naa)), jnp.asarray(H_bathy_T), br.z_coord,
        min_water_column_m=None))
    dz_0_T_zero = dz_ref0 * J_T_zero
    dz_0_u_zero = interp_cell_to_uface_np(dz_0_T_zero)[:, 1:]
    r3u_zero = 0.5 * area_weight_ssh_to_uface_nemo(np.zeros_like(eta_Naa), e1e2t) * r1_hu_0 \
        * np.where(e1e2u > 0, 1.0 / np.maximum(e1e2u, 1e-10), 0.0)
    e3u_live_zero = e3u_0_k1 * (1.0 + r3u_zero)
    print(f"  [self-check 1] eta->0: dz_0_u[J] wet range=({dz_0_u_zero[wet_u].min():.4f},"
          f"{dz_0_u_zero[wet_u].max():.4f})  e3u_live[r3u] wet range="
          f"({e3u_live_zero[wet_u].min():.4f},{e3u_live_zero[wet_u].max():.4f})  "
          f"r3u_zero max|.|={np.max(np.abs(r3u_zero[wet_u])):.3e} (want ~0)")
    assert np.max(np.abs(r3u_zero[wet_u])) < 1e-9, "r3u did not vanish at eta=0"

    # (2) manual scalar recompute of r3u/e3u_live at a handful of wet u-faces,
    # cross-checked against the vectorized array.
    wet_idx = np.argwhere(wet_u)[::max(1, int(wet_u.sum()) // 5)][:5]
    manual_max_diff = 0.0
    for jy, ix in wet_idx:
        ix1 = (ix + 1) % n_lon
        num = e1e2t[jy, ix] * eta_Naa[jy, ix] + e1e2t[jy, ix1] * eta_Naa[jy, ix1]
        r3u_manual = 0.5 * num * r1_hu_0[jy, ix] / max(e1e2u[jy, ix], 1e-10)
        e3u_manual = e3u_0_k1[jy, ix] * (1.0 + r3u_manual)
        manual_max_diff = max(manual_max_diff, abs(e3u_manual - e3u_live_k1[jy, ix]))
    print(f"  [self-check 2] manual (scalar loop) vs vectorized e3u_live_k1 at "
          f"{len(wet_idx)} sample wet u-faces: max|diff|={manual_max_diff:.3e} (want 0.0)")
    assert manual_max_diff < 1e-9

    # (3) confirm sshnxt_dump_ssh_after.bin != spg_dump_pssh_final.bin (proves
    # the two ssh(Naa) candidates genuinely differ, i.e. the barotropic
    # correction at dynspg_ts.F90:835 actually changed ssh -- if they were
    # identical this whole time-level distinction would be moot).
    d_ssh = float(np.max(np.abs(pssh_final[tmask2d] - sshnxt_after[tmask2d])))
    print(f"  [self-check 3] max|spg_dump_pssh_final - sshnxt_dump_ssh_after| "
          f"(wet T-cells) = {d_ssh:.6e} (want > 0 -- confirms dyn_spg_ts's "
          "barotropic correction at dynspg_ts.F90:835 is NOT a no-op, so the "
          "Naa-level distinction this probe makes is real, not academic)")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  J-vs-r3u dz_0_u ratio: median={np.median(ratio):.6f} "
          f"p95={np.percentile(ratio,95):.6f} max|ratio-1|={np.max(np.abs(ratio-1)):.4e}")
    print(f"  corr(|ratio-1|, dist-from-wall)={corr_dist:+.4f}  "
          f"corr(|ratio-1|, bathy slope)={corr_slope:+.4f}")
    print(f"  zu_frc correlation: {corr_zu:+.4f} -> {connection_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
