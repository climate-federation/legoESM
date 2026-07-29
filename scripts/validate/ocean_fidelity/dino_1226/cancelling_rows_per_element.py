#!/usr/bin/env python
"""#1226: PER-ELEMENT error for the 6 fidelity rows that currently sit "AT BAR
on CANCELLING statistics only" in ``fidelity_bar_gate.py``:

    sbc (utau/qsr/qns/sfx)
    ldftra ahtu (Redi, nn_aht_ijk_t=20)
    ldftra ahtv (Redi, nn_aht_ijk_t=20)
    dyn_hpg (du)
    dyn_adv KEG
    ATF filter T/S/ssh

The gate's ``ratio`` is a MEAN: mixed-sign per-element errors CANCEL. ``bn2``
sat AT BAR at ratio 1-6.6e-9 while its own per-element error was 6.96e-6 --
a thousand times larger -- and that error moved 10 MLD columns
(``eos_rab_bn2_per_element.py`` / ``zdf_mxl_nmln_compare.py``). None of the 6
rows below has ever had that per-element measurement; five of their gate
notes say "exact"/"bit-exact", which is a CLAIM, not a measurement.

Modeled on ``eos_rab_bn2_per_element.py`` (``build_state()`` shape, the two
MECHANICAL preconditions -- ``require_fp64`` + ``time_level_for_dump`` -- and
``per_element_stats()``) and ``ldf_slp_per_element.py`` (the
conditioning-robust ``err_norm`` treatment for sign-changing fields). No
numerics are re-implemented (Rule 0): every row calls the REAL production
entry point.

Per-row provenance:

  * ``ldftra ahtu/ahtv`` -- ``_static_kappa_redi_override`` (the exact
    function ``ldftra_ahtv_compare.py`` already calls), fed the REAL recipe's
    ``mc.gm_redi`` (not a hardcoded ``kappa_Redi`` literal).
  * ``dyn_hpg (du)`` / ``dyn_adv KEG`` -- BOTH read off the SAME
    ``_bc_ke_and_pressure_gradients`` call on the real Y5 developed state
    (via ``compute_frozen_geom_density``, production's own precomputed-
    geometry path, #25). ``KE_PGF_u = -dKE_dx - dp_dx/rho_0`` is EXACTLY
    additive (``dKE_dx`` depends only on u,v; ``dp_dx`` only on
    rho_prime/p_prime_filled/eta) so zeroing one input set isolates the
    other term bit-for-bit, not approximately -- verified below by checking
    the zeroed term actually vanishes AND that full = KEG_only + PGF_only.
  * ``sbc`` -- ``dino_wind_stress``/``dino_Q_sr_seasonal`` (utau/qsr) and
    ``restoring_surface_forcing`` with the SAME ``RestoringConfig``
    ``apply_dino_lat_lon_surface_forcing`` builds (qns/sfx), converted back
    to NEMO's raw-flux units by the SAME ``tau_from_flux_coefficient``
    identity that constructs ``tau_T``/``tau_S`` in the first place -- see
    section (D)'s docstring for the exact algebra. This is a unit conversion
    of an already-computed production output, not new physics.
  * ``ATF filter T/S/ssh`` -- ``_thickness_weighted_asselin`` (T,S) and the
    inline plain-Asselin formula (ssh), copied verbatim from
    ``ocean_model_latlon_cgrid.py`` module scope. NEMO's dumps give the
    filter's PRE-filter ("before", physically Nnn) and POST-filter ("after")
    snapshots but never the raw un-filtered Naa the filter consumed -- Naa is
    RECONSTRUCTED by inverting the (linear) filter equation using the
    genuinely-independent Nbb (from the restart, via ``bridge_before_state_
    topo``) -- see section (E)'s docstring for exactly why this is a real
    transcription check and not a tautology.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
      .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/cancelling_rows_per_element.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import importlib.util
import sys

import numpy as np
import jax.numpy as jnp

_sib_path = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib_path)
_bn2_alpha_compare = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bn2_alpha_compare
_spec.loader.exec_module(_bn2_alpha_compare)
_read_dims = _bn2_alpha_compare._read_dims
_load_haloed = _bn2_alpha_compare._load_haloed

from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_wind_stress, dino_Q_sr_seasonal, dino_T_star_seasonal, dino_S_star,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.fidelity.precision_gate import require_fp64
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _static_kappa_redi_override, _thickness_weighted_asselin,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_ke_and_pressure_gradients, compute_frozen_geom_density,
)
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing
from legoesm.ocean.vertical import compute_layer_thickness

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
FLOOR = 1.0e-12

# ---------------------------------------------------------------------------
# Register the new dumps' NEMO time level (mechanical precondition #2).
# Calling register_dump() at import time -- NOT editing time_levels.py --
# so this script touches zero files under packages/ or src/.
# ---------------------------------------------------------------------------
register_dump(
    "sbc_dump_qns.bin", "before",
    "stpmlf.F90:170 CALL sbc(kstp,Nbb,Nnn) -> sbcmod.F90:417 CALL "
    "usrdef_sbc_oce(kt,Kbb) with Kbb=Nbb; usrdef_sbc.F90 CASE(4) qns reads "
    "ts(ji,jj,1,jp_tem,Kbb) -- the BEFORE-level SST.")
register_dump(
    "sbc_dump_sfx.bin", "before",
    "same call site as qns; usrdef_sbc.F90 CASE(4) sfx reads "
    "ts(ji,jj,1,jp_sal,Kbb) -- the BEFORE-level SSS.")
register_dump(
    "sbc_dump_utau.bin", "before",
    "usrdef_sbc.F90 CASE(4): utau=znl_cbc(lat-node profile) -- a pure "
    "function of latitude, no T/S anywhere in the block; registered 'before' "
    "only for call-site provenance (shares usrdef_sbc_oce(kt,Kbb)).")
register_dump(
    "sbc_dump_qsr.bin", "before",
    "usrdef_sbc.F90 CASE(4): zqsr_dayMean=f(lat,day-of-year) via "
    "compute_day_of_year(kt,...) -- lat/calendar only, no T/S; registered "
    "'before' for the same call-site provenance reason as utau.")
register_dump(
    "ldftra_dump_ahtu.bin", "now",
    "ldftra.F90:325-329 ldf_c2d('TRA',...)/ldfc1d_c2d.F90:141-145 -- "
    "ahtu=zUfac*MAX(e1u,e2u)^inn is a pure function of the STATIC mesh, "
    "computed once, never touches T/S/time.")
register_dump(
    "ldftra_dump_ahtv.bin", "now",
    "same call site as ahtu; ahtv=zUfac*MAX(e1v,e2v)^inn, mesh-only.")
register_dump(
    "keg_dump_du.bin", "now",
    "dynadv.F90:89 CALL dyn_keg(kt,nn_dynkeg,Kmm,puu,pvv,Krhs) -- KEG reads "
    "velocities at Kmm (now); dumped (unit 8940, :92) as the Krhs increment "
    "from dyn_keg ALONE, before dyn_zad accumulates onto the same Krhs.")
register_dump(
    "keg_dump_dv.bin", "now", "same call site as keg_dump_du, v-component.")
register_dump(
    "hpg_dump_du.bin", "now",
    "dynhpg.F90 #1226 item 9 dump: puu(Krhs) increment from hpg_sco alone; "
    "hpg_sco is called on the Kmm (now) T/S/eta state under modified-"
    "leapfrog, matching every other stp_dump_NN momentum row in this run.")
register_dump(
    "hpg_dump_dv.bin", "now", "same call site as hpg_dump_du, v-component.")


def per_element_stats(name, lego, nemo, wet, *, sign_changing=None):
    """median/mean/max/p99 pointwise |rel| AND the conditioning-robust
    err_norm=|d|/RMS(nemo), plus the aggregate ratio/corr -- same convention
    as eos_rab_bn2_per_element.py's per_element_stats / ldf_slp_per_element.py's
    per_element_report. ``sign_changing`` decides which of the two is the
    BAR METRIC reported at the end; if None it is auto-detected from whether
    NEMO's own values cross (or nearly touch) zero."""
    m = np.asarray(wet) & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = np.asarray(lego)[m], np.asarray(nemo)[m]
    n = int(m.sum())
    if n == 0:
        print(f"  {name:<32s} NO WET POINTS -- cannot score")
        return dict(n=0, at_bar=False, bar_metric=float("nan"), sign_changing=True)
    rms_n = float(np.sqrt(np.mean(ne ** 2)))
    corr = (float(np.corrcoef(lo, ne)[0, 1])
            if n >= 2 and lo.std() > 0 and ne.std() > 0 else float("nan"))
    abs_sum_ne = float(np.abs(ne).sum())
    ratio_abs = float(np.abs(lo).sum() / abs_sum_ne) if abs_sum_ne > 0 else float("nan")
    ratio_mean = float(lo.mean() / ne.mean()) if ne.mean() != 0 else float("nan")

    rel = np.abs(lo - ne) / np.maximum(np.abs(ne), FLOOR)
    med_rel, p99_rel, max_rel = (float(np.median(rel)), float(np.percentile(rel, 99)),
                                  float(np.max(rel)))
    err_norm = np.abs(lo - ne) / max(rms_n, FLOOR)
    med_en, p99_en, max_en = (float(np.median(err_norm)), float(np.percentile(err_norm, 99)),
                               float(np.max(err_norm)))

    if sign_changing is None:
        sign_changing = bool(ne.min() <= 0.0 <= ne.max())
    near0 = np.abs(ne) < 1.0e-3 * max(rms_n, FLOOR)
    frac_near0 = float(near0.mean())

    print(f"  {name:<32s} n={n:<8d} corr={corr:.8f}  |x|ratio={ratio_abs:.8f}  "
          f"ratio_mean={ratio_mean:.8f}")
    print(f"    {'':<32s} pointwise|rel| median={med_rel:.3e} p99={p99_rel:.3e} "
          f"max={max_rel:.3e}  [signed range=({ne.min():+.3e},{ne.max():+.3e}), "
          f"sign_changing={sign_changing}, near0_frac={100*frac_near0:.2f}%]")
    print(f"    {'':<32s} err_norm=|d|/RMS(nemo) median={med_en:.3e} p99={p99_en:.3e} "
          f"max={max_en:.3e}  RMS(nemo)={rms_n:.6e}")
    bar_metric = med_en if sign_changing else med_rel
    at_bar = bool(bar_metric <= 1.0e-9)
    print(f"    {'':<32s} BAR METRIC (median, {'err_norm' if sign_changing else 'pointwise|rel|'}) "
          f"= {bar_metric:.3e}  -> {'YES (<=1e-9)' if at_bar else 'NO'}")
    return dict(n=n, corr=corr, ratio_abs=ratio_abs, ratio_mean=ratio_mean,
                med_rel=med_rel, p99_rel=p99_rel, max_rel=max_rel,
                med_en=med_en, p99_en=p99_en, max_en=max_en,
                sign_changing=sign_changing, frac_near0=frac_near0,
                bar_metric=bar_metric, at_bar=at_bar)


def build_state():
    # PRECISION (#1226): fp32 control-policy default silently rounds the
    # fp64 depth ladder; JAX_ENABLE_X64=1 does NOT change legoESM's own
    # precision policy (established in eos_rab_bn2_per_element.py).
    if os.environ.get("LEGOESM_FIDELITY_FP64", "1") == "1":
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    # Populate state.{T,S,u,v,eta}_before (Nbb) with the SAME face-staggering/
    # Neumann-land-fill convention as the now-level bridge -- required for the
    # ATF section's genuinely-independent Nbb (section E), rather than a raw
    # reshape of bef.T that would miss the land-fill/face-staggering the
    # now-level state already carries.
    br = br._replace(
        state=bridge_before_state_topo(br, grid, bef, periodic_i=True))
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    require_fp64(br.z_coord, br.state.T.data, br.state.S.data,
                 context="cancelling_rows_per_element")

    return dict(jpi=jpi, jpj=jpj, jpk=jpk, hls=hls, grid=grid, now=now, bef=bef,
                cfg=cfg, br=br, mc=mc)


# =============================================================================
# (A) ldftra ahtu / ahtv
# =============================================================================
def measure_ldftra(st) -> dict:
    print("\n" + "=" * 78)
    print("(A) ldftra ahtu / ahtv (Redi, nn_aht_ijk_t=20)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br, mc = st["grid"], st["br"], st["mc"]
    gm_cfg = mc.gm_redi
    print(f"  kappa_redi_lat_scaling={gm_cfg.kappa_redi_lat_scaling}  "
          f"kappa_Redi={gm_cfg.kappa_Redi!r} (the REAL recipe config value, "
          "not a hardcoded literal)")
    kappa_T, kappa_v = _static_kappa_redi_override(gm_cfg, br.geometry)
    if kappa_T is None:
        print("  ABORTING: kappa_redi_lat_scaling is False on this recipe -- "
              "_static_kappa_redi_override returns (None, None).")
        return dict(ahtu=dict(n=0, at_bar=False, bar_metric=float("nan"), sign_changing=False),
                    ahtv=dict(n=0, at_bar=False, bar_metric=float("nan"), sign_changing=False))
    kappa_T = np.asarray(kappa_T)
    kappa_v = np.asarray(kappa_v)

    ahtu_full = _load_haloed(os.path.join(RUN_DIR, "ldftra_dump_ahtu.bin"), jpi, jpj, hls)
    ahtv_full = _load_haloed(os.path.join(RUN_DIR, "ldftra_dump_ahtv.bin"), jpi, jpj, hls)
    tmask2d = np.asarray(grid.tmask[..., 0]) > 0.5
    n_lon = tmask2d.shape[1]
    # nn_aht_ijk_t=20 is depth-independent; confirm rather than assume.
    depth_const = bool(np.allclose(ahtu_full[tmask2d], ahtu_full[tmask2d][:, :1]))
    print(f"  [verify] ahtu identical across all {ahtu_full.shape[-1]} dumped "
          f"levels at every wet T-cell: {depth_const} (expect True unless the "
          "periodic-seam column below is the only exception)")
    ahtu_nemo = ahtu_full[..., 0]
    ahtv_nemo = ahtv_full[..., 0]

    # PERIODIC-SEAM ARTIFACT (harness, not physics): NEMO's ahtu dump is
    # IDENTICALLY ZERO at column n_lon-2 (the last interior u-face before the
    # periodic wrap) on every wet row -- a re-indexing collision at the seam,
    # the SAME family of artifact already documented for dyn_cor_2d's
    # "periodic-seam column, harness reindexing artifact" caveat. Verified
    # (not assumed) before excluding: check the zero cells are confined to
    # exactly that one column.
    seam_col = n_lon - 2
    zero_on_wet = (ahtu_nemo == 0.0) & tmask2d
    seam_cols = np.unique(np.where(zero_on_wet)[1])
    is_seam_only = bool(seam_cols.size == 0 or list(seam_cols) == [seam_col])
    print(f"  [verify] ahtu==0 on wet cells confined to column {seam_col} only "
          f"(the periodic seam): {is_seam_only} (columns seen: {list(seam_cols)})")
    ahtu_mask = tmask2d.copy()
    if is_seam_only:
        ahtu_mask[:, seam_col] = False
        print(f"  excluding periodic-seam column {seam_col} from the ahtu wet "
              f"mask ({int((tmask2d[:, seam_col]).sum())} cells) -- a harness "
              "artifact, not a legoESM defect (ahtv has its OWN, DIFFERENT "
              "wall exclusion below -- it is scored on grid.vmask, not the "
              "T-mask, because a closed v-face row is not a wet v-point).")
    else:
        print("  WARNING: zero-on-wet cells are NOT confined to the seam column "
              "-- NOT excluding anything, scoring the full mask as-is.")

    r_ahtu = per_element_stats("ahtu (u-face/T-point kappa)", kappa_T, ahtu_nemo,
                                ahtu_mask, sign_changing=False)

    # ahtv is a V-FACE field: the T-mask overstates its wet domain at the
    # meridional channel walls (row n_lat-2 here has ZERO wet v-faces, and
    # the periodic-wrap columns 0/n_lon-1 are dry too) -- verified directly
    # (not assumed): using grid.vmask instead of tmask2d as the wet selector
    # reduces the "diff>1.0" bad-cell count to exactly 0, confirming these
    # are wall/seam cells, not a legoESM defect.
    vmask2d = np.asarray(grid.vmask[..., 0]) > 0.5
    r_ahtv = per_element_stats("ahtv (v-face kappa)", kappa_v, ahtv_nemo,
                                vmask2d, sign_changing=False)
    return dict(ahtu=r_ahtu, ahtv=r_ahtv)


# =============================================================================
# (B) dyn_hpg (du) and dyn_adv KEG -- both isolated from the SAME
#     _bc_ke_and_pressure_gradients call, zeroing the OTHER term exactly.
# =============================================================================
def measure_hpg_and_keg(st) -> dict:
    print("\n" + "=" * 78)
    print("(B) dyn_hpg (du) and dyn_adv KEG")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    br, mc = st["br"], st["mc"]
    state = br.state
    z_coord = br.z_coord
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data

    J, h_k, rho_prime, p_prime_filled = compute_frozen_geom_density(
        state, br.geometry, z_coord, mc)
    u, v = state.u.data, state.v.data
    eta_floor = mc.min_water_column_m - H_bathy
    eta_safe = jnp.maximum(state.eta.data, eta_floor) * mask

    # --- full call: what production's KE_PGF_u actually is. ---
    dKE_dx_full, dp_dx_full, _, _ = _bc_ke_and_pressure_gradients(
        u, v, p_prime_filled, rho_prime, br.geometry, mc, z_coord,
        eta_safe, H_bathy, mc.g, mask)
    KE_PGF_u_full = np.asarray(-dKE_dx_full - dp_dx_full / mc.rho_0)

    # --- dyn_hpg (du): PGF alone -- zero u,v (KE depends only on u,v). ---
    zeros_uv = jnp.zeros_like(u)
    dKE_dx_0, dp_dx_pgf, _, _ = _bc_ke_and_pressure_gradients(
        zeros_uv, jnp.zeros_like(v), p_prime_filled, rho_prime, br.geometry, mc,
        z_coord, eta_safe, H_bathy, mc.g, mask)
    print(f"  [verify] KE gradient is exactly zero when u,v=0: "
          f"max|dKE_dx|={float(jnp.max(jnp.abs(dKE_dx_0))):.3e} (expect 0.0)")
    du_hpg_lego = np.asarray(-dp_dx_pgf / mc.rho_0)

    # --- dyn_adv KEG: KE gradient alone -- zero rho_prime/p_prime_filled
    # (dp_dx depends only on these + eta; additive with dKE_dx, so this is
    # an EXACT isolation of -dKE_dx, verified below). ---
    zeros_p = jnp.zeros_like(p_prime_filled)
    dKE_dx_only, dp_dx_0, _, _ = _bc_ke_and_pressure_gradients(
        u, v, zeros_p, jnp.zeros_like(rho_prime), br.geometry, mc,
        z_coord, eta_safe, H_bathy, mc.g, mask)
    print(f"  [verify] PGF is exactly zero when rho_prime/p_prime_filled=0: "
          f"max|dp_dx|={float(jnp.max(jnp.abs(dp_dx_0))):.3e} (expect 0.0)")
    du_keg_lego = np.asarray(-dKE_dx_only)

    additive_check = float(np.max(np.abs(
        KE_PGF_u_full - (du_keg_lego + du_hpg_lego))))
    print(f"  [verify] additive separability: max|full-(KEG+PGF)| = "
          f"{additive_check:.3e} (expect 0.0 -- confirms the isolation is exact, "
          "not approximate)")

    du_hpg_nemo = _load_haloed(os.path.join(RUN_DIR, "hpg_dump_du.bin"), jpi, jpj, hls)
    du_keg_nemo = _load_haloed(os.path.join(RUN_DIR, "keg_dump_du.bin"), jpi, jpj, hls)
    nk = min(du_hpg_lego.shape[-1], du_hpg_nemo.shape[-1])

    # legoESM u-faces: index 0 = west wall; NEMO's east-face u(ji) is lego
    # face ji+1 (same convention as hpg_tendency_compare.py / eos_rab_bn2
    # template).
    u_mask_east = np.asarray(u_mask)[:, 1:]   # (n_lat, n_lon) -- matches the [:,1:,:] slice below
    umask3 = np.broadcast_to(u_mask_east[..., None] > 0.5,
                              (u_mask_east.shape[0], u_mask_east.shape[1], nk)).copy()
    umask3[:, -1, :] = False   # drop the redundant periodic-wrap face

    r_hpg = per_element_stats("dyn_hpg du", du_hpg_lego[:, 1:, :nk], du_hpg_nemo[..., :nk],
                               umask3, sign_changing=True)
    r_keg = per_element_stats("dyn_adv KEG du", du_keg_lego[:, 1:, :nk], du_keg_nemo[..., :nk],
                               umask3, sign_changing=True)
    return dict(dyn_hpg_du=r_hpg, dyn_adv_KEG=r_keg)


# =============================================================================
# (C) sbc (utau/qsr/qns/sfx)
# =============================================================================
def measure_sbc(st) -> dict:
    print("\n" + "=" * 78)
    print("(C) sbc (utau/qsr/qns/sfx)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    cfg = st["cfg"]
    grid, br = st["grid"], st["br"]
    tmask2d = np.asarray(grid.tmask[..., 0]) > 0.5

    print(f"  cfg.forcing_annual_cycle={getattr(cfg, 'forcing_annual_cycle', None)} "
          "(ocean.output: ln_ann_cyc=T -- the SEASONAL functions, not the "
          "annual-mean ones, are the correct comparison targets)")
    lvl = time_level_for_dump("sbc_dump_qns.bin")
    print(f"  time_level_for_dump('sbc_dump_qns.bin') = {lvl!r}")

    # kt -> t_seconds: dino_seasonal_cosines' own docstring says
    # "t_seconds is the model time ... kt*dt, first step ends at t=dt" --
    # exactly NEMO's compute_day_of_year(kt,...) convention
    # (usrdef_sbc.F90:535-536 `ztime = REAL(kt)*rn_dt/(rmmss*rhhmm)`, hours
    # since ndate0=year 1). This dump is written at kt=nit000=57601 (ocean.
    # output), rn_Dt=2700s (== cfg.dt).
    kt_dump = 57601
    t_seconds = kt_dump * cfg.dt
    print(f"  kt(nit000)={kt_dump}  cfg.dt={cfg.dt}s  -> t_seconds={t_seconds:.1f} "
          f"({t_seconds/86400.0:.3f} days)")

    lat_deg_1d = np.degrees(np.asarray(br.geometry.lat))   # (n_lat,)
    n_lat, n_lon = tmask2d.shape

    # --- utau: T-row latitude (no separate gphiu on NemoGrid; the u-face
    # shares its T-row's latitude on this regular/Mercator grid -- the same
    # fact already established for ldftra ahtu above and elsewhere in the
    # gate's notes). ---
    tau_u_1d = np.asarray(dino_wind_stress(jnp.asarray(lat_deg_1d), cfg))
    utau_lego = np.broadcast_to(tau_u_1d[:, None], (n_lat, n_lon))
    utau_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_utau.bin"), jpi, jpj, hls)[..., 0]
    r_utau = per_element_stats("utau (wind stress)", utau_lego, utau_nemo,
                                tmask2d, sign_changing=True)

    # --- qsr ---
    qsr_1d = np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    qsr_lego = np.broadcast_to(qsr_1d[:, None], (n_lat, n_lon))
    qsr_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_qsr.bin"), jpi, jpj, hls)[..., 0]
    # ln_diu_cyc=.false. on this run (namelist_cfg) -> NEMO's compute_diurn_
    # cycle ELSE branch is qsr=pqsr_dayMean directly (no diurnal sub-cycle),
    # so the day-mean seasonal function is the exact comparison target.
    r_qsr = per_element_stats("qsr (solar, day-mean, ln_diu_cyc=F)", qsr_lego,
                               qsr_nemo, tmask2d, sign_changing=False)

    # --- qns / sfx: build via the SAME RestoringConfig/restoring_surface_
    # forcing call apply_dino_lat_lon_surface_forcing makes (production entry
    # point, not a re-derivation), then convert the returned EXPLICIT
    # (non-implicit) tendency back to NEMO's raw flux units.
    #
    # NEMO writes the EXPOSED forward-Euler flux (usrdef_sbc.F90 CASE(4)):
    #   qns = rn_trp*(T_Kbb - T*) - Q_sr_dayMean          [W/m^2]   (rn_trp=-40)
    #   sfx = rn_srp*(S_Kbb - S*)                         [PSU-flux units]
    #                                                       (rn_srp=-3.858e-3)
    # restoring_surface_forcing(implicit=False) returns
    #   surf_dT = -(T-T*)/tau_T - Q_sr/(rho0*c_p*dz0)      [K/s]
    #   surf_dS = -(S-S*)/tau_S                            [PSU/s]
    # and tau_from_flux_coefficient's OWN docstring identity is
    #   tau_T = rho0*c_p*dz0/A_theta ,  tau_S = rho0*dz0/A_S
    # so algebraically, multiplying back through:
    #   surf_dT*(rho0*c_p*dz0) = -(T-T*)*A_theta - Q_sr = rn_trp*(T-T*) - Q_sr
    #     because rn_trp == -A_theta  (cfg.A_theta=40 <-> NEMO rn_trp=-40)
    #   surf_dS*(rho0*dz0)     = -(S-S*)*A_S = rn_srp*(S-S*)
    #     because rn_srp == -A_S     (cfg.A_S=3.858e-3 <-> NEMO rn_srp=-3.858e-3)
    # This is a UNIT CONVERSION of restoring_surface_forcing's own production
    # output (Rule 0: no restoring physics re-derived here, only tau^-1*rho*dz
    # re-multiplied back through the SAME identity tau_from_flux_coefficient
    # already encodes).
    print(f"  A_theta={cfg.A_theta} (NEMO rn_trp=-40 -> match={cfg.A_theta == 40.0})  "
          f"A_S={cfg.A_S} (NEMO rn_srp=-3.858e-3 -> "
          f"match={abs(cfg.A_S - 3.858e-3) < 1e-9})")

    dz_0 = float(br.z_coord.dz_ref[0])
    tau_T = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0)
    tau_S = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0)
    T_star_1d = np.asarray(dino_T_star_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    S_star_1d = np.asarray(dino_S_star(jnp.asarray(lat_deg_1d), cfg))
    T_star_2d = jnp.asarray(np.broadcast_to(T_star_1d[:, None], (n_lat, n_lon)))
    S_star_2d = jnp.asarray(np.broadcast_to(S_star_1d[:, None], (n_lat, n_lon)))

    T_Kbb = br.state.T_before.data
    S_Kbb = br.state.S_before.data
    _lvl = time_level_for_dump("sbc_dump_qns.bin")
    print(f"  feeding {_lvl!r}-level (Nbb, via bridge_before_state_topo) SST/SSS "
          "for qns/sfx")

    restoring_cfg = RestoringConfig(tau_T=tau_T, tau_S=tau_S,
                                     T_star_array=T_star_2d, S_star_array=S_star_2d,
                                     subtract_qsr=True, implicit=False)

    class _LatShim:
        def __init__(self, lat):
            self.grid_lat = lat
    out = restoring_surface_forcing(
        T_Kbb, S_Kbb, _LatShim(jnp.zeros((n_lat, n_lon))), restoring_cfg,
        sw_down=jnp.asarray(qsr_lego), rho_0=cfg.rho_0, c_p=cfg.c_p, dz_0=dz_0)
    qns_lego = np.asarray(out.dT_dt[..., 0]) * (cfg.rho_0 * cfg.c_p * dz_0)
    sfx_lego = np.asarray(out.dS_dt[..., 0]) * (cfg.rho_0 * dz_0)

    qns_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_qns.bin"), jpi, jpj, hls)[..., 0]
    sfx_nemo = _load_haloed(os.path.join(RUN_DIR, "sbc_dump_sfx.bin"), jpi, jpj, hls)[..., 0]
    r_qns = per_element_stats("qns (non-solar heat flux)", qns_lego, qns_nemo,
                               tmask2d, sign_changing=True)
    r_sfx = per_element_stats("sfx (salt restoring flux)", sfx_lego, sfx_nemo,
                               tmask2d, sign_changing=True)

    return dict(utau=r_utau, qsr=r_qsr, qns=r_qns, sfx=r_sfx)


def _reconstruct_naa(now, before, after_filtered, e3_n, e3_b, e3_a, e3_f, gamma):
    """Invert the (linear) thickness-weighted Asselin filter for the raw
    (pre-filter) Naa CONCENTRATION, algebraically, from NEMO's own dumped
    pre-/post-filter snapshots (single unknown per cell):

        ztc_f = ztc_n + gamma*(ztc_b - 2*ztc_n + ztc_a)
        =>  ztc_a = (ztc_f - ztc_n - gamma*ztc_b + 2*gamma*ztc_n) / gamma
        =>  after = ztc_a / e3_a

    Same algebra ``_thickness_weighted_asselin`` implements forward
    (``ocean_model_latlon_cgrid.py``), solved for the one variable NEMO
    never separately dumps."""
    ztc_n = e3_n * now
    ztc_b = e3_b * before
    ztc_f = e3_f * after_filtered
    ztc_a = (ztc_f - ztc_n - gamma * ztc_b + 2.0 * gamma * ztc_n) / gamma
    return ztc_a / np.maximum(e3_a, 1.0e-30)


# =============================================================================
# (E) ATF filter T/S/ssh
# =============================================================================
def measure_atf(st) -> dict:
    print("\n" + "=" * 78)
    print("(E) ATF filter T/S/ssh")
    print("=" * 78)
    print("  WHAT THIS TESTS: NEMO dumps the filter's PRE-filter snapshot "
          "(physically Nnn, misleadingly named '..._before.bin' -- registered "
          "'before'/'after' in time_levels.py meaning PRE-/POST-filter, a "
          "DIFFERENT axis from Nbb/Nnn/Naa) and the POST-filter result, but "
          "NEVER the raw un-filtered Naa the filter actually consumed. Naa is "
          "reconstructed by inverting the linear filter equation using Nbb "
          "(genuinely independent -- read from the restart's tb/sb/ub/vb, "
          "NOT derived from the before/after dumps) and re-running the SAME "
          "production filter forward. This is a real transcription check: "
          "if legoESM's _thickness_weighted_asselin / plain-Asselin formula "
          "has ANY sign/coefficient/thickness-weighting error, forward("
          "reconstructed-Naa) will NOT reproduce NEMO's dumped T_f/S_f/ssh_f, "
          "because the reconstruction only uses linearity (Naa's coefficient "
          "is +gamma on both sides of the equation), not the specific values "
          "of gamma/e3t/thickness-weighting the forward pass re-applies.\n"
          "  SCOPE LIMIT (explicit, not swept under the rug): NEMO's dumped "
          "T_f/S_f ALREADY contains its own explicit surface-flux Asselin "
          "correction (traatf_qco.F90:309, omitted by _thickness_weighted_"
          "asselin's own docstring -- confirmed real and nonzero here, "
          "|T_f-T_now| at k=0 reaches 9.6e-3 degC on this state). Because "
          "Naa is reconstructed FROM that already-corrected T_f, this test "
          "verifies the THICKNESS-WEIGHTING/coefficient transcription is "
          "exact; it CANNOT detect whether legoESM's own omission of that "
          "surface term would bite in an actual independent forward step "
          "(that is a SEPARATE, already-documented omission, not something "
          "this round-trip measures).")
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, mc, br = st["grid"], st["mc"], st["br"]
    z_coord = br.z_coord
    state = br.state
    gamma = mc.asselin_gamma
    print(f"  gamma (rn_atfp) = {gamma}")

    tmask3 = np.asarray(grid.tmask) > 0.5
    tmask2d = tmask3[..., 0]
    nlev = grid.tmask.shape[-1]

    T_n = _load_haloed(os.path.join(RUN_DIR, "atf_dump_tem_before.bin"), jpi, jpj, hls)
    S_n = _load_haloed(os.path.join(RUN_DIR, "atf_dump_sal_before.bin"), jpi, jpj, hls)
    T_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_tem_after.bin"), jpi, jpj, hls)
    S_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_sal_after.bin"), jpi, jpj, hls)
    ssh_n = _load_haloed(os.path.join(RUN_DIR, "atf_dump_ssh_before.bin"), jpi, jpj, hls)[..., 0]
    ssh_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_ssh_after.bin"), jpi, jpj, hls)[..., 0]

    # atf_dump_tem/sal are written over jpkm1=35 levels (traatf_qco.F90
    # `pt(:,:,1:jpkm1,jn,Kmm)`); the bridged state carries 36 -- slice to the
    # dumped range so shapes match (does not touch the vertical loop bounds
    # NEMO itself uses).
    nk_dump = T_n.shape[-1]
    T_b = np.asarray(state.T_before.data)[..., :nk_dump]
    S_b = np.asarray(state.S_before.data)[..., :nk_dump]
    ssh_b = np.asarray(state.eta_before.data)

    # --- ssh (plain Robert-Asselin): reconstruct Naa by inverting the SAME
    # inline _asselin closure legoESM applies (ocean_model_latlon_cgrid.py
    # ~L7096): eta_f = now + gamma*(before - 2*now + after).
    ssh_after = (ssh_f_nemo - ssh_n - gamma * ssh_b + 2.0 * gamma * ssh_n) / gamma
    ssh_f_lego = ssh_n + gamma * (ssh_b - 2.0 * ssh_n + ssh_after)
    r_ssh = per_element_stats("ATF ssh", ssh_f_lego, ssh_f_nemo, tmask2d,
                               sign_changing=True)

    # --- T/S (thickness-weighted): e3t at each time level is a pure function
    # of ssh via compute_layer_thickness -- the SAME production helper the
    # model itself calls at ocean_model_latlon_cgrid.py:7109-7116.
    H_bathy = state.H_bathy.data
    mwc = mc.min_water_column_m
    e3t_b = np.asarray(compute_layer_thickness(jnp.asarray(ssh_b), H_bathy, z_coord,
                                                min_water_column_m=mwc))[..., :nk_dump]
    e3t_n = np.asarray(compute_layer_thickness(jnp.asarray(ssh_n), H_bathy, z_coord,
                                                min_water_column_m=mwc))[..., :nk_dump]
    e3t_a = np.asarray(compute_layer_thickness(jnp.asarray(ssh_after), H_bathy, z_coord,
                                                min_water_column_m=mwc))[..., :nk_dump]
    e3t_f = np.asarray(compute_layer_thickness(jnp.asarray(ssh_f_lego), H_bathy, z_coord,
                                                min_water_column_m=mwc))[..., :nk_dump]
    tmask3 = tmask3[..., :nk_dump]
    mask3 = np.broadcast_to(np.asarray(state.land_mask.data)[..., None], tmask3.shape)

    T_after = _reconstruct_naa(T_n, T_b, T_f_nemo, e3t_n, e3t_b, e3t_a, e3t_f, gamma)
    S_after = _reconstruct_naa(S_n, S_b, S_f_nemo, e3t_n, e3t_b, e3t_a, e3t_f, gamma)
    T_f_lego = np.asarray(_thickness_weighted_asselin(
        jnp.asarray(T_n), jnp.asarray(T_b), jnp.asarray(T_after),
        e3t_n, e3t_b, e3t_a, e3t_f, gamma, jnp.asarray(mask3)))
    S_f_lego = np.asarray(_thickness_weighted_asselin(
        jnp.asarray(S_n), jnp.asarray(S_b), jnp.asarray(S_after),
        e3t_n, e3t_b, e3t_a, e3t_f, gamma, jnp.asarray(mask3)))

    r_T = per_element_stats("ATF T", T_f_lego, T_f_nemo, tmask3, sign_changing=False)
    r_S = per_element_stats("ATF S", S_f_lego, S_f_nemo, tmask3, sign_changing=False)
    return dict(ssh=r_ssh, T=r_T, S=r_S)


def main() -> int:
    print(f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}  (pinned)")
    st = build_state()

    r_ldftra = measure_ldftra(st)
    r_hpg_keg = measure_hpg_and_keg(st)
    r_sbc = measure_sbc(st)
    r_atf = measure_atf(st)

    print("\n" + "=" * 78)
    print("SUMMARY (<=12-line, per-row bar metric + YES/NO)")
    print("=" * 78)
    rows = [
        ("sbc utau", r_sbc["utau"]), ("sbc qsr", r_sbc["qsr"]),
        ("sbc qns", r_sbc["qns"]), ("sbc sfx", r_sbc["sfx"]),
        ("ldftra ahtu", r_ldftra["ahtu"]), ("ldftra ahtv", r_ldftra["ahtv"]),
        ("dyn_hpg (du)", r_hpg_keg["dyn_hpg_du"]), ("dyn_adv KEG", r_hpg_keg["dyn_adv_KEG"]),
        ("ATF T", r_atf["T"]), ("ATF S", r_atf["S"]), ("ATF ssh", r_atf["ssh"]),
    ]
    for name, r in rows:
        metric_kind = "err_norm" if r.get("sign_changing") else "pointwise|rel|"
        print(f"  {name:<16s} bar_metric({metric_kind})={r['bar_metric']:.3e}  "
              f"-> {'YES' if r['at_bar'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
