"""#1226 v-unification RE-TEST at each term's CORRECT NEMO time level.

TASK (human): ``zu_frc_momentum_row_reconstruction.py`` (784e57406) reported
that reconstructing NEMO's own zu_frc formula
(``SUM(e3u_0*err*umask)*r1_hu_0``, dynspg_ts.F90:335-337) from the SIGNED
per-term errors of dyn_ldf + dyn_adv ZAD + dyn_vor EEN reproduces zv_frc's
error at corr=0.985/ratio=0.977 (v), while REFUTING it for u (corr=0.044).
That reconstruction fed ALL THREE terms the SAME "now"-bridged twin state.

``dyn_zad_ldf_walk.py`` (573c7fe5e) then found dyn_ldf's DEBT err_norm
(4.49e-2 u / 2.76e-2 v) is a MEASUREMENT-HARNESS time-level mismatch, not an
operator defect: NEMO's ``dynldf_lev_rot_scheme.h90:24-25,28-29`` reads
velocity at Kbb ("before"), confirmed from
``cfgs/DINO/MY_SRC/dynldf.F90:79-83`` (``CASE(np_lap): CALL dynldf_lev_lap``)
+ ``dynldf.F90:83`` (``CALL dynldf_lev_lap(kt, Kbb, Kmm, puu, pvv, Krhs)``).
Feeding legoESM's ``state.u_before``/``v_before`` (matching legoESM's own
production Nbb dissipative pass, ``ocean_model_latlon_cgrid.py:6947-6968``)
drops err_norm from 4.4948e-2/2.7573e-2 to 4.577e-5/4.505e-5 -- roundoff.

CONCERN: the v-unification was measured with the SAME now-state convention
for ALL THREE terms. If dyn_ldf's "error" is largely a harness artifact
(now-vs-before), and zv_frc's own measurement shares that now/before
convention, corr=0.985 may be measuring the SAME ARTIFACT correlating with
itself, not a physical relationship.

THIS SCRIPT re-runs the EXACT reconstruction from
``zu_frc_momentum_row_reconstruction.py``, changed in ONE variable only: each
term is fed the velocity time level its own NEMO call site actually uses,
per the ladder established below (NOT one convention applied uniformly).

=====================================================================
PER-TERM TIME LEVEL (established from NEMO source, cited file:line)
=====================================================================

dyn_ldf (dynldf_lev_rot_scheme.h90:21-52): velocity Kbb ("before").
  Call chain: cfgs/DINO/MY_SRC/dynldf.F90:79-83 (CASE(np_lap): CALL
  dynldf_lev_lap) -> dynldf.F90:83 (CALL dynldf_lev_lap(kt, Kbb, Kmm, puu,
  pvv, Krhs)) -> dynldf_lev_rot_scheme.h90:24-25,28-29 hardcode
  pu(...,jk,Kbb)/pv(...,jk,Kbb) for BOTH the curl and div inputs. Established
  by dyn_zad_ldf_walk.py (573c7fe5e), reused here, NOT re-derived.

dyn_adv ZAD (dynzad.F90:81-119): velocity Kmm ("now").
  Call site dynadv.F90:97: CALL dyn_zad(kt, Kmm, puu, pvv, Krhs) -- only ONE
  velocity-time-level argument exists at all (Kbb is not even in dyn_zad's
  signature, dynzad.F90:56). ww (vertical velocity) is also read at Kmm
  (dynzad.F90:93-94, "now" ww). Established by dyn_zad_ldf_walk.py: feeding
  "before" makes ZAD WORSE (u 3.999e-2->4.218e-2, v 5.075e-2->5.910e-2),
  confirming legoESM's now-state is ALREADY correct for this term -- so this
  term's correct-time-level err_norm is UNCHANGED from the original
  reconstruction (it already used the right level).

dyn_vor EEN (dynvor.F90:672-815 vor_een): velocity Kmm ("now").
  Call chain: stpmlf.F90:249 (CALL dyn_vor(kstp, Nnn, uu, vv, Nrhs) --
  Nnn=Kmm) -> dynvor.F90:138 (CASE(np_EEN): CALL vor_een(kt, Kmm, ncor,
  puu(:,:,:,Kmm), pvv(:,:,:,Kmm), puu(:,:,:,Krhs), pvv(:,:,:,Krhs))).
  Read vor_een's body directly (dynvor.F90:672-815): EVERY e3t/e3u/e3v
  reference inside also carries the SAME Kmm argument (lines ~725-741,
  transcribed below) -- there is no Kbb anywhere in this routine, unlike
  dyn_ldf which mixes Kbb (curl/div inputs) with Kmm (outer divisor). So
  EEN is uniformly "now" -- same level the original reconstruction already
  used. This DIFFERS from dyn_ldf (Kbb) and confirms the task's warning:
  the three terms do NOT share one convention.

Net: of the three momentum rows, ONLY dyn_ldf's correct level differs from
what the original reconstruction fed it. dyn_adv ZAD and dyn_vor EEN were
ALREADY at their correct ("now") level. So this retest changes ONLY
dyn_ldf's input state (now -> before) and recomputes the SAME reconstruction
formula on the SAME twin restart/state/config -- a controlled one-variable
change, per the task's rule.

zu_frc/zv_frc's OWN measurement (Part 4 of the task): the dumped zu_frc/
zv_frc (spg_dump_zu_frc.bin/spg_dump_zv_frc.bin) is an ACCUMULATED Krhs
bookkeeping quantity, not a single leapfrog-level field: dynspg_ts.F90:336-
337 forms it from puu(:,:,:,Krhs) (the accumulated RHS after dyn_ldf +
dyn_adv + dyn_vor + dyn_hpg + dyn_cor_2d have all added into Krhs earlier in
stpmlf.F90's ordered call sequence), then subtracts the 2-D Coriolis
zu_trd (dynspg_ts.F90:364, built from puu_b(:,:,Kmm)/pvv_b(:,:,Kmm) -- an
already-depth-averaged 2-D field with its own "now" convention), then adds
dyn_drg_init's bottom-drag contribution (:381-397 in the MY_SRC-numbered
file, captured just before the spg_dump_zu_frc.bin dump). It is registered
"before" in time_levels.py in the sense of "matches the pre-substep-loop
snapshot" (a Krhs bookkeeping label, exactly as the existing ldf_dump_du.bin/
zad_dump_du.bin/vor_dump_du.bin dumps are registered "now" for the SAME
reason -- see zu_frc_term_walk.py's own comments), NOT a claim that a single
prognostic field's leapfrog level was mis-selected. legoESM's own zu_frc/
zv_frc measurement (``F_slow_u``/``F_slow_v`` captured via the
``barotropic_substeps_latlon_cgrid`` spy, unchanged from the original script)
is likewise an accumulated-tendency capture at a fixed pipeline point, not a
single-field time-level read -- so there is no THIRD Kbb/Kmm choice hiding in
that comparison the way there was for dyn_ldf. Reported explicitly below,
not assumed.

Reuses ``zu_frc_term_walk.py``'s loaders/RUN_DIR/DT (imported, not
copy-pasted) and ``zu_frc_momentum_row_reconstruction.py``'s reconstruction
formula (re-derived here as the identical algebraic expression operating on
per-term-selected states -- the loaders/metric/mask machinery is imported,
the reconstruction loop is necessarily rewritten because now three
DIFFERENT states must be threaded through instead of one).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/v_unification_timelevel_retest.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_before_state_topo,
    bridge_nemo_to_legoesm_topo,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# --- Reuse WHOLESALE (not re-derived): the same loaders/RUN_DIR/DT the prior
# two scripts built (pre-impl grep found this file first, per project rule).
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior, _load_full, _load_full_3d,
)

register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:336-337 Krhs depth-mean, then :364-368 zu_trd "
              "subtraction, then dyn_drg_init bottom-drag add -- an "
              "accumulated Krhs bookkeeping quantity, registered 'before' "
              "for its pre-substep-loop snapshot point (see this module's "
              "docstring 'zu_frc/zv_frc's OWN measurement' section).")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
register_dump("ldf_dump_du.bin", "now",
              "dump-time-level label per zu_frc_term_walk.py convention "
              "(the Krhs INCREMENT dump); the INPUT VELOCITY level is Kbb "
              "-- see dyn_zad_ldf_walk.py + this module's docstring.")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")
register_dump("zad_dump_du.bin", "now",
              "dynadv.F90:97 CALL dyn_zad(kt, Kmm, puu, pvv, Krhs) -- "
              "velocity input Kmm ('now'), confirmed correct by "
              "dyn_zad_ldf_walk.py's before/now A/B (before is WORSE).")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")
register_dump("vor_dump_du.bin", "now",
              "stpmlf.F90:249 CALL dyn_vor(kstp, Nnn, uu, vv, Nrhs) -> "
              "dynvor.F90:138 CASE(np_EEN): CALL vor_een(kt, Kmm, ncor, "
              "puu(:,:,:,Kmm), pvv(:,:,:,Kmm), ...) -- velocity Kmm ('now'), "
              "and vor_een's OWN e3t/e3u/e3v reads (dynvor.F90 body) are ALSO "
              "uniformly Kmm (no Kbb anywhere in vor_een, unlike dyn_ldf).")
register_dump("vor_dump_dv.bin", "now", "same as vor_dump_du")
register_dump("cor2d_dump_zu_trd_substep1.bin", "now",
              "dynspg_ts.F90:760,778 zu_trd inside the substep loop at "
              "jn=1 -- 2-D, already depth-mean; same registration as the "
              "original reconstruction script (unchanged, not part of this "
              "retest's variable).")
register_dump("cor2d_dump_zv_trd_substep1.bin", "now", "same as zu_trd substep1")

for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin",
    "ldf_dump_du.bin", "ldf_dump_dv.bin",
    "zad_dump_du.bin", "zad_dump_dv.bin",
    "vor_dump_du.bin", "vor_dump_dv.bin",
    "cor2d_dump_zu_trd_substep1.bin", "cor2d_dump_zv_trd_substep1.bin",
):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _per_level_err(lego_3d, nemo_3d, mask2d, n_lev_common):
    """Same construction as zu_frc_momentum_row_reconstruction.py's
    ``_per_level_err`` (reproduced, not imported, since that module is
    READ-ONLY per the task's file list) -- identical body, verified by the
    self-check below reproducing the ORIGINAL now-state numbers first."""
    n_lat_c = min(lego_3d.shape[0], nemo_3d.shape[0], mask2d.shape[0])
    n_lon_c = min(lego_3d.shape[1], nemo_3d.shape[1], mask2d.shape[1])
    n_lev_c = min(lego_3d.shape[2], nemo_3d.shape[2], n_lev_common)
    lo = lego_3d[:n_lat_c, :n_lon_c, :n_lev_c]
    ne = nemo_3d[:n_lat_c, :n_lon_c, :n_lev_c]
    m = mask2d[:n_lat_c, :n_lon_c]
    err = lo - ne
    err_by_level = np.array([
        float(np.sqrt(np.nanmean(err[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    rms_nemo_by_level = np.array([
        float(np.sqrt(np.nanmean(ne[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    return err, err_by_level, rms_nemo_by_level, m


def _reconstruct_depth_mean_err(err3_terms, e3_0, h_0, n_lat_c, n_lon_c, n_lev_c):
    """SUM(e3u_0*err*umask)*r1_hu_0 -- NEMO's own zu_frc/zv_frc formula
    (dynspg_ts.F90:335-337), on the SIGNED sum of the per-term error fields.
    Identical algebra to zu_frc_momentum_row_reconstruction.py's function of
    the same purpose (the one-variable change is which STATE fed each err3,
    upstream of this function -- the formula itself is untouched)."""
    e3_c = e3_0[:n_lat_c, :n_lon_c, :n_lev_c] if e3_0.ndim == 3 else \
        np.broadcast_to(e3_0[None, None, :n_lev_c], (n_lat_c, n_lon_c, n_lev_c))
    h_c = h_0[:n_lat_c, :n_lon_c]
    total_err3 = sum(err3_terms)
    depth_sum = np.sum(e3_c * total_err3, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recon = np.where(h_c > 0, depth_sum / np.where(h_c > 0, h_c, 1.0), 0.0)
    return recon


def _corr_ratio(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
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
    e3t_mode = require_explicit_e3t_mode(context="v_unification_timelevel_retest")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vorticity_scheme={dcfg.vorticity_scheme!r}  "
          f"lateral_viscosity_operator={dcfg.lateral_viscosity_operator!r}  "
          f"vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vorticity_scheme == "een_total"
    assert dcfg.lateral_viscosity_operator == "nemo_div_curl"
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    st_now = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st_now, context="v_unification_timelevel_retest")
    print("dtype check: u", st_now.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype)

    # --- The BEFORE-fed twin state, for dyn_ldf's correct time level (Kbb).
    # Same substitution dyn_zad_ldf_walk.py used, mirroring
    # ocean_model_latlon_cgrid.py:6959-6961's production Nbb pass.
    st_before_fed = st_now._replace(
        u=st_now.u_before, v=st_now.v_before,
        T=st_now.T_before, S=st_now.S_before)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Capture zu_frc/zv_frc (F_slow_u/F_slow_v post zu_trd subtraction)
    # from a single production step on the NOW state -- unchanged from the
    # original reconstruction (zu_frc/zv_frc's own measurement convention is
    # NOT part of this retest's variable; see docstring).
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "F_slow_u" not in captured:
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    try:
        with jax.disable_jit():
            _ = model.step(st_now, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
    assert "F_slow_u" in captured, "barotropic solver never called with a seed"
    F_slow_u_actual = np.asarray(captured["F_slow_u"])
    F_slow_v_actual = np.asarray(captured["F_slow_v"])

    # --- Diagnostics from BOTH twin states (NOW for zad/vor, BEFORE for ldf).
    with jax.disable_jit():
        _tend_now, diag_now = model.tendencies_with_diagnostics(st_now, surface_forcing=sf, dt=DT)
        _tend_bef, diag_bef = model.tendencies_with_diagnostics(st_before_fed, surface_forcing=sf, dt=DT)

    ah_lap_u_now = np.asarray(diag_now.Ah_lap_u.data)
    ah_lap_v_now = np.asarray(diag_now.Ah_lap_v.data)
    ah_lap_u_bef = np.asarray(diag_bef.Ah_lap_u.data)
    ah_lap_v_bef = np.asarray(diag_bef.Ah_lap_v.data)
    vertadv_u_3d = np.asarray(diag_now.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag_now.vertadv_v.data)
    vortcor_u_3d = np.asarray(diag_now.vortcor_u.data)
    vortcor_v_3d = np.asarray(diag_now.vortcor_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    e3u_0 = np.asarray(g.e3u_0) if hasattr(g, "e3u_0") else None
    e3v_0 = np.asarray(g.e3v_0) if hasattr(g, "e3v_0") else None
    hu_0 = np.asarray(g.hu_0) if hasattr(g, "hu_0") else None
    hv_0 = np.asarray(g.hv_0) if hasattr(g, "hv_0") else None
    if e3u_0 is None or hu_0 is None:
        raise RuntimeError(
            "mesh_mask read did not expose e3u_0/hu_0 -- cannot reconstruct "
            "zu_frc the way NEMO forms it; STOP rather than approximate.")

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    nemo_ldf_du = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_dv = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_vor_du = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_vor_dv = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)
    nemo_zu_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zu_trd_substep1.bin"), jpi, jpj, hls)
    nemo_zv_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zv_trd_substep1.bin"), jpi, jpj, hls)

    ah_lap_u_now_f = _u_to_nemo(ah_lap_u_now)
    ah_lap_v_now_f = _v_to_nemo(ah_lap_v_now)
    ah_lap_u_bef_f = _u_to_nemo(ah_lap_u_bef)
    ah_lap_v_bef_f = _v_to_nemo(ah_lap_v_bef)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vertadv_v_f = _v_to_nemo(vertadv_v_3d)
    vortcor_u_f = _u_to_nemo(vortcor_u_3d)
    vortcor_v_f = _v_to_nemo(vortcor_v_3d)

    n_lev_common = min(nemo_ldf_du.shape[2], nemo_zad_du.shape[2], nemo_vor_du.shape[2],
                        ah_lap_u_now_f.shape[2], vertadv_u_f.shape[2], vortcor_u_f.shape[2])

    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce the ORIGINAL now-state numbers (784e57406)")
    print("BEFORE changing anything -- if this does not reproduce, STOP (Rule 1e).")
    print("=" * 78)
    err_ldf_u_now, ebl_ldf_u_now, rbl_ldf_u_now, _ = _per_level_err(ah_lap_u_now_f, nemo_ldf_du, umask2, n_lev_common)
    err_ldf_v_now, ebl_ldf_v_now, rbl_ldf_v_now, _ = _per_level_err(ah_lap_v_now_f, nemo_ldf_dv, vmask2, n_lev_common)
    tot_ldf_u_now = float(np.sqrt(np.mean(ebl_ldf_u_now ** 2))) / float(np.sqrt(np.mean(rbl_ldf_u_now ** 2)))
    tot_ldf_v_now = float(np.sqrt(np.mean(ebl_ldf_v_now ** 2))) / float(np.sqrt(np.mean(rbl_ldf_v_now ** 2)))
    print(f"  dyn_ldf (now-fed, should match dyn_zad_ldf_walk.py's 4.4948e-02/2.7573e-02):")
    print(f"    u err_norm={tot_ldf_u_now:.4e}  v err_norm={tot_ldf_v_now:.4e}")
    self_check_1_ok = abs(tot_ldf_u_now - 4.4948e-02) / 4.4948e-02 < 0.02 and \
        abs(tot_ldf_v_now - 2.7573e-02) / 2.7573e-02 < 0.02
    print(f"    self-check-1 (now-fed reproduces prior dyn_ldf number, <2% tol): {self_check_1_ok}")
    if not self_check_1_ok:
        print("  *** SELF-CHECK 1 FAILED -- reconcile before trusting anything below ***")

    print("\n" + "=" * 78)
    print("SELF-CHECK 2: reproduce the ORIGINAL v=0.985/0.977, u=0.044/0.045")
    print("reconstruction EXACTLY (all three terms now-fed, as 784e57406 did).")
    print("=" * 78)
    rows_u_orig = {}
    rows_v_orig = {}
    for label, lego3, nemo3, mask2, side in (
        ("dyn_ldf", ah_lap_u_now_f, nemo_ldf_du, umask2, "u"),
        ("dyn_adv ZAD", vertadv_u_f, nemo_zad_du, umask2, "u"),
        ("dyn_vor EEN", vortcor_u_f, nemo_vor_du, umask2, "u"),
        ("dyn_ldf", ah_lap_v_now_f, nemo_ldf_dv, vmask2, "v"),
        ("dyn_adv ZAD", vertadv_v_f, nemo_zad_dv, vmask2, "v"),
        ("dyn_vor EEN", vortcor_v_f, nemo_vor_dv, vmask2, "v"),
    ):
        err3, _, _, mask_c = _per_level_err(lego3, nemo3, mask2, n_lev_common)
        d = rows_u_orig if side == "u" else rows_v_orig
        d[label] = err3

    e3u_c = e3u_0[..., :n_lev_common] if e3u_0.ndim == 3 else e3u_0[:n_lev_common]
    e3v_c = e3v_0[..., :n_lev_common] if (e3v_0 is not None and e3v_0.ndim == 3) else e3v_0

    n_lat_u, n_lon_u, _ = rows_u_orig["dyn_ldf"].shape
    n_lat_v, n_lon_v, _ = rows_v_orig["dyn_ldf"].shape
    recon_u_orig = _reconstruct_depth_mean_err(
        list(rows_u_orig.values()), e3u_c, hu_0, n_lat_u, n_lon_u, n_lev_common)
    recon_v_orig = _reconstruct_depth_mean_err(
        list(rows_v_orig.values()), e3v_c, hv_0, n_lat_v, n_lon_v, n_lev_common)

    zu_frc_err = _u_to_nemo(F_slow_u_actual) - nemo_zu_frc
    zv_frc_err = _v_to_nemo(F_slow_v_actual) - nemo_zv_frc

    def _corr_ratio_masked(recon, err_field, mask2):
        n_lat_z = min(recon.shape[0], err_field.shape[0])
        n_lon_z = min(recon.shape[1], err_field.shape[1])
        m = mask2[:n_lat_z, :n_lon_z]
        r = recon[:n_lat_z, :n_lon_z][m]
        e = err_field[:n_lat_z, :n_lon_z][m]
        return _corr_ratio(r, e)

    corr_u_orig, ratio_u_orig = _corr_ratio_masked(recon_u_orig, zu_frc_err, umask2)
    corr_v_orig, ratio_v_orig = _corr_ratio_masked(recon_v_orig, zv_frc_err, vmask2)
    print(f"  u (all now-fed): corr={corr_u_orig:.4f}  ratio={ratio_u_orig:.4f}  "
          f"(want ~0.044/0.045)")
    print(f"  v (all now-fed): corr={corr_v_orig:.4f}  ratio={ratio_v_orig:.4f}  "
          f"(want ~0.985/0.977)")
    self_check_2_ok = (abs(corr_u_orig - 0.044) < 0.02 and abs(corr_v_orig - 0.985) < 0.02)
    print(f"    self-check-2 (reproduces prior reconstruction, <0.02 abs tol on corr): "
          f"{self_check_2_ok}")
    if not self_check_2_ok:
        print("  *** SELF-CHECK 2 FAILED -- reconcile before trusting anything below ***")

    print("\n" + "=" * 78)
    print("PART 1: per-term err_norm at EACH TERM'S CORRECT time level")
    print("(dyn_ldf: before/Kbb :: dyn_adv ZAD, dyn_vor EEN: now/Kmm -- unchanged")
    print("for the latter two, since 'now' was already their correct level)")
    print("=" * 78)
    err_ldf_u_bef, ebl_ldf_u_bef, rbl_ldf_u_bef, _ = _per_level_err(ah_lap_u_bef_f, nemo_ldf_du, umask2, n_lev_common)
    err_ldf_v_bef, ebl_ldf_v_bef, rbl_ldf_v_bef, _ = _per_level_err(ah_lap_v_bef_f, nemo_ldf_dv, vmask2, n_lev_common)
    tot_ldf_u_bef = float(np.sqrt(np.mean(ebl_ldf_u_bef ** 2))) / float(np.sqrt(np.mean(rbl_ldf_u_bef ** 2)))
    tot_ldf_v_bef = float(np.sqrt(np.mean(ebl_ldf_v_bef ** 2))) / float(np.sqrt(np.mean(rbl_ldf_v_bef ** 2)))
    print(f"  dyn_ldf   (CORRECT level=before): u err_norm={tot_ldf_u_bef:.4e}  "
          f"v err_norm={tot_ldf_v_bef:.4e}   (was now-fed: u={tot_ldf_u_now:.4e} v={tot_ldf_v_now:.4e})")

    err_zad_u, ebl_zad_u, rbl_zad_u, _ = _per_level_err(vertadv_u_f, nemo_zad_du, umask2, n_lev_common)
    err_zad_v, ebl_zad_v, rbl_zad_v, _ = _per_level_err(vertadv_v_f, nemo_zad_dv, vmask2, n_lev_common)
    tot_zad_u = float(np.sqrt(np.mean(ebl_zad_u ** 2))) / float(np.sqrt(np.mean(rbl_zad_u ** 2)))
    tot_zad_v = float(np.sqrt(np.mean(ebl_zad_v ** 2))) / float(np.sqrt(np.mean(rbl_zad_v ** 2)))
    print(f"  dyn_adv ZAD (CORRECT level=now, unchanged): u err_norm={tot_zad_u:.4e}  "
          f"v err_norm={tot_zad_v:.4e}")

    err_vor_u, ebl_vor_u, rbl_vor_u, _ = _per_level_err(vortcor_u_f, nemo_vor_du, umask2, n_lev_common)
    err_vor_v, ebl_vor_v, rbl_vor_v, _ = _per_level_err(vortcor_v_f, nemo_vor_dv, vmask2, n_lev_common)
    tot_vor_u = float(np.sqrt(np.mean(ebl_vor_u ** 2))) / float(np.sqrt(np.mean(rbl_vor_u ** 2)))
    tot_vor_v = float(np.sqrt(np.mean(ebl_vor_v ** 2))) / float(np.sqrt(np.mean(rbl_vor_v ** 2)))
    print(f"  dyn_vor EEN (CORRECT level=now, unchanged): u err_norm={tot_vor_u:.4e}  "
          f"v err_norm={tot_vor_v:.4e}")

    print("\n" + "=" * 78)
    print("PART 2: THE DECISIVE RECONSTRUCTION at CORRECTED time levels")
    print("SUM(e3u_0 * err * umask) * r1_hu_0, summing dyn_ldf(before) +")
    print("dyn_adv_ZAD(now) + dyn_vor_EEN(now)  (dynspg_ts.F90:335-337 formula)")
    print("=" * 78)
    rows_u_fixed = {"dyn_ldf": err_ldf_u_bef, "dyn_adv ZAD": err_zad_u, "dyn_vor EEN": err_vor_u}
    rows_v_fixed = {"dyn_ldf": err_ldf_v_bef, "dyn_adv ZAD": err_zad_v, "dyn_vor EEN": err_vor_v}

    recon_u_fixed = _reconstruct_depth_mean_err(
        list(rows_u_fixed.values()), e3u_c, hu_0, n_lat_u, n_lon_u, n_lev_common)
    recon_v_fixed = _reconstruct_depth_mean_err(
        list(rows_v_fixed.values()), e3v_c, hv_0, n_lat_v, n_lon_v, n_lev_common)

    corr_u_fixed, ratio_u_fixed = _corr_ratio_masked(recon_u_fixed, zu_frc_err, umask2)
    corr_v_fixed, ratio_v_fixed = _corr_ratio_masked(recon_v_fixed, zv_frc_err, vmask2)

    print(f"  u (dyn_ldf now BEFORE-fed, ZAD/EEN unchanged): corr={corr_u_fixed:.4f}  "
          f"ratio={ratio_u_fixed:.4f}   (control -- was already refuted at now-level: "
          f"corr={corr_u_orig:.4f})")
    print(f"  v (dyn_ldf now BEFORE-fed, ZAD/EEN unchanged): corr={corr_v_fixed:.4f}  "
          f"ratio={ratio_v_fixed:.4f}   (compare to now-level unification: "
          f"corr={corr_v_orig:.4f} ratio={ratio_v_orig:.4f})")

    print("\n" + "=" * 78)
    print("PART 3: is zu_frc/zv_frc's OWN measurement itself at a mismatched level?")
    print("(Krhs bookkeeping quantity -- see docstring; no second Kbb/Kmm choice")
    print("was found for THIS comparison the way there was for dyn_ldf's velocity")
    print("input. Reporting the fact, not assuming it.)")
    print("=" * 78)
    print("  zu_frc/zv_frc dump: dynspg_ts.F90:336-337 SUM(e3u(Kmm)*puu(Krhs)*umask)")
    print("  *r1_hu(Kmm) [linssh/qco variant: e3u_0 static ref, r1_hu_0 static ref,")
    print("  DINO's actual #else-branch per key_qco -- confirmed by grep above],")
    print("  Krhs is the ACCUMULATED RHS after dyn_hpg+dyn_ldf+dyn_adv+dyn_vor all")
    print("  wrote into it earlier in stpmlf.F90's ordered sequence -- there is no")
    print("  single 'time level' to mis-select for an accumulator; the dump is a")
    print("  pipeline-position snapshot (post zu_trd subtraction, post dyn_drg_init,")
    print("  pre atm-pressure/wind-forcing additions), matched on legoESM's side by")
    print("  capturing F_slow_u/F_slow_v at the SAME pipeline position (the")
    print("  barotropic_substeps_latlon_cgrid call boundary) -- unchanged from the")
    print("  original script, NOT a variable in this retest.")

    print("\n" + "=" * 78)
    print("SUMMARY (raw numbers only -- interpretation belongs in the report, not here)")
    print("=" * 78)
    print(f"self_check_1_ok={self_check_1_ok}  self_check_2_ok={self_check_2_ok}")
    print(f"dyn_ldf err_norm: now u={tot_ldf_u_now:.4e} v={tot_ldf_v_now:.4e}  "
          f"| before(correct) u={tot_ldf_u_bef:.4e} v={tot_ldf_v_bef:.4e}")
    print(f"dyn_adv_ZAD err_norm (correct=now, unchanged): u={tot_zad_u:.4e} v={tot_zad_v:.4e}")
    print(f"dyn_vor_EEN err_norm (correct=now, unchanged): u={tot_vor_u:.4e} v={tot_vor_v:.4e}")
    print(f"reconstruction ORIGINAL (all now-fed): u corr={corr_u_orig:.4f} ratio={ratio_u_orig:.4f}  "
          f"v corr={corr_v_orig:.4f} ratio={ratio_v_orig:.4f}")
    print(f"reconstruction CORRECTED (ldf before-fed): u corr={corr_u_fixed:.4f} ratio={ratio_u_fixed:.4f}  "
          f"v corr={corr_v_fixed:.4f} ratio={ratio_v_fixed:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
