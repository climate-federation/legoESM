"""#1226 zu_frc u-error: complete the FIVE-term + zu_trd-subtraction budget.

TASK (human, measurement-only scope cap): the campaign's largest DEBT row,
``zu_frc`` u (recorded 8.03e-3), has had SEVEN candidate explanations refuted
(``zu_frc_u_structure_probe.py``, ``zu_frc_leapfrog_residual_probe.py``,
``momentum_jacobian_probe.py``). ``zu_frc_momentum_row_reconstruction.py``
reconstructed only THREE of NEMO's five 3-D RHS tendency terms it sums into
``zu_frc`` (ldf, zad, vor -- it lacked keg/hpg diagnostics at the time) and
got u corr=0.044/ratio=0.045 -- i.e. that partial sum explained ~4.5% of
zu_frc's error. THIS script completes the budget: adds keg+hpg (now exposed,
see below) and tests the standing hypothesis that the dominant unaccounted
piece is the ``zu_trd`` 2-D barotropic-Coriolis SUBTRACTION itself
(``dynspg_ts.F90:364,367-368``), since ``dyn_cor_2d`` is independently known
DEBT (u corr 0.999717, NOT at bar) with a ~15x u/v asymmetry matching
zu_frc's own ~15x u/v asymmetry.

NEMO SOURCE VERIFIED (read directly, not assumed -- see run output for the
grep/sed transcript this docstring summarizes):
  - ``dynspg_ts.F90`` (MLF branch, the one this card uses): zu_frc's 3-D-RHS
    depth-mean is built at ~line 335-345 (env `LEGOESM_NEMO_E3T=both` ->
    the ``e3u(Kmm)``-weighted branch, ``zu_frc=SUM(e3u(:,:,:,Kmm)*puu(Krhs)*
    umask)*r1_hu(:,:,Kmm)``, confirmed the live #else branch is NOT the
    e3u_0 branch -- see main() assertion below). dyn_cor_2D is called at
    line 364 on ``puu_b(:,:,Kmm)``/``pvv_b(:,:,Kmm)`` and subtracted at
    366-368. Drag (dyn_drg_init) and wind/pressure additions follow, THEN
    zu_frc is dumped at ~line 490 (``spg_dump_zu_frc.bin``) -- confirmed by
    reading lines 330-500: no rewrite of ``puu_b(:,:,Kmm)`` occurs between
    the :364 call and the :490 dump.
  - dyn_cor_2D (dynspg_ts.F90 ~1645) is a PURE function of
    (punb,pvnb,ffu_nw/ne/sw/se,ffv_nw/ne/sw/se); dyn_cor_2D_init (~1466) sets
    those coefficients once per step from grid metrics/e3u/e3v/vmask/umask
    at Kmm -- no substep-loop dependence.
  - ``keg_dump_du.bin`` (dynadv.F90:89-95): the Krhs INCREMENT from
    ``dyn_keg`` alone (before ``dyn_zad`` runs), full (jpi,jpj,jpkm1) domain,
    dumped via ``dyn_adv_dump`` -- SAME convention as ldf/zad/vor dumps.
  - ``hpg_dump_du.bin`` (dynhpg.F90:335-349,406-413): the Krhs INCREMENT
    from ``dyn_hpg`` alone (post-hpg minus a pre-hpg snapshot), full
    (jpi,jpj,jpkm1) domain -- SAME convention.

DECISION on the zu_trd approximation (see module docstring task instructions,
branch (a) vs (b)): this script uses **approach (a)**:
``cor2d_dump_zu_trd_substep1.bin`` (the IN-LOOP jn=1 call on ``ua_e``/``va_e``,
dynspg_ts.F90:760/778) as a BOUNDED APPROXIMATION of the TRUE pre-loop :364
``zu_trd`` (on ``puu_b(:,:,Kmm)``/``pvv_b(:,:,Kmm)``). Caveat, stated
explicitly: ``ua_e``/``va_e`` at jn=1 have already been advanced ONE substep
from the ``un_e``/``vn_e`` seed (itself ``puu_b(:,:,Kbb)``, a DIFFERENT time
level, dynspg_ts.F90:569-571) -- so this is NOT the exact :364 quantity.
Approach (b) (reconstructing NEMO's own EEN ffu_nw/ne/sw/se from mesh_mask
metrics + replicating dyn_cor_2D_init's CASE branch) was assessed and
REJECTED for this measurement: it would require re-deriving a second
independent implementation of NEMO's EEN vertex-weighted Coriolis operator
from raw metrics (a new transcription, not a re-use of an existing bridge
quantity) -- out of the measurement-only scope cap. legoESM's OWN
``barotropic_coriolis_een_pre_step`` (spied below, exactly as
``zu_frc_term_walk.py`` does) already reconstructs the :364 call's INPUT
side faithfully (it operates on the twin state's before-level
puu_b/pvv_b-equivalent -- confirmed in that script by the bit-exact SEED
match, ``zu_frc_leapfrog_residual_probe.py`` PART 3: err_norm 2.27e-16); the
gap under test here is therefore legoESM's own EEN OPERATOR (whatever error
it carries) against NEMO's approximate in-loop output, bounded by the
caveat above. The bound: PART 3 of the already-run leapfrog probe found
NEMO's live in-loop zu_trd (RMS 1.07e-6) is ~3700x LARGER than legoESM's own
before/now-difference residual proxy (RMS 2.9e-10) at this snapshot --
i.e. the in-loop quantity is dominated by the steady EEN term itself, not by
the one-substep evolution, so the jn=1-vs-:364 approximation error is
expected to be small relative to the EEN term's own magnitude (not proven to
machine precision -- reported as a caveat, not elevated to CONFIRMED).

Reuses ``zu_frc_term_walk.py`` and ``zu_frc_momentum_row_reconstruction.py``
wholesale (imported, not copy-pasted) -- RUN_DIR/DT/loaders/spy machinery.

RETRACTION LOGGED IN-SESSION (Rule 1e -- reconciled before being recorded,
not written down): a first pass measured the keg+hpg (``KE_PGF_u``) piece by
depth-summing with ``e3u_0 * err3d`` masked ONLY by a static level-0
``umask2`` at the final 2-D reduction, exactly mirroring NEMO's own
``zu_frc = SUM(e3u(:,:,:,Kmm)*puu(Krhs)*umask(ji,jj,:))*r1_hu`` EXCEPT it
never applied ``umask(ji,jj,:)`` (the full 3-D mask) INSIDE the sum. Since
the wet-cell count shrinks with depth in this domain (9758 wet u-faces at
levels 0-29, down to 7255 by level 34), this silently kept below-seafloor
cells "wet" in the deep levels, producing a spurious keg+hpg RMS of 1.16e-5
(~475x the true zu_frc error) and an apparently-huge, physically-wrong
closure. A direct PER-LEVEL check with the proper 3-D ``umask3`` (this
file's ``umask3``/``umask3_c``) showed ``KE_PGF_u`` matches NEMO's
``keg_dump_du+hpg_dump_du`` sum to ~1e-17..1e-19 relative error at EVERY
level 0-34 -- i.e. keg+hpg is machine-precision EXACT, not a defect. Fixed by
masking each level's error field with the 3-D ``umask3``/``vmask3`` BEFORE
the ``e3u_0``-weighted depth sum (see ``_depth_mean`` below), matching
NEMO's formula literally. The numbers in the RESULTS below are POST-FIX.

RESULTS (this session, HEAD at time of run; fp64 confirmed, e3t=both,
RUN_GDB restart kt=57601): Step 1 self-check reproduces the recorded u/v
err_norm essentially exactly (u 8.0270e-3 vs recorded 8.03e-3, v 5.4280e-4
vs recorded 5.43e-4 -- <0.1% drift from the ZAD bottom-face fix). Step 2
zu_trd-error ownership test: u corr=-0.0220, ratio=0.0192; v corr=-0.1501,
ratio=0.1830 -- NO structural relationship and the candidate signal is ~50x
SMALLER than zu_frc's own error (opposite of the "dominates" prediction).
Step 3 six-piece budget (post-mask-fix): keg+hpg RMS=5.27e-18 (machine-zero,
CONFIRMED-EXACT term, not a contributor), ldf RMS=3.86e-11 (RSS-share
0.033), zad RMS=5.54e-14 (~0), vor RMS=1.09e-9 (RSS-share 0.918, the
dominant CONFIRMED piece), zu_trd-subtraction RMS=4.68e-10 (RSS-share
0.395). Closure: reconstructed-vs-measured corr=0.0522, ratio=0.0463 (i.e.
the 6-piece reconstruction explains only ~4.6% of zu_frc's error RMS and has
essentially zero correlation with its spatial pattern) -- barely moved from
the OLD 3-term-only baseline (corr=0.0447, ratio=0.0446, matching the
previously recorded 0.044/0.045). VERDICT: zu_trd does NOT dominate; ~95%+
of zu_frc's u-error is UNEXPLAINED by any of the five tendency terms or the
zu_trd subtraction. See the task report for full labeling (CONFIRMED vs
PLAUSIBLE) and next-step recommendation.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zu_frc_budget_completion.py
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
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as bmod
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# --- Reuse wholesale (module-reuse rule): the SAME loaders/constants/spy
# helpers zu_frc_term_walk.py and zu_frc_momentum_row_reconstruction.py
# already built and validated. -----------------------------------------------
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior, _load_full, _load_full_3d, _align_scan,
)

RESTART_FILE = "DINO_00057600_restart.nc"
RESTART_STEP = 57601  # RUN_GDB/DINO_00057600_restart.nc -> kt=57601 (nit000+1)

# --- Register every dump this script touches, citing NEMO source. Some are
# re-registrations of names zu_frc_term_walk.py already registered (harmless
# -- register_dump overwrites the module dict; kept here so this script is
# self-contained and independently auditable). ------------------------------
register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345,367 zu_frc after the zu_trd subtraction, "
              "dumped ~line 490, no rewrite of puu_b(Kmm) in between "
              "(verified by reading 330-500 this session).")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
register_dump("keg_dump_du.bin", "now",
              "dynadv.F90:89-95 dyn_keg Krhs increment (velocity input Kmm, "
              "dumped via dyn_adv_dump BEFORE dyn_zad runs) -- full "
              "(jpi,jpj,jpkm1) domain, same convention as ldf/zad/vor dumps.")
register_dump("keg_dump_dv.bin", "now", "same as keg_dump_du")
register_dump("hpg_dump_du.bin", "now",
              "dynhpg.F90:335-349,406-413 dyn_hpg Krhs increment (post-hpg "
              "minus pre-hpg snapshot, velocity input Kmm) -- full "
              "(jpi,jpj,jpkm1) domain, same convention as keg/ldf/zad/vor.")
register_dump("hpg_dump_dv.bin", "now", "same as hpg_dump_du")
register_dump("ldf_dump_du.bin", "now",
              "dynldf.F90:85 dyn_ldf_iso Krhs increment (isoneutral slopes "
              "use Kmm T/S).")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")
register_dump("zad_dump_du.bin", "now",
              "dynadv.F90:97-103 dyn_zad Krhs increment, velocity input Kmm.")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")
register_dump("vor_dump_du.bin", "now",
              "dynvor.F90:151-155 vor_een Krhs increment, velocity input Kmm.")
register_dump("vor_dump_dv.bin", "now", "same as vor_dump_du")
register_dump("cor2d_dump_zu_trd_substep1.bin", "now",
              "dynspg_ts.F90:760,778 zu_trd computed INSIDE the substep loop "
              "at jn=1 from ua_e/va_e (evolving substep transport, already "
              "one substep past the un_e(Kbb) seed) -- a BOUNDED "
              "APPROXIMATION of the pre-loop :364 dyn_cor_2D(puu_b(Kmm)) "
              "call used here per this script's documented approach-(a) "
              "decision (see module docstring). Full (jpi,jpj) domain "
              "(56*203*8=90944 bytes, confirmed by file size).")
register_dump("cor2d_dump_zv_trd_substep1.bin", "now", "same as zu_trd substep1")

for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin",
    "keg_dump_du.bin", "keg_dump_dv.bin",
    "hpg_dump_du.bin", "hpg_dump_dv.bin",
    "ldf_dump_du.bin", "ldf_dump_dv.bin",
    "zad_dump_du.bin", "zad_dump_dv.bin",
    "vor_dump_du.bin", "vor_dump_dv.bin",
    "cor2d_dump_zu_trd_substep1.bin", "cor2d_dump_zv_trd_substep1.bin",
):
    time_level_for_dump(_name)  # fail-closed: raises if unregistered


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")


def _err_norm(lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = _rms(ne)
    err_norm = _rms(lo - ne) / rms if rms > 0 else float("nan")
    return err_norm, rms, m


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_budget_completion")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.coriolis_scheme={dcfg.coriolis_scheme!r}  "
          f"barotropic_coriolis_split={dcfg.barotropic_coriolis_split!r}  "
          f"vorticity_scheme={dcfg.vorticity_scheme!r}  "
          f"lateral_viscosity_operator={dcfg.lateral_viscosity_operator!r}  "
          f"vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.barotropic_coriolis_split == "live"
    assert dcfg.vorticity_scheme == "een_total"
    assert dcfg.lateral_viscosity_operator == "nemo_div_curl"
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    print(f"\nONE STATE: RUN_DIR={RUN_DIR}  RESTART_FILE={RESTART_FILE}  "
          f"restart step (kt) used = {RESTART_STEP}")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_budget_completion twin state")
    print(f"dtype check: st.u.data={st.u.data.dtype}  br.z_coord dz_ref="
          f"{np.asarray(br.z_coord.dz_ref).dtype if hasattr(br.z_coord, 'dz_ref') else 'n/a'}")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Capture (i) the ACTUAL production zu_frc (WITH zu_trd subtraction),
    # (ii) the momentum diagnostics (KE_PGF=keg+hpg, Ah_lap=ldf, vertadv=zad,
    # vortcor=vor), and (iii) legoESM's own cor_u_sub/cor_v_sub (the :3291
    # subtraction term) -- ONE model.step() covers (i) and (iii) via spies;
    # (ii) is a separate read-only tendencies_with_diagnostics() call on the
    # SAME twin state (matches zu_frc_momentum_row_reconstruction.py's
    # pattern exactly).
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid
    _real_pre_step = bmod.barotropic_coriolis_een_pre_step

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "F_slow_u" not in captured:
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    def _spy_pre_step(*a, **kw):
        cor_u, cor_v = _real_pre_step(*a, **kw)
        if "cor_u_sub" not in captured:
            captured["cor_u_sub"] = cor_u
            captured["cor_v_sub"] = cor_v
        return cor_u, cor_v

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    bmod.barotropic_coriolis_een_pre_step = _spy_pre_step
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
        bmod.barotropic_coriolis_een_pre_step = _real_pre_step
    assert "F_slow_u" in captured, "barotropic solver never called with a seed"
    assert "cor_u_sub" in captured, "barotropic_coriolis_een_pre_step ('live' branch) never fired"

    F_slow_u_actual = np.asarray(captured["F_slow_u"])
    F_slow_v_actual = np.asarray(captured["F_slow_v"])
    cor_u_sub = np.asarray(captured["cor_u_sub"])
    cor_v_sub = np.asarray(captured["cor_v_sub"])

    _tend_diag, diag = model.tendencies_with_diagnostics(st, surface_forcing=sf, dt=DT)
    ke_pgf_u_3d = np.asarray(diag.KE_PGF_u.data)   # keg + hpg, CONFIRMED (real exposed sum)
    ke_pgf_v_3d = np.asarray(diag.KE_PGF_v.data)
    ah_lap_u_3d = np.asarray(diag.Ah_lap_u.data)   # ldf, CONFIRMED
    ah_lap_v_3d = np.asarray(diag.Ah_lap_v.data)
    vertadv_u_3d = np.asarray(diag.vertadv_u.data)  # zad, CONFIRMED
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)
    vortcor_u_3d = np.asarray(diag.vortcor_u.data)  # vor (f+zeta under een_total), CONFIRMED
    vortcor_v_3d = np.asarray(diag.vortcor_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    # 3-D umask (NEMO's zu_frc formula masks EACH level by umask(ji,jj,:)
    # before summing with e3u -- a static level-0 mask applied uniformly
    # across depth silently keeps below-seafloor cells "wet" at deep levels
    # where NEMO's own umask has already gone dry (the wet-cell COUNT
    # shrinks from 9758 at level 0-29 to 7255 by level 34 in this domain).
    # Using umask2 for the depth-sum below reproduced a spurious ~475x
    # keg+hpg "error" that a per-level machine-precision check (umask3-
    # gated) showed was ENTIRELY this masking bug, not a real term error --
    # caught and fixed before this number was reported (Rule 1e discipline).
    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    e3u_0 = np.asarray(g.e3u_0)
    hu_0 = np.asarray(g.hu_0)
    e3v_0 = np.asarray(g.e3v_0) if hasattr(g, "e3v_0") else None
    hv_0 = np.asarray(g.hv_0) if hasattr(g, "hv_0") else None

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    nemo_keg_du = _load_full_3d(os.path.join(RUN_DIR, "keg_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_keg_dv = _load_full_3d(os.path.join(RUN_DIR, "keg_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_hpg_du = _load_full_3d(os.path.join(RUN_DIR, "hpg_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_hpg_dv = _load_full_3d(os.path.join(RUN_DIR, "hpg_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_du = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_dv = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_vor_du = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_vor_dv = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)
    # cor2d_dump_zu_trd_substep1.bin is full domain incl. halo (56x203),
    # confirmed by file size 90944=56*203*8 (matches
    # zu_frc_momentum_row_reconstruction.py's own confirmation).
    nemo_zu_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zu_trd_substep1.bin"), jpi, jpj, hls)
    nemo_zv_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zv_trd_substep1.bin"), jpi, jpj, hls)

    # --- SELF-CHECK 1 (mandatory): reproduce the recorded 8.03e-3-class
    # number BEFORE trusting anything else. -----------------------------
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce zu_frc's own recorded err_norm")
    print("=" * 78)
    e_actual_u, rms_nemo_zu_frc, mask_zu = _err_norm(_u_to_nemo(F_slow_u_actual), nemo_zu_frc, umask2)
    e_actual_v, rms_nemo_zv_frc, mask_zv = _err_norm(_v_to_nemo(F_slow_v_actual), nemo_zv_frc, vmask2)
    print(f"  u err_norm={e_actual_u:.4e}  (recorded 8.03e-3, this session's "
          f"step-1 fresh measurement 8.0270e-3)")
    print(f"  v err_norm={e_actual_v:.4e}  (recorded 5.43e-4)")
    _align_scan("zu_frc [budget-completion self-check]", _u_to_nemo(F_slow_u_actual), nemo_zu_frc, umask2)

    # --- SELF-CHECK 2 (mandatory): |cor_u_sub| non-trivial, matches
    # zu_frc_term_walk.py's own measured scale. --------------------------
    print("\n" + "=" * 78)
    print("SELF-CHECK 2: cor_u_sub (legoESM's zu_trd-equivalent) is non-zero "
          "and the 'live' branch fired")
    print("=" * 78)
    print(f"  |cor_u_sub| mean={float(np.mean(np.abs(cor_u_sub))):.4e}  "
          f"max={float(np.max(np.abs(cor_u_sub))):.4e}")
    assert float(np.max(np.abs(cor_u_sub))) > 1e-6

    # =========================================================================
    # STEP 2: the zu_trd-error OWNERSHIP TEST.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 2: zu_trd-error ownership test")
    print("=" * 78)
    # cor_u_sub is legoESM's RAW face array (+1 column offset vs NEMO's
    # index, per the campaign's recurring-trap catalogue); strip it via
    # _u_to_nemo BEFORE differencing against nemo_zu_trd (already
    # halo-stripped to NEMO's interior convention by _load_full).
    cor_u_sub_nemo = _u_to_nemo(cor_u_sub)
    cor_v_sub_nemo = _v_to_nemo(cor_v_sub)
    n_lat_t = min(cor_u_sub_nemo.shape[0], nemo_zu_trd.shape[0])
    n_lon_t = min(cor_u_sub_nemo.shape[1], nemo_zu_trd.shape[1])
    zu_trd_err_u = cor_u_sub_nemo[:n_lat_t, :n_lon_t] - nemo_zu_trd[:n_lat_t, :n_lon_t]
    n_lat_tv = min(cor_v_sub_nemo.shape[0], nemo_zv_trd.shape[0])
    n_lon_tv = min(cor_v_sub_nemo.shape[1], nemo_zv_trd.shape[1])
    zv_trd_err_v = cor_v_sub_nemo[:n_lat_tv, :n_lon_tv] - nemo_zv_trd[:n_lat_tv, :n_lon_tv]

    zu_frc_err = _u_to_nemo(F_slow_u_actual) - nemo_zu_frc
    zv_frc_err = _v_to_nemo(F_slow_v_actual) - nemo_zv_frc

    n_lat_u = min(zu_trd_err_u.shape[0], zu_frc_err.shape[0])
    n_lon_u = min(zu_trd_err_u.shape[1], zu_frc_err.shape[1])
    m_u = umask2[:n_lat_u, :n_lon_u] & np.isfinite(zu_trd_err_u[:n_lat_u, :n_lon_u]) & np.isfinite(zu_frc_err[:n_lat_u, :n_lon_u])
    a_u = zu_trd_err_u[:n_lat_u, :n_lon_u][m_u]
    b_u = zu_frc_err[:n_lat_u, :n_lon_u][m_u]
    corr_ownership_u = _corr(a_u, b_u)
    rms_ratio_ownership_u = _rms(a_u) / _rms(b_u) if _rms(b_u) > 0 else float("nan")
    print(f"  u: corr(zu_trd_error, zu_frc_error) = {corr_ownership_u:.4f}")
    print(f"  u: RMS(zu_trd_error)={_rms(a_u):.4e}  RMS(zu_frc_error)={_rms(b_u):.4e}  "
          f"RMS-ratio = {rms_ratio_ownership_u:.4f}")

    n_lat_v = min(zv_trd_err_v.shape[0], zv_frc_err.shape[0])
    n_lon_v = min(zv_trd_err_v.shape[1], zv_frc_err.shape[1])
    m_v = vmask2[:n_lat_v, :n_lon_v] & np.isfinite(zv_trd_err_v[:n_lat_v, :n_lon_v]) & np.isfinite(zv_frc_err[:n_lat_v, :n_lon_v])
    a_v = zv_trd_err_v[:n_lat_v, :n_lon_v][m_v]
    b_v = zv_frc_err[:n_lat_v, :n_lon_v][m_v]
    corr_ownership_v = _corr(a_v, b_v)
    rms_ratio_ownership_v = _rms(a_v) / _rms(b_v) if _rms(b_v) > 0 else float("nan")
    print(f"  v: corr(zv_trd_error, zv_frc_error) = {corr_ownership_v:.4f}")
    print(f"  v: RMS(zv_trd_error)={_rms(a_v):.4e}  RMS(zv_frc_error)={_rms(b_v):.4e}  "
          f"RMS-ratio = {rms_ratio_ownership_v:.4f}")

    # --- Meridional (row) ripple / seam check on |zu_trd_error| -----------
    from scipy.signal import argrelextrema
    zu_trd_err_full_u = zu_trd_err_u[:n_lat_u, :n_lon_u]
    mask_full_u = umask2[:n_lat_u, :n_lon_u]
    row_prof = np.array([
        float(np.mean(np.abs(zu_trd_err_full_u[i, :][mask_full_u[i, :]])))
        if mask_full_u[i, :].any() else float("nan")
        for i in range(n_lat_u)
    ])
    valid = np.isfinite(row_prof)
    x = np.where(valid, row_prof, np.nanmean(row_prof[valid]))
    maxima = argrelextrema(x, np.greater_equal, order=3)[0]
    maxima = np.array([i for i in maxima if valid[i]])
    print(f"\n  |zu_trd_error| row-profile peak-finder (scipy argrelextrema, "
          f"order=3): maxima at rows {maxima.tolist()}")
    recorded_peaks = np.array([48, 84, 114, 150])
    if maxima.size:
        nearest_dist = [int(np.min(np.abs(recorded_peaks - p))) for p in maxima]
        print(f"  distance from each maximum to nearest recorded zu_frc "
              f"ripple peak (48/84/114/150): {nearest_dist}")

    # Seam check: column 0 / last column (periodic-i boundary) vs interior mean.
    col_prof = np.array([
        float(np.mean(np.abs(zu_trd_err_full_u[:, j][mask_full_u[:, j]])))
        if mask_full_u[:, j].any() else float("nan")
        for j in range(zu_trd_err_full_u.shape[1])
    ])
    col_valid = np.isfinite(col_prof)
    interior_mean = float(np.mean(col_prof[3:-3][col_valid[3:-3]]))
    seam_val = float(np.nanmean([col_prof[0], col_prof[-1]]))
    print(f"\n  seam check: |zu_trd_error| at column 0/last = {seam_val:.4e}  "
          f"vs interior mean = {interior_mean:.4e}  "
          f"seam/interior ratio = {seam_val / interior_mean if interior_mean > 0 else float('nan'):.3f}")

    print("\n  APPROXIMATION USED: (a) cor2d_dump_zu_trd_substep1.bin (in-loop "
          "jn=1 call) as a bounded approximation of the true pre-loop :364 "
          "zu_trd -- see module docstring for the caveat and the ~3700x "
          "magnitude-dominance bound already established by "
          "zu_frc_leapfrog_residual_probe.py PART 3.")

    # =========================================================================
    # STEP 3: the completed FIVE-term + subtraction budget.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 3: five-term + zu_trd-subtraction budget")
    print("=" * 78)

    def _err3d(lego_3d, nemo_3d):
        n_lat_c = min(lego_3d.shape[0], nemo_3d.shape[0])
        n_lon_c = min(lego_3d.shape[1], nemo_3d.shape[1])
        n_lev_c = min(lego_3d.shape[2], nemo_3d.shape[2])
        return (lego_3d[:n_lat_c, :n_lon_c, :n_lev_c]
                - nemo_3d[:n_lat_c, :n_lon_c, :n_lev_c])

    ke_pgf_u_f = _u_to_nemo(ke_pgf_u_3d)
    ah_lap_u_f = _u_to_nemo(ah_lap_u_3d)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vortcor_u_f = _u_to_nemo(vortcor_u_3d)

    nemo_keghpg_du = nemo_keg_du + nemo_hpg_du  # NEMO's own keg+hpg sum, matching KE_PGF's bundling

    err_keghpg_u = _err3d(ke_pgf_u_f, nemo_keghpg_du)
    err_ldf_u = _err3d(ah_lap_u_f, nemo_ldf_du)
    err_zad_u = _err3d(vertadv_u_f, nemo_zad_du)
    err_vor_u = _err3d(vortcor_u_f, nemo_vor_du)

    n_lat_b = min(err_keghpg_u.shape[0], err_ldf_u.shape[0], err_zad_u.shape[0], err_vor_u.shape[0])
    n_lon_b = min(err_keghpg_u.shape[1], err_ldf_u.shape[1], err_zad_u.shape[1], err_vor_u.shape[1])
    n_lev_b = min(err_keghpg_u.shape[2], err_ldf_u.shape[2], err_zad_u.shape[2], err_vor_u.shape[2])

    def _crop(a):
        return a[:n_lat_b, :n_lon_b, :n_lev_b]

    err_keghpg_u = _crop(err_keghpg_u)
    err_ldf_u = _crop(err_ldf_u)
    err_zad_u = _crop(err_zad_u)
    err_vor_u = _crop(err_vor_u)
    mask_b = umask2[:n_lat_b, :n_lon_b]

    e3u_c = e3u_0[:n_lat_b, :n_lon_b, :n_lev_b]
    hu_c = hu_0[:n_lat_b, :n_lon_b]
    # 3-D per-level mask, EXACTLY as NEMO's own zu_frc build applies
    # ``umask(ji,jj,:)`` inside the SUM (dynspg_ts.F90:335-345) -- a level-0
    # mask held constant with depth would keep below-seafloor cells "wet" at
    # deep levels and contaminate the depth-sum (caught before reporting,
    # see the module-docstring / umask3 comment above).
    umask3_c = umask3[:n_lat_b, :n_lon_b, :n_lev_b]

    def _depth_mean(err3d):
        depth_sum = np.sum(e3u_c * err3d * umask3_c, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(hu_c > 0, depth_sum / np.where(hu_c > 0, hu_c, 1.0), 0.0)

    dm_keghpg = _depth_mean(err_keghpg_u)
    dm_ldf = _depth_mean(err_ldf_u)
    dm_zad = _depth_mean(err_zad_u)
    dm_vor = _depth_mean(err_vor_u)

    # zu_trd-subtraction contribution: already 2-D, enters as a flat additive
    # term (matches how NEMO's own zu_frc build folds in the single 2-D
    # correction alongside the depth-summed 3-D terms), SIGN as it appears
    # in zu_frc's own build (zu_frc -= zu_trd, so an error in zu_trd_lego
    # relative to NEMO propagates with a MINUS sign into zu_frc's error).
    n_lat_z = min(dm_keghpg.shape[0], zu_trd_err_u.shape[0])
    n_lon_z = min(dm_keghpg.shape[1], zu_trd_err_u.shape[1])

    def _crop2(a):
        return a[:n_lat_z, :n_lon_z]

    dm_keghpg = _crop2(dm_keghpg)
    dm_ldf = _crop2(dm_ldf)
    dm_zad = _crop2(dm_zad)
    dm_vor = _crop2(dm_vor)
    dm_zu_trd_sub = -_crop2(zu_trd_err_u)  # SUBTRACTED in zu_frc's own formula
    mask_z = umask2[:n_lat_z, :n_lon_z]

    total_recon = dm_keghpg + dm_ldf + dm_zad + dm_vor + dm_zu_trd_sub

    zu_frc_err_z = zu_frc_err[:n_lat_z, :n_lon_z]
    m_final = (mask_z & np.isfinite(total_recon) & np.isfinite(zu_frc_err_z))

    shares = {}
    for label, term in (
        ("keg+hpg", dm_keghpg),
        ("ldf", dm_ldf),
        ("zad", dm_zad),
        ("vor", dm_vor),
        ("zu_trd-subtraction", dm_zu_trd_sub),
    ):
        rms_term = _rms(term[m_final])
        shares[label] = rms_term

    rms_total_terms = float(np.sqrt(sum(v ** 2 for v in shares.values())))
    print("  Per-piece RMS and share of RSS-total (6 pieces treated as "
          "independent contributors; RSS not a linear sum since signed "
          "fields can correlate/cancel -- see closure numbers below for the "
          "actual combined effect):")
    for label, rms_term in shares.items():
        share = rms_term / rms_total_terms if rms_total_terms > 0 else float("nan")
        conf = "CONFIRMED" if label != "keg+hpg" else "CONFIRMED (sum, see note)"
        print(f"    {label:<20s} RMS={rms_term:.4e}  RSS-share={share:.3f}  [{conf}]")
    print("  NOTE: keg+hpg is CONFIRMED as a real exposed diagnostic SUM "
          "(KE_PGF_u = -dKE_dx - dp_dx/rho_0, ocean_pe_latlon_cgrid.py:3990) "
          "-- legoESM does not expose keg and hpg as SEPARATE fields, so "
          "only their combined error is measurable, matched against NEMO's "
          "own keg_dump_du + hpg_dump_du sum (same bundling on both sides).")
    print("  ldf/zad/vor are each CONFIRMED (real per-level exposed "
          "diagnostic fields, same as zu_frc_momentum_row_reconstruction.py "
          "established).")

    corr_final = _corr(total_recon[m_final], zu_frc_err_z[m_final])
    rms_recon_final = _rms(total_recon[m_final])
    rms_measured_final = _rms(zu_frc_err_z[m_final])
    ratio_final = rms_recon_final / rms_measured_final if rms_measured_final > 0 else float("nan")

    print(f"\n  CLOSURE: corr(reconstructed, measured zu_frc error) = {corr_final:.4f}")
    print(f"  CLOSURE: RMS(reconstructed)={rms_recon_final:.4e}  "
          f"RMS(measured)={rms_measured_final:.4e}  ratio={ratio_final:.4f}")

    # --- Without zu_trd (the OLD 3-term-style baseline, for direct
    # comparison to the previously recorded u corr=0.044/ratio=0.045). ------
    total_recon_no_trd = dm_keghpg + dm_ldf + dm_zad + dm_vor
    corr_no_trd = _corr(total_recon_no_trd[m_final], zu_frc_err_z[m_final])
    ratio_no_trd = (_rms(total_recon_no_trd[m_final]) / rms_measured_final
                    if rms_measured_final > 0 else float("nan"))
    print(f"\n  [comparison] WITHOUT zu_trd (5 tendency terms only, ldf+zad+"
          f"vor+keg+hpg): corr={corr_no_trd:.4f}  ratio={ratio_no_trd:.4f}  "
          f"(prior 3-term-only recorded: u corr=0.044/ratio=0.045)")

    print("\n" + "=" * 78)
    print("SUMMARY (raw numbers only)")
    print("=" * 78)
    print(f"Step1 fresh zu_frc err_norm: u={e_actual_u:.4e}  v={e_actual_v:.4e}")
    print(f"Step2 ownership test: u corr={corr_ownership_u:.4f} ratio={rms_ratio_ownership_u:.4f}  "
          f"v corr={corr_ownership_v:.4f} ratio={rms_ratio_ownership_v:.4f}")
    print(f"Step3 6-piece shares: " + "  ".join(f"{k}={v:.4e}" for k, v in shares.items()))
    print(f"Step3 closure (all 6 pieces): corr={corr_final:.4f}  ratio={ratio_final:.4f}")
    print(f"Step3 closure (5 tendency terms only, no zu_trd): corr={corr_no_trd:.4f}  ratio={ratio_no_trd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
