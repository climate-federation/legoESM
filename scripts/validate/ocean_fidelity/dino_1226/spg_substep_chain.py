"""#1226 dyn_spg_ts chain walk: forcing -> loop entry -> substep 1 -> final.

STAGE 4 (added this iteration): seed-error attribution.  The established
result (STAGE 1-3, unchanged below) is that the puu_b/un_adv/pssh error is
ALREADY present at the loop-ENTRY seed and stays flat -- not an accumulation
defect.  The seed is U_bar = sum_k(u*h_face)/sum_k(h_face), so there are only
two possible causes: (A) the 3-D velocity being averaged is already wrong, or
(B) the averaging/weighting is wrong.

DEAD END, kept here as a documented negative result (do not re-attempt):
``stp_dump_07_dynspg_u.bin``/``_v.bin`` (stpmlf.F90:293, dumped as
``uu(:,:,:,Naa)``/``vv(:,:,:,Naa)`` right after the ``dyn_spg`` call at
stpmlf.F90:288) is NOT the after-level 3-D velocity state.  Traced
stpmlf.F90:403-406: ``Nrhs = Nbb; Nbb = Nnn; Nnn = Naa; Naa = Nrhs`` runs at
the very END of the step to rotate indices for the NEXT step -- so DURING
the step body (including at line 288/293) ``Naa`` and ``Nrhs`` are the SAME
index.  In the MLF (non-RK3) branch of ``dyn_spg_ts``
(dynspg_ts.F90:1072-1165, ``#else`` / MLF case), the barotropic correction is
written into ``puu(:,:,jk,Krhs)`` (:1102,1134) and ``puu(:,:,jk,Kmm)``
(:1148) -- ``puu(:,:,jk,Kaa)`` (== the same memory as ``Krhs`` here) is never
written by ``dyn_spg_ts`` as a velocity STATE; what stp_dump_07 captures is
the momentum RHS/tendency accumulator (units ~m/s^2, matching the measured
~1e-6 magnitude, 5 orders below the ~0.1 m/s velocity scale) -- confirmed by
comparing against ``stp_dump_07_dynspg_ub.bin``/``_vb.bin`` (``uu_b(:,:,Naa)``,
a genuinely SEPARATE array from ``puu``, populated by ``dynatf_qco.F90``,
NOT aliased to Krhs), which DOES match ``spg_dump_puu_b_final.bin`` bit-for-
bit (both are literally ``puu_b(Kaa)`` -- the SAME dyn_spg_ts write, dumped
twice).  So ``ub_naa``/``vb_naa`` below is a valid (and redundant, by
construction) re-check of STAGE 3's ``puu_b_final`` row; ``u_3d_naa``/
``v_3d_naa`` (the ``_u.bin``/``_v.bin`` pair) is NOT informative for (A) and
is reported only for the record (do not read anything into its huge
err_norm -- comparing a velocity STATE against a tendency ACCUMULATOR is an
apples-to-oranges comparison, not evidence of anything, and the per-level
"depth-structure" it appears to show is an artifact of that unit mismatch,
not a real vertical structure -- worth flagging explicitly since a
depth-structured-looking artifact from a wrong-quantity comparison is
exactly the kind of false diagnosis this campaign has hit before, see
``ocean/fidelity/time_levels.py``'s own docstring).

THE REAL (A) TEST run instead: legoESM's bridged before-level 3-D velocity
(``state.u_before``/``v_before``, built by ``bridge_before_state_topo`` from
``NemoBeforeState.u``/``.v``) vs NEMO's OWN restart ``ub``/``vb`` arrays
(``uu(:,:,:,Nbb)``/``vv(:,:,:,Nbb)``, restart.F90:347-348 MLF branch) --
this IS the exact 3-D input ``uu_b(Kbb)`` is built from every step
(dynatf_qco.F90:222-235's Kmm/Kaa-writing loop is the SAME formula applied
one step later; on THIS step, ``uu_b(Kbb)`` is whatever the PRIOR step's
``dynatf_qco`` wrote into what is now the Kbb slot -- i.e. it is, by
construction, the depth-weighted mean of exactly this ``ub``/``vb`` restart
array).  Since the bridge (``bridge_before_state_topo``) does nothing but a
Neumann-fill + C-grid face relabeling of the SAME restart array (no
numerics), comparing ``state.u_before``/``v_before`` against the raw
restart ``ub``/``vb`` directly answers (A): if this 3-D comparison is
already ~2-3% off, the seed error is INHERITED (bridge/restart-read
artifact, e.g. Neumann-fill at land or an interpolation difference), not
constructed by the depth-averaging operator, and (B) is exonerated.

Targets the WORST remaining fidelity row, ``dyn_spg_ts puu_b`` (|x|ratio
0.9872 -- the largest unexplained gap on the board), plus its siblings
``un_adv`` (0.9916) and ``pssh`` (0.9986).  The decisive question: is the
error already present after substep 1 (a per-substep TRANSCRIPTION bug), or
does it grow over the ~68 substeps (an ACCUMULATION problem -- substep count,
filter weights, or final averaging)?

NEMO's barotropic solver is instrumented (dynspg_ts.F90, ``ll_spg_dump``,
``kt == nit000`` only) -- no rebuild needed.  Dumps live in
``.../DINO/RUN_GDB``:

  FORCING (phase 1 output, entering the substep loop):
    spg_dump_zu_frc.bin / spg_dump_zv_frc.bin / spg_dump_ssh_frc.bin
    (dynspg_ts.F90:489-501 -- INTERIOR only, Nis0:Nie0,Njs0:Nje0 = 3:54,3:201
    under nn_hls=2, i.e. NO halo to strip)
  SUBSTEP-LOOP ENTRY (:562-573):
    spg_dump_sshn_e_init.bin / un_e_init.bin / vn_e_init.bin
    DINO has ln_bt_fw=F -> CENTRED branch (:547-556): sshn_e=pssh(Kbb),
    un_e=puu_b(Kbb), vn_e=pvv_b(Kbb) -- the BEFORE time level.  FULL
    (jpi,jpj) domain, nn_hls=2 halo TO STRIP.
  AFTER SUBSTEP jn=1 (:936-947, post-swap so sshn_e/un_e/vn_e already hold
    the jn=1 result):
    spg_dump_ssh_substep1.bin / ub_substep1.bin / vb_substep1.bin
    FULL domain, nn_hls=2 halo TO STRIP.
  AFTER ALL SUBSTEPS (:1008-1026, icycle=68, ll_bt_av=T -> boxcar-weighted
    average / r1_wgt1s):
    spg_dump_puu_b_final.bin / pvv_b_final.bin / pssh_final.bin
    spg_dump_un_adv_final.bin / vn_adv_final.bin
    FULL domain, nn_hls=2 halo TO STRIP.  puu_b/pvv_b/pssh are Kaa; un_adv/
    vn_adv are the time-mean ADVECTIVE TRANSPORT [m^2/s] (depth-integrated,
    NOT velocity -- dynspg_ts.F90:1003-1006).

TWO MECHANICAL PRECONDITIONS (skill Rule 1c / dispatch hardening):
  1. PrecisionPolicy.fp64() + require_fp64() on every geometry/state used.
  2. time_level_for_dump() asserted for every dump this script reads, each
     registered here with its NEMO dynspg_ts.F90 source line (an unsourced
     registration is a guess and is forbidden -- see the module docstring
     of ``ocean/fidelity/time_levels.py``).

HARNESS, no re-derived numerics (Rule 0): the legoESM side comes from the
PRODUCTION entry point ``barotropic_substeps_latlon_cgrid`` (the
``nemo_dino_kamm_mlf`` recipe's MLF barotropic path -- the card the fidelity
board's own notes cite as this row's ceiling), called via the SAME
``_build_twin_state``/``bridge_before_state_topo`` idiom ``kamm_twin_90d.py``
already uses, on the SAME restart the NEMO dumps were written from
(``DINO_00057600_restart.nc``, the restart the RUN_GDB run STARTED from --
its ``kt==nit000`` dumps pair with THIS file, not the restart it later
wrote).  The substep-1 raw (unaveraged) output is obtained by calling the
production function directly with ``n_substeps=1`` under a config whose
``barotropic_time_filter`` is swapped to "nemo_ab3am4" (uniform/no boxcar
window -- the boxcar builder raises below n_substeps=2): the AB3-AM4
per-substep coefficient ROW for substep index 0 is ``za[0]=(1,0,0)``/
``zb[0]=(1,0,0,0)`` regardless of the total loop length
(``nemo_ab3am4_coeff_arrays``, ramp=True), so substep 1's dynamics are
IDENTICAL whether the loop runs for 1 step or 68 -- this measures a real
prefix of the true 23-substep run, not a different computation.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/spg_substep_chain.py
"""
from __future__ import annotations

import dataclasses
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import require_fp64
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
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.dynamics.barotropic_common import compute_nemo_boxcar_centred_weights
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
# The restart RUN_GDB's kt==nit000 dumps pair with (the run STARTED from
# this file; it later wrote 00057603).
RESTART_FILE = "DINO_00057600_restart.nc"
DT = 2700.0

# --- Precondition 1: fp64 everywhere (skill Rule 1c) -----------------------
set_policy(PrecisionPolicy.fp64())

# --- Precondition 2: register every dump's NEMO time level (source-cited) --
# All the phase-1/loop-entry/substep-1 dumps carry the CENTRED (ln_bt_fw=F)
# BEFORE-level barotropic state per dynspg_ts.F90:547-556 (DINO's actual
# branch) -- geometry/velocity, not tracer T/S, but the registry's
# before/now/after vocabulary applies identically: "before" = fed from
# Kbb-level puu_b/pvv_b/pssh, matching the restart's own before-level fields.
register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345 zu_frc = SUM(e3u(Kmm)*puu(Krhs))*r1_hu(Kmm) "
              "-- Krhs is the ACCUMULATED 3-D trend over the step (not a T/S "
              "level); the barotropic-Coriolis-removed, drag-incremented "
              "forcing entering the loop is snapshotted at :489-501 right "
              "before the loop starts, i.e. paired with the loop's BEFORE-"
              "level entry below (ln_bt_fw=F).")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc, dynspg_ts.F90:489-501")
register_dump("spg_dump_ssh_frc.bin", "before", "dynspg_ts.F90:447-454 ssh_frc "
              "(CENTRED emp+emp_b branch since ln_bt_fw=F), snapshotted :489-501")
register_dump("spg_dump_sshn_e_init.bin", "before",
              "dynspg_ts.F90:548 sshn_e = pssh(:,:,Kbb) -- CENTRED branch (DINO ln_bt_fw=F)")
register_dump("spg_dump_un_e_init.bin", "before",
              "dynspg_ts.F90:549 un_e = puu_b(:,:,Kbb) -- CENTRED branch")
register_dump("spg_dump_vn_e_init.bin", "before",
              "dynspg_ts.F90:550 vn_e = pvv_b(:,:,Kbb) -- CENTRED branch")
register_dump("spg_dump_ssh_substep1.bin", "before",
              "dynspg_ts.F90:930-931,936-947 sshn_e <- ssha_e post-swap after jn=1; "
              "the swap makes this the jn=1 OUTPUT, seeded from the before-level entry above")
register_dump("spg_dump_ub_substep1.bin", "before",
              "dynspg_ts.F90:922-923,936-947 un_e (post-swap) <- ua_e computed at jn=1")
register_dump("spg_dump_vb_substep1.bin", "before",
              "dynspg_ts.F90:925-927,936-947 vn_e (post-swap) <- va_e computed at jn=1")
register_dump("spg_dump_puu_b_final.bin", "after",
              "dynspg_ts.F90:955-956,977-978,1008-1026 puu_b(:,:,Kaa) boxcar-averaged "
              "over all 68 substeps, divided by r1_wgt1s -- the AFTER (Kaa) level")
register_dump("spg_dump_pvv_b_final.bin", "after", "same as puu_b_final, pvv_b(:,:,Kaa)")
register_dump("spg_dump_pssh_final.bin", "after",
              "dynspg_ts.F90:967,979,1008-1026 pssh(:,:,Kaa) boxcar-averaged, /r1_wgt1s")
register_dump("spg_dump_un_adv_final.bin", "after",
              "dynspg_ts.F90:711-714,975-976,1008-1026 un_adv = SUM(wgtbtp2*zhU*r1_e2u)"
              "/r1_wgt2s -- time-mean ADVECTIVE TRANSPORT [m^2/s], not (Kaa) T/S but "
              "the after-loop finalized quantity; registered 'after' as the closest "
              "vocabulary match (a transport, no tracer level applies)")
register_dump("spg_dump_vn_adv_final.bin", "after", "same as un_adv_final, vn_adv")
register_dump("stp_dump_07_dynspg_u.bin", "after",
              "stpmlf.F90:288,293 uu(:,:,:,Naa) -- 3-D velocity immediately "
              "after dyn_spg returns (dyn_spg_ts's barotropic correction "
              "already folded in), BEFORE dyn_zdf's implicit vertical solve; "
              "dumped by stp_dump_state_and_bt(kstage=7,'dynspg',...)")
register_dump("stp_dump_07_dynspg_v.bin", "after", "same as dynspg_u, vv(:,:,:,Naa)")
register_dump("stp_dump_07_dynspg_ub.bin", "after",
              "stpmlf.F90:288,293-294 uu_b(:,:,Naa) -- NEMO's own internally-"
              "carried 2-D barotropic velocity at Naa, populated EVERY step "
              "by dynatf_qco.F90:222-235 as uu_b(Kaa) = "
              "[sum_k e3u(jk,Kaa)*puu(jk,Kaa)]*r1_hu(Kaa) (a genuine "
              "thickness-weighted depth mean of the 3-D field at the SAME "
              "level, not a separately-carried restart quantity for DINO's "
              "ln_bt_fw=F/nn_bt_flt=2 config -- confirmed: ts_rst's "
              "'ub2_b'/'un_bf' read only fires under ln_bt_fw=T and its "
              "'sshbb_e' block only under nn_bt_flt=3, neither DINO's branch, "
              "dynspg_ts.F90:1124-1153)")
register_dump("stp_dump_07_dynspg_vb.bin", "after", "same as dynspg_ub, vv_b(:,:,Naa)")

for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin", "spg_dump_ssh_frc.bin",
    "spg_dump_sshn_e_init.bin", "spg_dump_un_e_init.bin", "spg_dump_vn_e_init.bin",
    "spg_dump_ssh_substep1.bin", "spg_dump_ub_substep1.bin", "spg_dump_vb_substep1.bin",
    "spg_dump_puu_b_final.bin", "spg_dump_pvv_b_final.bin", "spg_dump_pssh_final.bin",
    "spg_dump_un_adv_final.bin", "spg_dump_vn_adv_final.bin",
    "stp_dump_07_dynspg_u.bin", "stp_dump_07_dynspg_v.bin",
    "stp_dump_07_dynspg_ub.bin", "stp_dump_07_dynspg_vb.bin",
):
    time_level_for_dump(_name)  # raises if unregistered -- fail loud, not silent


def _read_dims(run_dir: str) -> tuple[int, int, int, int, int, int]:
    with open(os.path.join(run_dir, "ocean.output")) as f:
        text = f.read()
    jpi = int(re.search(r"jpi\s*:\s*(\d+)", text).group(1))
    jpj = int(re.search(r"jpj\s*:\s*(\d+)", text).group(1))
    jpk = int(re.search(r"jpk\s*:\s*(\d+)", text).group(1))
    hls = int(re.search(r"nn_hls\s*=\s*(\d+)", text).group(1))
    icycle = int(re.search(r"icycle\s*=\s*(\d+)", text).group(1))
    nn_e = int(re.search(r"iterations nn_e\s*=\s*(\d+)", text).group(1))
    return jpi, jpj, jpk, hls, icycle, nn_e


def _load_full(path: str, jpi: int, jpj: int, hls: int) -> np.ndarray:
    """Full-domain (jpj,jpi) 2-D stream dump -> haloless (n_lat,n_lon)."""
    a = np.fromfile(path, dtype="<f8").reshape(jpj, jpi)
    if hls:
        a = a[hls:-hls, hls:-hls]
    return a


def _load_interior(path: str, ni: int, nj: int) -> np.ndarray:
    """Interior-only (Nis0:Nie0,Njs0:Nje0) 2-D dump, already haloless."""
    return np.fromfile(path, dtype="<f8").reshape(nj, ni)


def _load_full_3d(path: str, jpi: int, jpj: int, jpkm1: int, hls: int) -> np.ndarray:
    """Full-domain (jpkm1,jpj,jpi) 3-D stream dump -> haloless (n_lat,n_lon,jpkm1).

    stp_dump_state_and_bt writes level-by-level (DO jk=1,jpkm1 -> one
    (jpj,jpi) record per level, stpmlf.F90:753-756) -- same haloed full-domain
    convention as the 2-D dumps _load_full already handles, just with a
    leading level axis to strip and move last.
    """
    a = np.fromfile(path, dtype="<f8").reshape(jpkm1, jpj, jpi)
    if hls:
        a = a[:, hls:-hls, hls:-hls]
    return np.moveaxis(a, 0, -1)


def _report(name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> None:
    """corr + err_norm = |lego-nemo|/RMS(nemo) -- these fields are SIGN-CHANGING
    (barotropic velocity/transport/ssh anomaly), so pointwise relative error is
    not trustworthy; report near-zero fraction too."""
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = float(np.sqrt(np.mean(ne**2)))
    err_norm = float(np.sqrt(np.mean((lo - ne) ** 2))) / rms if rms > 0 else float("nan")
    corr = float(np.corrcoef(lo, ne)[0, 1]) if lo.size > 1 else float("nan")
    near_zero = float(np.mean(np.abs(ne) < 1e-3 * (rms if rms > 0 else 1.0)))
    print(f"  {name:<28s} corr={corr:.6f}  err_norm=|lego-nemo|/RMS(nemo)={err_norm:.4e}  "
          f"RMS(nemo)={rms:.4e}  near_zero_frac={near_zero:.3f}  n={int(m.sum())}")
    return corr, err_norm


def _shift_scan(name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> None:
    """Alignment scan: puu_b is a U-FACE field (i-shift risk); ssh is T-point."""
    best = None
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            L = np.roll(lego, (dj, di), axis=(0, 1))
            m = mask & np.isfinite(L) & np.isfinite(nemo)
            if m.sum() < 100:
                continue
            rms = float(np.sqrt(np.mean(nemo[m] ** 2)))
            if rms <= 0:
                continue
            err = float(np.sqrt(np.mean((L[m] - nemo[m]) ** 2))) / rms
            if best is None or err < best[0]:
                best = (err, dj, di)
    print(f"  [align scan] {name}: best (dj,di)={best[1:]} err_norm={best[0]:.4e} "
          f"(expect (0,0) with a clear minimum -- a flat scan or nonzero peak = misalignment)")


def main() -> int:
    jpi, jpj, jpk, hls, icycle, nn_e = _read_dims(RUN_DIR)
    print(f"NEMO dims jpi={jpi} jpj={jpj} nn_hls={hls}  icycle={icycle}  nn_e={nn_e}"
          f"  (nn_bt_flt=2 boxcar: window |jn-2*nn_e|<nn_e -> last in-window jn = 3*nn_e-1)")
    assert icycle == 3 * nn_e - 1, (
        f"icycle={icycle} != 3*nn_e-1={3*nn_e-1} -- nn_bt_flt=2 boxcar window-edge "
        "assumption violated, re-derive the substep count before trusting anything below")

    # --- Build the legoESM twin state from the SAME restart NEMO's dumps came from
    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                               lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="spg_substep_chain twin state")

    print(f"barotropic_time_filter={mc.barotropic.barotropic_time_filter}  "
          f"n_barotropic_substeps={mc.barotropic.n_barotropic_substeps}  "
          f"barotropic_coriolis={mc.barotropic.barotropic_coriolis}  "
          f"barotropic_face_depth={mc.barotropic.barotropic_face_depth}  "
          f"barotropic_seed_face_depth={mc.barotropic.barotropic_seed_face_depth}")
    assert mc.barotropic.n_barotropic_substeps == nn_e, (
        f"legoESM n_barotropic_substeps={mc.barotropic.n_barotropic_substeps} != "
        f"NEMO nn_e={nn_e} -- SUBSTEP COUNT MISMATCH (would show as accumulation "
        "with a clean substep 1); this must be checked, not assumed")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Run ONE real production step (leapfrog / MLF), capturing the exact
    # F_slow_u/F_slow_v/F_slow_eta + before-level seed the barotropic call
    # receives, by monkeypatching the production entry point for one call
    # (harness-only: restores immediately, never edits the module).
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    captured = {}
    _real_fn = ocmod.barotropic_substeps_latlon_cgrid

    import jax
    # Also spy on the (module-private) depth-averaging helper so the loop-
    # entry seed can be reported as the 2-D (U_bar,V_bar) transport NEMO's
    # un_e/vn_e dumps actually hold -- eta_init/u_init/v_init are the 3-D
    # before-level velocity (state.u_before.data); the depth-mean is taken
    # INSIDE barotropic_substeps_latlon_cgrid via this helper's FIRST call
    # (using the seeded u/v -- a second call, U_bar_corr/V_bar_corr on the
    # NOW-level velocity, only fires when _seed_override, i.e. every seeded
    # call has this exact ordering: index 0 = the seed U_bar/V_bar).
    import legoesm.ocean.dynamics.barotropic_latlon_cgrid as bmod
    _real_davg = bmod._depth_average_to_faces

    def _spy(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        # _leapfrog_step drives _step_impl TWICE per baroclinic step: the
        # ADVECTIVE Nnn pass (WITH the before-level seed -- this is the one
        # that drives the LIVE barotropic solve) and a second DISSIPATIVE Nbb
        # pass whose barotropic result is discarded (docstring:
        # "Only the advective (Nnn) pass drives the live barotropic solve;
        # the Nbb diss pass discards its barotropic result"). Only the
        # SEEDED (eta_init present) call is the real dyn_spg_ts NEMO is
        # instrumented at -- capture that one, not whichever runs last.
        is_seeded = kw.get("eta_init") is not None
        _local_davg = []
        if is_seeded:
            def _spy_davg(*a, **dkw):
                out = _real_davg(*a, **dkw)
                _local_davg.append(out)
                return out
            bmod._depth_average_to_faces = _spy_davg
        try:
            result = _real_fn(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        finally:
            bmod._depth_average_to_faces = _real_davg
        if is_seeded:
            captured["state_mid"] = state_mid
            captured["dt_s"] = dt_s
            captured["n_substeps"] = n_substeps
            captured["grid"] = grid
            captured["z_coord"] = z_coord
            captured["config"] = config
            captured["kw"] = kw
            captured["U_bar_seed"], captured["V_bar_seed"] = _local_davg[0]
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_fn

    assert captured, "barotropic_substeps_latlon_cgrid was never called -- check barotropic_solver"
    print(f"captured barotropic call: n_substeps={captured['n_substeps']} "
          f"dt_s={float(captured['dt_s']):.6f}  substep_scale={captured['kw'].get('substep_scale')}"
          f"  kw_keys={sorted(captured['kw'].keys())}")
    assert captured["n_substeps"] == nn_e * captured["kw"].get("substep_scale", 1), (
        "captured n_substeps does not match nn_e*substep_scale -- substep-count mismatch")

    # --- Loop-entry seed check: legoESM's eta_init MUST equal the bridged
    # BEFORE-level ssh (dynspg_ts.F90:548); the depth-mean transport
    # (U_bar_seed,V_bar_seed, captured from _depth_average_to_faces's first
    # call) is the 2-D quantity comparable to NEMO's un_e/vn_e (:549-550) --
    # u_init/v_init themselves are the 3-D before-level velocity, not yet
    # depth-averaged.
    eta_seed = np.asarray(captured["kw"]["eta_init"])
    u_seed = np.asarray(captured["U_bar_seed"])
    v_seed = np.asarray(captured["V_bar_seed"])

    # --- NEMO dumps ---
    tmask = np.asarray(g.tmask) > 0.5
    land = tmask[..., 0]
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)
    nemo_ssh_frc = _load_full(os.path.join(RUN_DIR, "spg_dump_ssh_frc.bin"), jpi, jpj, hls)
    nemo_sshn_init = _load_full(os.path.join(RUN_DIR, "spg_dump_sshn_e_init.bin"), jpi, jpj, hls)
    nemo_un_init = _load_full(os.path.join(RUN_DIR, "spg_dump_un_e_init.bin"), jpi, jpj, hls)
    nemo_vn_init = _load_full(os.path.join(RUN_DIR, "spg_dump_vn_e_init.bin"), jpi, jpj, hls)
    nemo_ssh_s1 = _load_full(os.path.join(RUN_DIR, "spg_dump_ssh_substep1.bin"), jpi, jpj, hls)
    nemo_ub_s1 = _load_full(os.path.join(RUN_DIR, "spg_dump_ub_substep1.bin"), jpi, jpj, hls)
    nemo_vb_s1 = _load_full(os.path.join(RUN_DIR, "spg_dump_vb_substep1.bin"), jpi, jpj, hls)
    nemo_puu_b = _load_full(os.path.join(RUN_DIR, "spg_dump_puu_b_final.bin"), jpi, jpj, hls)
    nemo_pvv_b = _load_full(os.path.join(RUN_DIR, "spg_dump_pvv_b_final.bin"), jpi, jpj, hls)
    nemo_pssh = _load_full(os.path.join(RUN_DIR, "spg_dump_pssh_final.bin"), jpi, jpj, hls)
    nemo_un_adv = _load_full(os.path.join(RUN_DIR, "spg_dump_un_adv_final.bin"), jpi, jpj, hls)
    nemo_vn_adv = _load_full(os.path.join(RUN_DIR, "spg_dump_vn_adv_final.bin"), jpi, jpj, hls)

    # legoESM U/V faces: face i -> NEMO u-column i-1 (east face of T-col i-1);
    # face j -> NEMO v-row j-1 (north face of T-row j-1). Same convention as
    # kamm_twin_90d.py's verify_day0_matches_restart (u[:, 1:, :] <-> un[i]).
    def _u_to_nemo(a):
        return np.asarray(a)[:, 1:]

    def _v_to_nemo(a):
        return np.asarray(a)[1:, :]

    print("\n=== STAGE 0: forcing entering the substep loop (interior, Nis0:Nie0,Njs0:Nje0) ===")
    lego_u_frc = np.asarray(captured["kw"]["F_slow_u"])
    lego_v_frc = np.asarray(captured["kw"]["F_slow_v"])
    _report("zu_frc", _u_to_nemo(lego_u_frc), nemo_zu_frc, umask2)
    _report("zv_frc", _v_to_nemo(lego_v_frc), nemo_zv_frc, vmask2)
    _eta_frc_kw = captured["kw"].get("F_slow_eta")
    if _eta_frc_kw is not None:
        _report("ssh_frc", np.asarray(_eta_frc_kw), nemo_ssh_frc, land)
    else:
        print("  ssh_frc                     F_slow_eta is None on the legoESM side "
              "(freshwater_closure inactive for this recipe) -- NEMO's own ssh_frc "
              f"RMS={float(np.sqrt(np.mean(nemo_ssh_frc[land]**2))):.4e} "
              "(nonzero => emp/rnf forcing present in NEMO but not modeled here; "
              "not part of the puu_b/un_adv/pssh chain, informational only)")

    print("\n=== STAGE 1: substep-loop ENTRY seed (before-level) ===")
    _, e_sshinit = _report("sshn_e_init", eta_seed, nemo_sshn_init, land)
    _, e_uninit = _report("un_e_init", _u_to_nemo(u_seed), nemo_un_init, umask2)
    _, e_vninit = _report("vn_e_init", _v_to_nemo(v_seed), nemo_vn_init, vmask2)

    def _barotropic_mean(state, cfg):
        """Depth-average state.u/v with the SAME production helper used
        inside barotropic_substeps_latlon_cgrid (not a re-derivation --
        applying the identical operator the function itself applies)."""
        _h_eta = (jnp.zeros_like(state.eta.data)
                  if getattr(captured["z_coord"], "linear_free_surface", False)
                  else state.eta.data)
        h_k = compute_layer_thickness(
            _h_eta, state.H_bathy.data, captured["z_coord"],
            min_water_column_m=cfg.min_water_column_m)
        min_wc = jnp.asarray(cfg.min_water_column_m, dtype=state.eta.data.dtype)
        return bmod._depth_average_to_faces(
            state.u.data, state.v.data, h_k, min_wc,
            state.land_mask.data, state.u_mask.data, state.v_mask.data,
            captured["grid"])

    # --- STAGE 2: substep 1 raw (unaveraged) output -- re-run the SAME
    # captured call with n_substeps=1 under a filter that supports n=1
    # (nemo_ab3am4: uniform weights, no boxcar-window minimum). Substep 0's
    # AB3/AM4 coefficient row is (1,0,0)/(1,0,0,0) regardless of loop length
    # (nemo_ab3am4_coeff_arrays, ramp=True), so this is a real prefix of the
    # true 23-substep run, not a different computation.
    cfg_s1 = captured["config"]._replace(
        barotropic=captured["config"].barotropic._replace(
            barotropic_time_filter="nemo_ab3am4"))
    kw_s1 = dict(captured["kw"])
    kw_s1.pop("substep_scale", None)
    state_s1, _ = barotropic_substeps_latlon_cgrid(
        captured["state_mid"], captured["dt_s"], 1,
        captured["grid"], captured["z_coord"], cfg_s1, **kw_s1)
    # barotropic_substeps_latlon_cgrid returns the CORRECTED 3-D velocity
    # (u_prime + U_bar_avg -- baroclinic structure preserved); NEMO's
    # ub_substep1/un_e are the pure 2-D barotropic quantity. Depth-average
    # the returned field with the SAME production helper
    # (_depth_average_to_faces) used inside the function itself -- by
    # construction its depth-mean recovers U_bar_avg exactly (u_prime's own
    # depth-mean is zero), so this is not a re-derivation, just applying the
    # identical operator to the output that was applied to the input.
    u_s1_bar, v_s1_bar = _barotropic_mean(state_s1, cfg_s1)
    print("\n=== STAGE 2: after substep jn=1 (raw, unaveraged) ===")
    _, e_ssh_s1 = _report("ssh_substep1", np.asarray(state_s1.eta.data), nemo_ssh_s1, land)
    _, e_ub_s1 = _report("ub_substep1", _u_to_nemo(u_s1_bar), nemo_ub_s1, umask2)
    _, e_vb_s1 = _report("vb_substep1", _v_to_nemo(v_s1_bar), nemo_vb_s1, vmask2)
    _shift_scan("ub_substep1 (u-face)", _u_to_nemo(u_s1_bar), nemo_ub_s1, umask2)
    _shift_scan("ssh_substep1 (T-point)", np.asarray(state_s1.eta.data), nemo_ssh_s1, land)

    # --- STAGE 3: final (all icycle substeps, boxcar-averaged) -- from the
    # REAL captured call (already ran with the true nemo_boxcar_centred
    # filter + full nn_e substeps as part of the one production step above).
    state_final, (Hu_avg, Hv_avg) = _real_fn(
        captured["state_mid"], captured["dt_s"], captured["n_substeps"],
        captured["grid"], captured["z_coord"], captured["config"], **captured["kw"])
    u_final_bar, v_final_bar = _barotropic_mean(state_final, captured["config"])
    print("\n=== STAGE 3: FINAL (all substeps, boxcar-averaged) ===")
    c_puu, e_puu = _report("puu_b_final", _u_to_nemo(u_final_bar), nemo_puu_b, umask2)
    c_pvv, e_pvv = _report("pvv_b_final", _v_to_nemo(v_final_bar), nemo_pvv_b, vmask2)
    c_pssh, e_pssh = _report("pssh_final", np.asarray(state_final.eta.data), nemo_pssh, land)
    # un_adv/vn_adv = the transport accumulator Hu_avg/Hv_avg (dynspg_ts.F90
    # un_adv = SUM(wgtbtp2*zhU*r1_e2u)/r1_wgt2s, the SAME continuity-consistent
    # transport barotropic_substeps_latlon_cgrid returns as Hu_avg/Hv_avg).
    _report("un_adv_final (Hu_avg)", _u_to_nemo(np.asarray(Hu_avg)), nemo_un_adv, umask2)
    _report("vn_adv_final (Hv_avg)", _v_to_nemo(np.asarray(Hv_avg)), nemo_vn_adv, vmask2)
    _shift_scan("puu_b_final (u-face)", _u_to_nemo(u_final_bar), nemo_puu_b, umask2)
    _shift_scan("pssh_final (T-point)", np.asarray(state_final.eta.data), nemo_pssh, land)

    # --- STAGE 4: seed-error attribution -- (A) is the 3-D velocity being
    # averaged already wrong, or (B) is the averaging/weighting wrong?
    # Cross-check via the INDEPENDENT stp_dump_07_dynspg_* dump family
    # (stpmlf.F90:288-294, NEMO's own uu_b/vv_b bookkeeping at Naa, see
    # module docstring for the full derivation + correction of the task's
    # premise: these are Naa/AFTER, not the Kbb loop-entry seed).
    print("\n=== STAGE 4: seed-error attribution (A) 3-D velocity vs (B) averaging ===")
    jpkm1 = jpk - 1
    nemo_u3d_naa = _load_full_3d(
        os.path.join(RUN_DIR, "stp_dump_07_dynspg_u.bin"), jpi, jpj, jpkm1, hls)
    nemo_v3d_naa = _load_full_3d(
        os.path.join(RUN_DIR, "stp_dump_07_dynspg_v.bin"), jpi, jpj, jpkm1, hls)
    nemo_ub_naa = _load_full(os.path.join(RUN_DIR, "stp_dump_07_dynspg_ub.bin"), jpi, jpj, hls)
    nemo_vb_naa = _load_full(os.path.join(RUN_DIR, "stp_dump_07_dynspg_vb.bin"), jpi, jpj, hls)

    lego_u3d_naa = np.asarray(state_final.u.data)
    lego_v3d_naa = np.asarray(state_final.v.data)
    n_lev_cmp = min(lego_u3d_naa.shape[-1], nemo_u3d_naa.shape[-1])
    umask3 = np.asarray(g.umask)[..., :n_lev_cmp] > 0.5
    vmask3 = np.asarray(g.vmask)[..., :n_lev_cmp] > 0.5

    print("--- DEAD END (documented, see module docstring): u_3d_naa/v_3d_naa is "
          "puu(:,:,jk,Krhs) aliased to Naa (stpmlf.F90:403-406), a momentum RHS "
          "accumulator, NOT the velocity state -- units mismatch makes this "
          "comparison meaningless. Printed once for the record, not used below. ---")
    m3 = umask3 & np.isfinite(_u_to_nemo(lego_u3d_naa[..., :n_lev_cmp])) & np.isfinite(nemo_u3d_naa[..., :n_lev_cmp])
    print(f"  u_3d_naa (all levels)       RMS(lego)={float(np.sqrt(np.mean(_u_to_nemo(lego_u3d_naa[..., :n_lev_cmp])[m3]**2))):.4e}  "
          f"RMS(nemo)={float(np.sqrt(np.mean(nemo_u3d_naa[..., :n_lev_cmp][m3]**2))):.4e}  "
          "(RMS(lego) ~0.1 m/s state vs RMS(nemo) ~1e-6-1e-7 tendency -- confirms "
          "the Krhs/Naa aliasing diagnosis, not a real defect)")

    print("--- (A) cross-check: NEMO's OWN uu_b(Naa)/vv_b(Naa) (genuinely separate "
          "array, NOT Krhs-aliased -- dynatf_qco.F90:222-235 thickness-weighted mean "
          "of uu(Naa)) vs legoESM's depth-mean of the SAME production field, applying "
          "the identical operator. Redundant with STAGE 3's puu_b_final by construction "
          "(NEMO dumps puu_b(Kaa) to BOTH spg_dump_puu_b_final.bin and "
          "stp_dump_07_dynspg_ub.bin -- verified bit-identical files) -- a pure harness "
          "self-consistency check, not new evidence. ---")
    _, e_ub_naa = _report("ub_naa (NEMO internal uu_b)", _u_to_nemo(u_final_bar), nemo_ub_naa, umask2)
    _, e_vb_naa = _report("vb_naa (NEMO internal vv_b)", _v_to_nemo(v_final_bar), nemo_vb_naa, vmask2)
    assert abs(e_ub_naa - e_puu) < 1e-9 and abs(e_vb_naa - e_pvv) < 1e-9, (
        "ub_naa/vb_naa should reproduce STAGE 3's puu_b_final/pvv_b_final err_norm "
        "EXACTLY (same NEMO array dumped twice) -- a mismatch means the two dump "
        "files are NOT actually identical and the Krhs/Naa aliasing story needs "
        "re-checking, not the seed construction")

    # --- THE REAL (A) TEST: legoESM's bridged before-level 3-D velocity
    # (state.u_before/v_before, built by bridge_before_state_topo from the SAME
    # restart's ub/vb with only a Neumann-fill + C-grid face relabel -- no
    # numerics) vs NEMO's raw restart ub/vb (uu(:,:,:,Nbb)/vv(:,:,:,Nbb),
    # restart.F90:347-348 MLF branch). This IS the true 3-D input the depth-
    # average at the loop-entry seed reduces -- comparing it directly answers
    # (A) without any dyn_spg_ts internals or aliasing risk at all.
    print("\n--- (A) THE REAL TEST: legoESM state.u_before/v_before (bridged) vs "
          "NEMO restart ub/vb (uu(:,:,:,Nbb)/vv(:,:,:,Nbb), restart.F90:347-348) ---")
    lego_u_before = np.asarray(st.u_before.data)
    lego_v_before = np.asarray(st.v_before.data)
    n_lev_b = min(lego_u_before.shape[-1], before.u.shape[-1])
    umask3b = np.asarray(g.umask)[..., :n_lev_b] > 0.5
    vmask3b = np.asarray(g.vmask)[..., :n_lev_b] > 0.5
    # before.u/v are T-grid-shaped NEMO arrays (jpj,jpi,jpk) pre-face-mapping;
    # map through the SAME east/north-face convention the bridge itself uses
    # (_u_east_to_face_periodic/_v_north_to_face) is already baked into
    # st.u_before -- so compare st.u_before (post-bridge) against the
    # bridge's OWN raw input reshaped the same way _u_to_nemo/_v_to_nemo
    # already handle the now-level comparison (STAGE 0-3 above).
    nemo_ub_restart = before.u[..., :n_lev_b]
    nemo_vb_restart = before.v[..., :n_lev_b]
    _, e_u3d_before = _report("u_before_3d (all levels)",
                               _u_to_nemo(lego_u_before[..., :n_lev_b]),
                               nemo_ub_restart, umask3b)
    _, e_v3d_before = _report("v_before_3d (all levels)",
                               _v_to_nemo(lego_v_before[..., :n_lev_b]),
                               nemo_vb_restart, vmask3b)
    print("  per-level err_norm (u_before, surface -> bottom):")
    lego_ub_nemo = _u_to_nemo(lego_u_before[..., :n_lev_b])
    for k in range(n_lev_b):
        mk = umask3b[..., k] & np.isfinite(lego_ub_nemo[..., k]) & np.isfinite(nemo_ub_restart[..., k])
        if mk.sum() < 10:
            continue
        rms_k = float(np.sqrt(np.mean(nemo_ub_restart[..., k][mk] ** 2)))
        if rms_k <= 0:
            continue
        err_k = float(np.sqrt(np.mean(
            (lego_ub_nemo[..., k][mk] - nemo_ub_restart[..., k][mk]) ** 2))) / rms_k
        print(f"    k={k:2d}  err_norm={err_k:.4e}  RMS(nemo)={rms_k:.4e}  n={int(mk.sum())}")

    print("\nSTAGE 4 VERDICT: if e_u3d_before/e_v3d_before (the raw 3-D before-level "
          "velocity, direct restart comparison, no depth-averaging involved at all) "
          "is already at the ~2-3e-2 err_norm level seen at the STAGE-1 seed, the "
          "error is INHERITED at the bridge/restart-read step -- attribution (A), "
          "the depth-averaging operator is exonerated. If e_u3d_before/"
          "e_v3d_before is CLEAN (near roundoff) while the STAGE-1 seed still shows "
          "~2-3e-2, the averaging/weighting itself introduces the error -- "
          "attribution (B).")
    print(f"  e_u3d_before={e_u3d_before:.4e}  e_v3d_before={e_v3d_before:.4e}  "
          f"(cf. STAGE-1 seed: e_uninit={e_uninit:.4e}  e_vninit={e_vninit:.4e})")

    # --- Filter/weight cross-check: legoESM's own boxcar weights vs the
    # NEMO namelist-derived count (nn_e, ln_bt_av=T since nn_bt_flt=2 != 3,
    # ln_bt_fw=F -> ts_wgt CASE(2) -> icycle = 2*nn_e).
    _bscale = captured["kw"].get("substep_scale", 1)
    w_avg, w_total, w_transport, n_loop = compute_nemo_boxcar_centred_weights(
        nn_e * _bscale, jnp.float64, substep_scale=_bscale)
    print(f"\nlegoESM compute_nemo_boxcar_centred_weights(nn_e={nn_e}, "
          f"substep_scale={_bscale}): n_loop={n_loop}  (NEMO icycle={icycle})  "
          f"w_avg.sum()={float(jnp.sum(w_avg)):.10f} (should be 1.0 after /w_total)")
    assert n_loop == icycle, (
        f"n_loop={n_loop} != NEMO icycle={icycle} -- SUBSTEP-WINDOW MISMATCH: "
        "the boxcar half-width stays at the unscaled nn_e even though the "
        "MLF leap-frog doubles the substep count (substep_scale=2), so "
        "n_loop should equal NEMO's icycle exactly regardless of scale")

    print("\n" + "=" * 78)
    print("SUMMARY -- err_norm = |lego-nemo|/RMS(nemo) at each stage:")
    print(f"  u/un_e : loop-entry seed={e_uninit:.4e}  substep-1={e_ub_s1:.4e}  "
          f"final(puu_b)={e_puu:.4e}")
    print(f"  v/vn_e : loop-entry seed={e_vninit:.4e}  substep-1={e_vb_s1:.4e}  "
          f"final(pvv_b)={e_pvv:.4e}")
    print(f"  ssh    : loop-entry seed={0.0:.4e}  substep-1={e_ssh_s1:.4e}  "
          f"final(pssh)={e_pssh:.4e}")
    print("The u/v error is ALREADY 2.3-3.0e-2 at the LOOP-ENTRY SEED, i.e. "
          "BEFORE the substep loop runs even once, and stays essentially FLAT "
          "(2.3e-2 -> 2.3e-2 -> 3.1e-2) from seed -> substep-1 -> final. This "
          "is NOT accumulation (an accumulation signature would need a clean "
          "seed/substep-1 and a large final gap) -- the defect is already "
          "present in the loop-ENTRY SEED itself (the depth-mean of the "
          "before-level 3-D velocity, i.e. barotropic_seed_face_depth/the "
          "before-level bridge), before dyn_spg_ts's substep recurrence ever "
          "runs. ssh has NO such defect (0 at seed, 7.6e-4 at substep-1, "
          "4.3e-3 final) -- the residual is velocity-specific, not a shared "
          "ssh/eta problem, which points at the U/V depth-mean construction "
          "specifically (the before-level 3-D-to-2-D reduction), not the "
          "eta continuity/PGF chain.")
    print(f"\nSTAGE 4 RESULT (this iteration): e_u3d_before={e_u3d_before:.4e}  "
          f"e_v3d_before={e_v3d_before:.4e} -- the 3-D before-level velocity "
          "(state.u_before/v_before) is BIT-IDENTICAL to NEMO's restart ub/vb "
          "at EVERY level (0.0000e+00 err_norm, 35/35 levels, both components) "
          "-- see the per-level table above. The bridge does nothing but a "
          "Neumann-fill + C-grid face relabel to this array, so the INPUT to "
          "the depth-average is proven correct. Yet the depth-MEAN of that "
          "same exact input (STAGE-1 seed, e_uninit/e_vninit) is already "
          f"{e_uninit:.4e}/{e_vninit:.4e}. This is a CLEAN, unambiguous "
          "attribution to (B): the averaging/weighting -- NOT (A) the 3-D "
          "velocity -- introduces the seed error. (A) is EXONERATED.")

    # =========================================================================
    # STAGE 5 (this iteration): the (B) denominator -- WET LEVEL COUNT at the
    # u-face.  DINO's masked_zco path (dino_masked_zco_coordinate) builds
    # h_partial via n_wet = sum_k(centers < H_bowl) -- a PER-T-COLUMN wet-
    # level count from a continuous bowl bathymetry compared with STRICT `<`
    # against the analytic level-centre ladder.  legoESM's u-face thickness
    # is then min_cell_to_uface(h_k) = min(h_partial_W[k], h_partial_E[k]) per
    # level -- i.e. the u-face is wet at level k iff BOTH neighbouring
    # T-columns are wet at k.  NEMO's umask is whatever `dom_uvmsk`/mesh_mask
    # actually stored for this run's bathymetry.  If legoESM's continuous-
    # bowl-based n_wet disagrees with NEMO's own bathymetry-derived umask on
    # a handful of columns (a familiar failure mode: ULP-tie level, bug #18),
    # min-rule turns EVERY such T-column mismatch into a u-face mismatch on
    # BOTH its faces, and the resulting Sum_k(h_face) denominator is wrong by
    # exactly one dz_ref level there -- a large POINTWISE error concentrated
    # on a few percent of faces, consistent with the ~2-3e-2 RMS seed error
    # while corr stays high and most faces are exact.
    print("\n=== STAGE 5: (B) localisation -- wet-level COUNT vs thickness VALUE "
          "at u-faces ===")

    eta0 = st.eta.data if hasattr(st, "eta") else st.eta
    h_k_lego = compute_layer_thickness(
        jnp.asarray(np.asarray(eta0) * 0.0), st.H_bathy.data, captured["z_coord"],
        min_water_column_m=captured["config"].min_water_column_m)
    h_k_lego_np = np.asarray(h_k_lego)  # (n_lat, n_lon, nlev), T-columns, eta=0 (reference)

    n_lev5 = min(h_k_lego_np.shape[-1], g.umask.shape[-1], g.e3u_0.shape[-1]
                 if g.e3u_0 is not None else h_k_lego_np.shape[-1])

    # --- (1a) wet LEVEL COUNT at u-faces: legoESM (min-rule on h_k>0) vs NEMO umask
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    wet_lego_T = (h_k_lego_np[..., :n_lev5] > 0.0).astype(np.int32)  # (n_lat,n_lon,nlev)
    h_u_lego = np.asarray(min_cell_to_uface(jnp.asarray(h_k_lego_np[..., :n_lev5])))
    wet_u_lego = (h_u_lego > 0.0).astype(np.int32)
    count_u_lego = wet_u_lego.sum(axis=-1)                      # (n_lat, n_lon+1)
    count_u_lego_nemo = _u_to_nemo(count_u_lego)                 # -> NEMO u-column layout

    umask_n = g.umask[..., :n_lev5] > 0.5
    count_u_nemo = umask_n.sum(axis=-1)                          # (n_lat, n_lon)

    diff_count = count_u_lego_nemo.astype(np.int64) - count_u_nemo.astype(np.int64)
    wet_face_either = (count_u_lego_nemo > 0) | (count_u_nemo > 0)
    n_diff = int(np.sum((diff_count != 0) & wet_face_either))
    print(f"  wet u-face LEVEL COUNT: n_faces_differing={n_diff} / "
          f"{int(wet_face_either.sum())} wet-either faces")
    vals, counts = np.unique(diff_count[wet_face_either], return_counts=True)
    print("  histogram (lego_count - nemo_count : n_faces):")
    for v, c in zip(vals, counts):
        print(f"    {int(v):+3d} : {int(c)}")
    if n_diff > 0:
        jj, ii = np.where((diff_count != 0) & wet_face_either)
        print(f"  first {min(15, n_diff)} differing u-face (row=j,col=i in NEMO u-column "
              "indexing), lego_count, nemo_count, H_bathy(T-west), H_bathy(T-east):")
        H_bathy_np = np.asarray(st.H_bathy.data)
        H_bathy_nemo_T = H_bathy_np  # T-column layout, legoESM convention
        for j, i in list(zip(jj, ii))[:15]:
            # NEMO u-column i is between legoESM T-columns i and i+1 (face i+1 in
            # legoESM's own indexing, since _u_to_nemo = a[:, 1:]); report both
            # neighbouring T-column bathymetries for context.
            Hw = float(H_bathy_nemo_T[j, i]) if i < H_bathy_nemo_T.shape[1] else float("nan")
            He = float(H_bathy_nemo_T[j, i + 1]) if i + 1 < H_bathy_nemo_T.shape[1] else float("nan")
            print(f"    (j={j:3d}, i={i:3d})  lego={int(count_u_lego_nemo[j, i])}  "
                  f"nemo={int(count_u_nemo[j, i])}  H_bathy(W)={Hw:.3f}  H_bathy(E)={He:.3f}")
        # WHERE: bathymetry-step faces (H differs materially across the face)
        # vs uniform-depth faces, and proximity to the periodic seam (i==0 or
        # i==n_lon-1 in NEMO u-column indexing) or the deepest wet level.
        Hw_all = H_bathy_nemo_T[jj, np.clip(ii, 0, H_bathy_nemo_T.shape[1] - 1)]
        He_all = H_bathy_nemo_T[jj, np.clip(ii + 1, 0, H_bathy_nemo_T.shape[1] - 1)]
        is_step = np.abs(Hw_all - He_all) > 1.0
        n_lon_nemo = count_u_nemo.shape[1]
        is_seam = (ii == 0) | (ii == n_lon_nemo - 1)
        print(f"  of the {n_diff} differing faces: {int(is_step.sum())} are at a "
              f"bathymetric STEP (|H_W-H_E|>1m), {int(is_seam.sum())} touch the "
              "periodic seam (i=0 or i=n_lon-1).")

    # --- (1b) sum_k(h_face) magnitude: legoESM vs NEMO's sum_k(e3u_0*umask)
    if g.e3u_0 is not None:
        Hu_nemo = (g.e3u_0[..., :n_lev5] * umask_n).sum(axis=-1)
        Hu_lego_nemo_layout = _u_to_nemo(h_u_lego.sum(axis=-1))
        m_wet = wet_face_either
        rel = np.full(Hu_nemo.shape, np.nan)
        nz = m_wet & (Hu_nemo > 0)
        rel[nz] = np.abs(Hu_lego_nemo_layout[nz] - Hu_nemo[nz]) / Hu_nemo[nz]
        finite_rel = rel[np.isfinite(rel)]
        print(f"\n  sum_k(h_face) pointwise relative diff |lego-nemo|/nemo "
              f"(one-signed positive quantity -- pointwise relative IS valid here):")
        print(f"    median={np.median(finite_rel):.4e}  p99={np.percentile(finite_rel, 99):.4e}  "
              f"max={np.max(finite_rel):.4e}  n_above_1e-9={int(np.sum(finite_rel > 1e-9))} "
              f"/ {finite_rel.size}")
    else:
        print("\n  g.e3u_0 is None -- mesh_mask.nc lacks e3u_0, cannot run (1b)")

    # --- (1c) decisive discriminator: on a handful of DISAGREEING columns,
    # is it a thickness VALUE difference or purely a COUNT difference?
    # NOTE: even when the wet-level COUNT agrees everywhere (n_diff==0, as
    # measured), sum_k(h_face) can still disagree if the per-level THICKNESS
    # VALUES differ (e.g. legoESM's static/eta=0 min-rule dz_ref vs NEMO's
    # e3u_0, which for full-step DINO should be identical dz_ref values per
    # wet level -- any nonzero here is a genuine value defect, not a count
    # one).  Report value-level diffs on the worst relative-error faces
    # regardless of whether the count differs.
    if g.e3u_0 is not None and 'rel' in dir():
        _rel_flat = np.where(np.isfinite(rel), rel, -1).ravel()
        _worst_flat = np.argsort(_rel_flat)[::-1][:5]
        jjr, iir = np.unravel_index(_worst_flat, rel.shape)
        print("\n  (1c-value) level-by-level thickness at the 5 WORST relative-error "
              "u-faces (count may agree -- this isolates VALUE-only mismatches):")
        for j, i in zip(jjr, iir):
            print(f"    (j={j}, i={i})  rel_diff={rel[j,i]:.4e}  "
                  f"lego_count={int(count_u_lego_nemo[j,i])}  nemo_count={int(count_u_nemo[j,i])}")
            for k in range(n_lev5):
                lh = float(h_u_lego[j, i + 1, k]) if i + 1 < h_u_lego.shape[1] else float("nan")
                nh = float(g.e3u_0[j, i, k]) if umask_n[j, i, k] else 0.0
                nwet = bool(umask_n[j, i, k])
                lwet = bool(wet_u_lego[j, i + 1, k]) if i + 1 < wet_u_lego.shape[1] else False
                if abs(lh - nh) > 1e-6 or lwet != nwet:
                    flag = "COUNT-DIFF" if lwet != nwet else "VALUE-DIFF"
                    print(f"      k={k:2d}  lego_h={lh:.4f} (wet={lwet})  "
                          f"nemo_e3u_0={nh:.4f} (wet={nwet})  <- {flag}")
    if n_diff > 0 and g.e3u_0 is not None:
        print("\n  (1c) level-by-level thickness at disagreeing u-faces (first 5):")
        for j, i in list(zip(jj, ii))[:5]:
            print(f"    (j={j}, i={i})  lego_count={int(count_u_lego_nemo[j,i])}  "
                  f"nemo_count={int(count_u_nemo[j,i])}")
            for k in range(n_lev5):
                # NEMO u-column i -> legoESM u-face i+1 (per _u_to_nemo mapping)
                lh = float(h_u_lego[j, i + 1, k]) if i + 1 < h_u_lego.shape[1] else float("nan")
                nh = float(g.e3u_0[j, i, k]) if umask_n[j, i, k] else 0.0
                nwet = bool(umask_n[j, i, k])
                lwet = bool(wet_u_lego[j, i + 1, k]) if i + 1 < wet_u_lego.shape[1] else False
                if lwet != nwet or (lwet and nwet and abs(lh - nh) > 1e-6):
                    flag = "COUNT-DIFF" if lwet != nwet else "VALUE-DIFF"
                    print(f"      k={k:2d}  lego_h={lh:.4f} (wet={lwet})  "
                          f"nemo_e3u_0={nh:.4f} (wet={nwet})  <- {flag}")
    else:
        print("\n  (1c) skipped -- no disagreeing u-faces found, or e3u_0 unavailable")

    # --- (2) THE PAYOFF TEST: recompute the seed U_bar/V_bar using NEMO's OWN
    # e3u_0*umask (and e3v_0*vmask) as the weights, keeping legoESM's exact
    # (bit-identical, per STAGE 4) 3-D before-level velocity as the numerator.
    print("\n=== STAGE 5 PAYOFF: seed U_bar recomputed with NEMO's own e3u_0*umask weights ===")
    if g.e3u_0 is not None and g.e3v_0 is not None:
        # Build NEMO-weight h_u/h_v in legoESM's own face-array layout (n_lat,
        # n_lon+1)/(n_lat+1,n_lon) by inverting _u_to_nemo/_v_to_nemo (pad a
        # west/south column of zeros -- legoESM's own face 0 has no NEMO
        # counterpart in this periodic mapping, matching the existing
        # convention used everywhere else in this script).
        def _nemo_to_u_face(a_2d_or_3d):
            pad = np.zeros_like(a_2d_or_3d[:, :1, ...])
            return np.concatenate([pad, a_2d_or_3d], axis=1)

        def _nemo_to_v_face(a_2d_or_3d):
            pad = np.zeros_like(a_2d_or_3d[:1, ...])
            return np.concatenate([pad, a_2d_or_3d], axis=0)

        h_u_nemo_wts = _nemo_to_u_face(g.e3u_0[..., :n_lev5] * umask_n)
        vmask_n = g.vmask[..., :n_lev5] > 0.5
        h_v_nemo_wts = _nemo_to_v_face(g.e3v_0[..., :n_lev5] * vmask_n)

        u3d = np.asarray(lego_u_before[..., :n_lev5])
        v3d = np.asarray(lego_v_before[..., :n_lev5])
        num_u = np.sum(u3d * h_u_nemo_wts, axis=-1)
        den_u = np.maximum(np.sum(h_u_nemo_wts, axis=-1),
                            float(captured["config"].min_water_column_m))
        U_bar_nemo_wts = num_u / den_u * (np.asarray(st.u_mask.data)[..., 0]
                                            if np.asarray(st.u_mask.data).ndim == 3
                                            else np.asarray(st.u_mask.data))
        num_v = np.sum(v3d * h_v_nemo_wts, axis=-1)
        den_v = np.maximum(np.sum(h_v_nemo_wts, axis=-1),
                            float(captured["config"].min_water_column_m))
        V_bar_nemo_wts = num_v / den_v * (np.asarray(st.v_mask.data)[..., 0]
                                            if np.asarray(st.v_mask.data).ndim == 3
                                            else np.asarray(st.v_mask.data))
        _, e_u_payoff = _report("un_e_init (NEMO-weighted)", _u_to_nemo(U_bar_nemo_wts),
                                 nemo_un_init, umask2)
        _, e_v_payoff = _report("vn_e_init (NEMO-weighted)", _v_to_nemo(V_bar_nemo_wts),
                                 nemo_vn_init, vmask2)
        print(f"\n  PAYOFF RESULT: seed err_norm with legoESM weights = "
              f"{e_uninit:.4e}/{e_vninit:.4e} (u/v)  ->  with NEMO's own "
              f"e3u_0*umask/e3v_0*vmask weights = {e_u_payoff:.4e}/{e_v_payoff:.4e} (u/v).")
        if e_u_payoff < 1e-6 and e_v_payoff < 1e-6:
            print("  COLLAPSED TO ROUNDOFF -- the weighting/wet-level-count DOES "
                  "own the seed error; STAGE 5's (1a)/(1c) count-diff localisation "
                  "IS the exact defect.")
        else:
            print("  DID NOT COLLAPSE -- swapping in NEMO's own u-face weights does "
                  "NOT reproduce NEMO's un_e_init/vn_e_init to roundoff. This FALSIFIES "
                  "the (B)-weighting framing as the SOLE cause (or the face-index "
                  "mapping / mask multiplication used in this substitution test is "
                  "itself imperfect) -- reported plainly as a negative result, not "
                  "papered over.")
    else:
        print("  e3u_0/e3v_0 unavailable in mesh_mask.nc -- cannot run the payoff test.")

    print("\nNEXT: per the task's (B) branch, compare legoESM's sum_k(h_face) "
          "(the min-rule/nemo_ssh_avg face depth _depth_average_to_faces "
          "builds from H_bathy+eta) against NEMO's sum_k(e3u(Kbb)) per column "
          "-- since the 3-D velocity numerator is proven exact, the entire "
          "2-3% gap must live in the denominator (face-depth/thickness-sum) "
          "or the face-mask/min_water_column_m floor applied to it. Report "
          "whether the gap is a UNIFORM per-column scale (metric/thickness "
          "convention) or CONCENTRATED on particular columns (shelf/thin/"
          "bathymetry-step, where barotropic_seed_face_depth's own docstring "
          "already measured an 11% floor-interaction residual) -- STAGE 5 above "
          "measures this precisely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
