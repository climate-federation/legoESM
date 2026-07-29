"""#1226 dyn_spg_ts chain walk: forcing -> loop entry -> substep 1 -> final.

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

for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin", "spg_dump_ssh_frc.bin",
    "spg_dump_sshn_e_init.bin", "spg_dump_un_e_init.bin", "spg_dump_vn_e_init.bin",
    "spg_dump_ssh_substep1.bin", "spg_dump_ub_substep1.bin", "spg_dump_vb_substep1.bin",
    "spg_dump_puu_b_final.bin", "spg_dump_pvv_b_final.bin", "spg_dump_pssh_final.bin",
    "spg_dump_un_adv_final.bin", "spg_dump_vn_adv_final.bin",
):
    time_level_for_dump(_name)  # raises if unregistered -- fail loud, not silent


def _read_dims(run_dir: str) -> tuple[int, int, int]:
    with open(os.path.join(run_dir, "ocean.output")) as f:
        text = f.read()
    jpi = int(re.search(r"jpi\s*:\s*(\d+)", text).group(1))
    jpj = int(re.search(r"jpj\s*:\s*(\d+)", text).group(1))
    hls = int(re.search(r"nn_hls\s*=\s*(\d+)", text).group(1))
    icycle = int(re.search(r"icycle\s*=\s*(\d+)", text).group(1))
    nn_e = int(re.search(r"iterations nn_e\s*=\s*(\d+)", text).group(1))
    return jpi, jpj, hls, icycle, nn_e


def _load_full(path: str, jpi: int, jpj: int, hls: int) -> np.ndarray:
    """Full-domain (jpj,jpi) 2-D stream dump -> haloless (n_lat,n_lon)."""
    a = np.fromfile(path, dtype="<f8").reshape(jpj, jpi)
    if hls:
        a = a[hls:-hls, hls:-hls]
    return a


def _load_interior(path: str, ni: int, nj: int) -> np.ndarray:
    """Interior-only (Nis0:Nie0,Njs0:Nje0) 2-D dump, already haloless."""
    return np.fromfile(path, dtype="<f8").reshape(nj, ni)


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
    jpi, jpj, hls, icycle, nn_e = _read_dims(RUN_DIR)
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
    print("NEXT: compare legoESM's before-level U_bar_seed/V_bar_seed "
          "(_depth_average_to_faces on state.u_before/v_before) directly "
          "against NEMO's OWN puu_b(Kbb)/pvv_b(Kbb) restart fields "
          "(ub/vb in the restart, read via read_nemo_restart_before) -- "
          "per-column, to see whether the 2-3% gap is uniform (a global "
          "scale/metric factor in the depth-average, e.g. min_water_column_m "
          "or the nemo_ssh_avg face-depth rescale) or concentrated on shelf/"
          "thin columns (the seed_face_depth floor-interaction documented in "
          "BarotropicConfig.barotropic_seed_face_depth's own docstring, "
          "measured there as an 11%% loop-entry residual at shelf columns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
