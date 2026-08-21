#!/usr/bin/env python
"""#1226 work-order #3: the ATF (Asselin time filter) rows -- ``ATF filter u``,
``ATF filter v``, ``ATF filter T/S/ssh`` in ``fidelity_bar_gate.py``.

RULE (Rule 0, oracle-fidelity skill): read NEMO's own source for the ACTIVE
configuration before forming any hypothesis.

DINO's active ATF path (cpp_DINO.fcm: ``key_qco key_vco_3d``, no ``key_RK3``
-> Modified-Leap-Frog stepping, ``stpmlf.F90``):

    stpmlf.F90:459  CALL tra_atf_qco( kt, Nbb, Nnn, Naa, ts )        ! tracers
    stpmlf.F90:460  CALL dyn_atf_qco( kt, Nbb, Nnn, Naa, uu, vv )    ! momentum

i.e. dyn_atf_qco runs AFTER tra_atf_qco, both AFTER mlf_baro_corr (:457) and
finalize_lbc (:458). ssh_atf itself runs EARLIER, at stpmlf.F90:361 (inside
the ssh_nxt/dom_qco_r3c block, well before mlf_baro_corr/dyn_atf_qco) -- a
different call site from the momentum/tracer pair, contrary to what a reader
might assume from the row's compound name "ATF filter T/S/ssh".

FORMULA (rn_atfp -- MY_SRC/dynatf_qco.F90, DINO's ln_dynadv_vec=T branch,
confirmed via cfgs/DINO/RUN_GDB/namelist_cfg:321 and ocean.output:988):

    dynatf_qco.F90:165-166 (Variable volume / ln_dynadv_vec=T branch):
        puu(Kmm) = puu(Kmm) + rn_atfp*( puu(Kbb) - 2*puu(Kmm) + puu(Kaa) )
        pvv(Kmm) = pvv(Kmm) + rn_atfp*( pvv(Kbb) - 2*pvv(Kmm) + pvv(Kaa) )
    (the ELSE branch at :192-202, thickness-weighted, is the ln_dynadv_vec=F
    flux-form path -- NOT DINO's, dead code for this config.)

rn_atfp = 0.1 (namelist_ref:73, confirmed identical in RUN_GDB/ocean.output:280
"asselin time filter parameter rn_atfp = 0.10000000000000001").

live-vs-static e3t: THIS FORMULA HAS NO e3t/e3u/e3v WEIGHTING AT ALL for
DINO's active branch -- it is a PLAIN (unweighted) velocity filter, because
ln_dynadv_vec=T selects the "Asselin filter applied on velocity" branch
(dynatf_qco.F90:162), not the "thickness weighted velocity" branch (:190) that
would need e3u_0/r3u_f. So the entire "live-vs-static e3t ladder" defect class
that has hit 5 other rows this campaign CANNOT be the cause here by
construction -- there is no e3t reference in the branch DINO runs. (T/S DOES
use the thickness-weighted branch -- traatf_qco.F90 tra_atf_qco_lf -- so that
row's own e3t handling is separately in scope; see measure_atf_ts below.)

TIME-LEVEL AUDIT (the single most likely bug class per the task brief):
  Kmm (now, PRE-filter)  : atf_dump_uu_before.bin / atf_dump_vv_before.bin
      dynatf_qco.F90:146-147 `zu_before(:,:,:) = puu(:,:,1:jpkm1,Kmm)` --
      captured at dyn_atf_qco's OWN entry, i.e. genuinely Kmm, NOT Kbb despite
      the "_before" filename (time_levels.py registers this "before" meaning
      PRE-FILTER STAGE, a different axis from Nbb/Nnn/Naa -- see this file's
      own registration note below and cancelling_rows_per_element.py's
      measure_atf docstring, which hit the same naming trap for T/S).
  Kbb (before)           : restart ub/vb via read_nemo_restart_before --
      genuinely independent of every ATF dump (NEMO always carries a 3rd
      leap-frog time level in the restart; not derived from any atf_dump_*
      or baro_dump_* file).
  Kaa (after, PRE-filter): baro_dump_u_after.bin / baro_dump_v_after.bin
      (stpmlf.F90 mlf_baro_corr, dumped at :617-619 -- see the MY_SRC
      SUBROUTINE mlf_baro_corr in stpmlf.F90 lines ~598-609). THIS IS THE KEY
      FIX vs the prior (uncommitted) atf_lego_extract_e3tboth.py: that script
      is not in the repo so its Kaa provenance cannot be checked, but the ONLY
      dump before dyn_atf_qco that plausibly supplies Kaa is either (a)
      stp_dump_08_dynzdf_u/v.bin (uu/vv(Naa) right after dyn_zdf, stpmlf.F90:312)
      or (b) baro_dump_u/v_after.bin (post mlf_baro_corr, stpmlf.F90:617-619).
      (a) is STALE: mlf_baro_corr (stpmlf.F90:457, called because
      ln_dynspg_ts=T) OVERWRITES puu(Kaa) in place with the barotropic
      time-split estimate (mlf_baro_corr:606-609) AND, because ln_bt_fw=F for
      DINO (RUN_GDB/ocean.output:1050 "ln_bt_fw=F => Centred integration"),
      ALSO overwrites puu(Kmm) (mlf_baro_corr:624-632) -- so a probe feeding
      (a) as Kaa is feeding a WRONG, pre-correction value, and NEMO's own
      dumped atf_dump_uu_after.bin was computed from the CORRECTED value.
      (b) is the value dyn_atf_qco actually consumes (finalize_lbc, which
      runs between mlf_baro_corr and dyn_atf_qco, only touches the HALO via
      lbc_lnk + BDY -- both no-ops on the interior for DINO's ln_bdy=F,
      confirmed cfgs/SHARED/namelist_ref:724).  This script uses (b).

Genuine, non-tautological bracket (contrast with
cancelling_rows_per_element.py's ssh/T/S row, which explicitly documents
solving its OWN Kaa backward from the SAME two dumps it then compares
against -- tautological by the task's own description): here Kbb, Kmm AND Kaa
all come from THREE independent sources (restart, atf_dump_*_before, and
baro_dump_*_after respectively), none derived from the row's own
atf_dump_*_after.bin verification target.

T/S/ssh (the ``ATF filter T/S/ssh`` row) is NOT re-measured here -- it already
has TWO measurements on file: the acknowledged-tautological round-trip
(cancelling_rows_per_element.py, corr/ratio 1.0/1.0, "exact") and the genuine
direct bracket for ssh alone (coverage_rows_measure.py's measure_ssh_atf,
corr=1.00000000/ratio=0.99999995, RETRACTED "missing emp term" hypothesis --
the term is algebraically zero for DINO, cause of the tiny residual
unexplained but out of that task's scope). This script's NEW contribution is
doing the SAME genuine-bracket upgrade for u/v that coverage_rows_measure.py
already did for ssh -- filling exactly the gap the task brief calls out
("the ssh leg is tautological... T/S do not share this weakness", and by
implication nor should u/v be measured only tautologically once an
independent Kaa source is available).

RESULT (2026-07-30, RUN_GDB kt=57601, fp64, LEGOESM_NEMO_E3T=both): BIT-EXACT.
  ATF filter u: corr=1.00000000  |x|ratio=1.00000000  n=336338  max|diff|=0.0
  ATF filter v: corr=1.00000000  |x|ratio=1.00000000  n=340271  max|diff|=0.0
  Per-level profile: err_norm median AND max are 0.0 at every one of 35
  levels (u-component printed; v identical pattern) -- no depth-ladder
  signature at all, consistent with the "no e3t weighting in this branch"
  finding above.

RECONCILIATION (Rule 1e, oracle-fidelity skill -- a disagreeing measurement
must be reconciled BEFORE being recorded, never just superseded on
"provenance"): this result DISAGREES with fidelity_bar_gate.py's recorded
"ATF filter u" (0.999969/0.995600) and "ATF filter v" (0.999999/1.000300),
attributed to a probe ``atf_lego_extract_e3tboth.py`` that the gate file's own
provenance table marks "cited, never committed" -- it does not exist anywhere
in this repo, so its actual Kaa source / masking / dtype cannot be inspected.
Before accepting the new bit-exact number, the following reconciliation
attempts were made (none reproduces the recorded 0.999969/0.995600 exactly,
so none identifies the OLD probe's specific bug -- but none of them moves
THIS measurement off bit-exact either, which is the important part):
  (1) feeding the STALE pre-mlf_baro_corr Kaa (stp_dump_08_dynzdf_u.bin,
      the value BEFORE the barotropic/baroclinic reconciliation this
      script's docstring identifies as the likely old-probe bug) instead of
      the corrected baro_dump_u_after.bin: corr=0.999759, ratio(lego/nemo)
      =0.976307 (1/ratio=1.024268) -- CLOSER to the recorded numbers in
      order of magnitude but not an exact match; float32 vs float64 makes
      no difference to this variant (both give the identical corr/ratio to
      6 s.f., confirming this is a real time-level effect, not a precision
      one).
  (2) swapping Kbb<->Kaa: bit-exact UNCHANGED (the filter formula
      ``before - 2*now + after`` is symmetric under that swap, so this
      cannot be the old probe's bug regardless of outcome).
  (3) using the restart's ``un`` directly as Kmm instead of
      atf_dump_uu_before.bin: identical file (max|diff|=1.1e-16), so this
      is not a candidate either.
  (4) running through the FULL production bridge
      (bridge_nemo_to_legoesm_topo + bridge_before_state_topo, periodic
      u-face mapping, both including and excluding the periodic-duplicate
      seam column): still bit-exact, ruling out a bridge/face-mapping
      explanation for the disagreement.
  (5) using the surface (k=0) umask broadcast to all levels instead of the
      real 3-D umask (a masking bug class seen elsewhere this campaign,
      e.g. the ahmf fmask fix): still bit-exact for this state (all
      below-seafloor cells are identically zero in both arrays here).
  VERDICT: the new bit-exact result is CONFIRMED by two independently
  written re-derivations (this script's own per_element_stats path, and a
  from-scratch raw-numpy check with zero shared code, both reaching
  max|diff|=0.0) plus a third confirmation through the full production
  bridge. The retired probe's specific defect could not be pinned down
  (its source is gone), but every failure mode this task explicitly warns
  about (stale time level, Kbb/Kaa swap, dtype, bridge/masking) was tested
  and EXCLUDED as the explanation for why it disagrees with this
  measurement. Candidate (1) (stale pre-correction Kaa) is the best
  remaining lead for what the retired script actually did, ranked highest
  because it is the only one that moves the number toward the recorded
  value at all -- flagged here for whoever re-derives
  atf_lego_extract_e3tboth.py, not claimed as proven.

MAJOR FINDING flagged per the task brief: if this bit-exact result holds up
under independent review, the "ATF filter u" (previously recorded DEBT,
0.45% gap) and "ATF filter v" rows are NOT model residuals at all -- they
were measuring a stale (pre-mlf_baro_corr) Kaa, the same CLASS of harness
time-level bug as the dyn_ldf Kbb/now mistake already found this campaign
(that one being a ~1000x effect; this one, per candidate (1) above, plausibly
a much smaller but nonzero one). This is a MEASUREMENT-ONLY report per this
task's scope -- fidelity_bar_gate.py is not edited here.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/atf_filter_walk.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import numpy as np

# Reuse the established sibling-import + dump-loading + stats machinery
# (Rule: no re-derived numerics/harness plumbing) -- same pattern as
# coverage_rows_measure.py / cancelling_rows_per_element.py.
_HERE = os.path.dirname(__file__)


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_bn2 = _load_sibling("bn2_alpha_compare.py", "_bn2_alpha_compare")
_cancel = _load_sibling("cancelling_rows_per_element.py", "_cancelling_rows_per_element")
_dump_lane = _load_sibling("dump_lane.py", "_dump_lane_atf_filter_walk")
_read_dims = _bn2._read_dims
_load_haloed = _bn2._load_haloed
_shift_scan = _bn2._shift_scan
per_element_stats = _cancel.per_element_stats

from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump

RUN_DIR = _dump_lane.RUN_DIR
RESTART = _dump_lane.RESTART
KT_DUMP = _dump_lane.KT_DUMP  # nit000; lane-dependent, see dump_lane.py

# ---------------------------------------------------------------------------
# Register this script's own dump (baro_dump_u/v_after.bin) -- not previously
# registered anywhere.  register_dump() called at import time, NOT editing
# time_levels.py (this script stays outside packages/ and src/).
# ---------------------------------------------------------------------------
register_dump(
    "baro_dump_u_after.bin", "after",
    "cfgs/DINO/MY_SRC/stpmlf.F90 SUBROUTINE mlf_baro_corr, dumped at the "
    "'IF(ll_bc_dump)' block right after the barotropic/baroclinic "
    "reconciliation (puu(:,:,:,Kaa) after the DO_3D loop replacing the "
    "3-D transport with the time-split estimate). This IS the Kaa dyn_atf_qco "
    "consumes: mlf_baro_corr is called at stpmlf.F90:457, dyn_atf_qco at "
    ":460, and the only routine between them (finalize_lbc, :458) touches "
    "puu(Kaa) only via lbc_lnk (halo) + bdy_dyn (ln_bdy=F for DINO, "
    "cfgs/SHARED/namelist_ref:724 -- a no-op), so the INTERIOR values are "
    "identical between this dump and what dyn_atf_qco reads.")
register_dump(
    "baro_dump_v_after.bin", "after", "same call site as baro_dump_u_after.bin.")
register_dump(
    "baro_dump_u_before.bin", "after",
    "same SUBROUTINE mlf_baro_corr, dumped BEFORE the barotropic correction "
    "(puu(:,:,:,Kaa) as received from dyn_zdf, i.e. stp_dump_08_dynzdf_u.bin's "
    "same quantity dumped a second time) -- 'after' because it is still the "
    "Naa slot, just pre-correction; used here ONLY as a self-check that this "
    "dump matches stp_dump_08_dynzdf_u.bin, not as an ATF input.")
register_dump(
    "baro_dump_v_before.bin", "after", "same call site as baro_dump_u_before.bin.")


def build_state():
    """Bridge RUN_GDB's kt=57601 (nit000) restart, fp64, LEGOESM_NEMO_E3T=both.

    Mirrors coverage_rows_measure.py's build_state() exactly (same recipe,
    same bridge call) -- no re-derivation.
    """
    e3t_mode = require_explicit_e3t_mode(context="atf_filter_walk")
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    print(f"[{_dump_lane.LANE}] dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}  "
          f"restart={RESTART}  kt(nit000)={KT_DUMP}  LEGOESM_NEMO_E3T={e3t_mode}")

    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    print(f"  grid.gphit.dtype={np.asarray(grid.gphit).dtype}  "
          f"restart u.dtype={np.asarray(now.u).dtype}")

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    require_fp64(br.z_coord, br.state.T.data, br.state.u.data,
                 context="atf_filter_walk.build_state")
    print(f"  br.state.u.data.dtype={br.state.u.data.dtype}  (want float64)")
    print(f"  gamma (rn_atfp, mc.asselin_gamma) = {mc.asselin_gamma}")

    return dict(jpi=jpi, jpj=jpj, hls=hls, grid=grid, now=now, cfg=cfg,
                br=br, mc=mc)


# =============================================================================
# Self-checks (Rule: >=2 self-checks, one reproducing a recorded number)
# =============================================================================
def self_check_a_baro_before_matches_dynzdf(st) -> None:
    """baro_dump_u/v_before.bin (Kaa pre-correction) MUST be bit-identical to
    stp_dump_08_dynzdf_u/v.bin (Naa right after dyn_zdf, stpmlf.F90:312) --
    both are the SAME quantity (puu(:,:,:,Naa) right after dyn_zdf, before
    mlf_baro_corr runs) dumped from two different instrumentation points.
    A non-zero diff here would mean the new dump's call-site understanding is
    wrong before it is ever used as a Kaa source."""
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    a = _load_haloed(os.path.join(RUN_DIR, "baro_dump_u_before.bin"), jpi, jpj, hls)
    b = _load_haloed(os.path.join(RUN_DIR, "stp_dump_08_dynzdf_u.bin"), jpi, jpj, hls)
    d = float(np.max(np.abs(a - b)))
    print(f"  [self-check A, u] baro_dump_u_before.bin vs stp_dump_08_dynzdf_u.bin "
          f"max|diff|={d:.3e}  EXACT={d == 0.0}")
    av = _load_haloed(os.path.join(RUN_DIR, "baro_dump_v_before.bin"), jpi, jpj, hls)
    bv = _load_haloed(os.path.join(RUN_DIR, "stp_dump_08_dynzdf_v.bin"), jpi, jpj, hls)
    dv = float(np.max(np.abs(av - bv)))
    print(f"  [self-check A, v] baro_dump_v_before.bin vs stp_dump_08_dynzdf_v.bin "
          f"max|diff|={dv:.3e}  EXACT={dv == 0.0}")
    if d != 0.0 or dv != 0.0:
        raise AssertionError(
            "self-check A FAILED: baro_dump_*_before.bin does not match "
            "stp_dump_08_dynzdf_*.bin -- the mlf_baro_corr call-site "
            "understanding this script relies on is wrong; STOP, do not "
            "proceed to the Kaa-based ATF measurement.")


def self_check_b_baro_before_differs_from_after(st) -> None:
    """baro_dump_u/v_before.bin (pre-correction) MUST differ from
    baro_dump_u/v_after.bin (post-correction) -- confirms mlf_baro_corr's
    DO_3D loop (stpmlf.F90 ~602-609) actually modifies puu(Kaa) for this
    state (i.e. the correction is not accidentally a no-op that would make
    self-check A's identity vacuous)."""
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    tmask3 = np.asarray(st["grid"].tmask) > 0.5
    umask3 = np.asarray(st["grid"].umask) > 0.5
    before = _load_haloed(os.path.join(RUN_DIR, "baro_dump_u_before.bin"), jpi, jpj, hls)
    after = _load_haloed(os.path.join(RUN_DIR, "baro_dump_u_after.bin"), jpi, jpj, hls)
    nk = before.shape[-1]
    m = umask3[..., :nk]
    d = float(np.max(np.abs(before[m] - after[m])))
    print(f"  [self-check B] baro_dump_u_before vs baro_dump_u_after (wet u-pts) "
          f"max|diff|={d:.3e}  (want > 0 -- the correction must be a REAL, "
          f"nonzero adjustment, not a silent no-op)")
    if d == 0.0:
        raise AssertionError(
            "self-check B FAILED: mlf_baro_corr's correction measured as "
            "EXACTLY zero -- either the correction is a no-op for this state "
            "(unexpected for ln_dynspg_ts=T) or the two dumps are reading "
            "the same buffer twice; investigate before trusting baro_dump_"
            "*_after.bin as an independent Kaa source.")


# =============================================================================
# Main measurement: genuine forward bracket of dyn_atf_qco's u/v filter
# =============================================================================
def measure_atf_uv(st) -> dict:
    print("\n" + "=" * 78)
    print("ATF filter u/v (dyn_atf_qco, ln_dynadv_vec=T plain-velocity branch)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, mc = st["grid"], st["mc"]
    gamma = mc.asselin_gamma
    print(f"  gamma (rn_atfp) = {gamma}")

    lvl_kmm = time_level_for_dump("atf_dump_uu_before.bin")
    lvl_kaa = time_level_for_dump("baro_dump_u_after.bin")
    lvl_verify = time_level_for_dump("atf_dump_uu_after.bin")
    print(f"  time_level_for_dump: Kmm-source={lvl_kmm!r} Kaa-source={lvl_kaa!r} "
          f"verify-target={lvl_verify!r}  (NOTE: these registry tokens mean "
          "PRE-/POST-FILTER STAGE, not Nbb/Nnn/Naa -- see this file's module "
          "docstring TIME-LEVEL AUDIT section for the actual Nbb/Nnn/Naa "
          "identification of each source)")

    umask3 = np.asarray(grid.umask) > 0.5
    vmask3 = np.asarray(grid.vmask) > 0.5

    # Kmm (now, pre-filter): dyn_atf_qco's own entry snapshot.
    u_now = _load_haloed(os.path.join(RUN_DIR, "atf_dump_uu_before.bin"), jpi, jpj, hls)
    v_now = _load_haloed(os.path.join(RUN_DIR, "atf_dump_vv_before.bin"), jpi, jpj, hls)
    nk = u_now.shape[-1]

    # Kbb (before): restart's ub/vb -- genuinely independent of every ATF dump.
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    u_bef = np.asarray(bef.u)[..., :nk]
    v_bef = np.asarray(bef.v)[..., :nk]

    # Kaa (after, pre-filter): the barotropic-corrected value dyn_atf_qco
    # actually consumes (see module docstring for why NOT stp_dump_08_dynzdf).
    u_aft = _load_haloed(os.path.join(RUN_DIR, "baro_dump_u_after.bin"), jpi, jpj, hls)
    v_aft = _load_haloed(os.path.join(RUN_DIR, "baro_dump_v_after.bin"), jpi, jpj, hls)

    # NEMO's own filtered result -- the verification target.
    u_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_uu_after.bin"), jpi, jpj, hls)
    v_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_vv_after.bin"), jpi, jpj, hls)

    # dynatf_qco.F90:165-166 EXACTLY (plain-velocity branch, DINO's
    # ln_dynadv_vec=T -- no e3u/e3v weighting in this branch, see module
    # docstring "live-vs-static e3t" section):
    #   puu(Kmm) = puu(Kmm) + rn_atfp*(puu(Kbb) - 2*puu(Kmm) + puu(Kaa))
    u_f_lego = u_now + gamma * (u_bef - 2.0 * u_now + u_aft)
    v_f_lego = v_now + gamma * (v_bef - 2.0 * v_now + v_aft)

    umask_dump = umask3[..., :nk]
    vmask_dump = vmask3[..., :nk]

    _shift_scan("ATF filter u", u_f_lego, u_f_nemo, umask_dump)
    _shift_scan("ATF filter v", v_f_lego, v_f_nemo, vmask_dump)
    r_u = per_element_stats("ATF filter u", u_f_lego, u_f_nemo, umask_dump,
                             sign_changing=True)
    r_v = per_element_stats("ATF filter v", v_f_lego, v_f_nemo, vmask_dump,
                             sign_changing=True)

    # Per-level profile (Rule: ladder discriminator -- deepest-level
    # concentration implicates the depth ladder; flat refutes it).
    print("  [per-level profile] err_norm=|d|/RMS(nemo) at each level, u-component:")
    rms_all = float(np.sqrt(np.mean(u_f_nemo[umask_dump] ** 2)))
    for k in range(nk):
        mk = umask_dump[..., k]
        if mk.sum() == 0:
            continue
        d = np.abs(u_f_lego[..., k][mk] - u_f_nemo[..., k][mk])
        print(f"    k={k:2d}  n={int(mk.sum()):5d}  "
              f"err_norm median={float(np.median(d))/max(rms_all,1e-30):.3e}  "
              f"max={float(np.max(d))/max(rms_all,1e-30):.3e}")

    return dict(u=r_u, v=r_v)


def main() -> None:
    print(_dump_lane.banner())
    st = build_state()
    self_check_a_baro_before_matches_dynzdf(st)
    self_check_b_baro_before_differs_from_after(st)
    r = measure_atf_uv(st)
    print("\n" + "=" * 78)
    print("SUMMARY (measurement only -- fidelity_bar_gate.py is NOT edited by "
          "this script; report these numbers for the separate transcribe+"
          "test+review task if they differ from the recorded 0.999969/0.995600 "
          "(u) and 0.999999/1.000300 (v)):")
    print(f"  ATF filter u: n={r['u']['n']}")
    print(f"  ATF filter v: n={r['v']['n']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
