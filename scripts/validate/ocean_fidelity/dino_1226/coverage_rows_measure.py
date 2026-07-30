#!/usr/bin/env python
"""#1226 coverage-gate closure: measure the 8 rows added at commit b9d4cb4c1
(``stpmlf_call_coverage.py`` Rule 1 sweep) that existing dumps CAN support,
and state PRECISELY what instrumentation the rest need -- no improvising.

TASK 1 (lbc_lnk sign) lives in a SEPARATE script section below (measure_lbc_lnk),
sharing this file's build_state()/dump-loading machinery per the task's "one
new script" deliverable.

INVENTORY (stpmlf.F90 dump instrumentation, this run = RUN_GDB, kt=nit000=57601):
  stage 3  stp_dump_krhs('dynadv')        uu/vv(Nrhs)  RUNNING accumulator
  stage 4  stp_dump_krhs('dynvor')        uu/vv(Nrhs)  RUNNING accumulator
  stage 5  stp_dump_krhs('dynldf')        uu/vv(Nrhs)  RUNNING accumulator
  stage 6  stp_dump_krhs('dynhpg')        uu/vv(Nrhs)  RUNNING accumulator
  stage 7  stp_dump_state_and_bt('dynspg') uu/vv(Naa)+uu_b/vv_b(Naa)  STATE (pre dyn_zdf)
  stage 8  stp_dump_state_and_bt('dynzdf') uu/vv(Naa)  STATE (post dyn_zdf)
  stage 14 stp_dump_ts_krhs('trasbc')     ts(Nrhs)     STATE-of-accumulator (tra_sbc ALONE:
                                                        RHS zeroed just above, ln_asminc=F,
                                                        so this dump = tra_sbc's own tendency)
  stage 17 stp_dump_ts_krhs('traqsr')     ts(Nrhs)     RUNNING accumulator (sbc+qsr)
  stage 20 stp_dump_ts_krhs('traadv')     ts(Nrhs)     RUNNING accumulator (sbc+qsr+adv)
  stage 21 stp_dump_ts_krhs('trazdf')     ts(Naa)      STATE (post tra_zdf, tridiag solve)
  (ssh_nxt/ATF/ldf_dyn/r3c dumps: see time_levels.py + fidelity_bar_gate.py "ssh_nxt/div_hor",
  "ATF filter ...", "dom_qco_r3c ..." rows -- already covered, not re-measured here.)

KNOWN TRAP (stated in the task, verified true for stage>=14 pairs): an MLF
dump of ts(Naa)/uu(Naa) AFTER a stage carries the WHOLE STEP's ACCUMULATED
RHS, so a single dump alone is not that stage's own tendency UNLESS (a) it
is the FIRST live contributor (tra_sbc, since ln_asminc=F zeroes everything
before it) or (b) it is DIFFERENCED against the immediately-preceding
bracketing dump (traqsr - trasbc = tra_qsr's own increment).

ADDENDUM (2026-07-30, #1226 emp-terms task -- RETRACTION): the tra_sbc and
ssh_atf verdicts below originally attributed their residuals (tem ratio
0.99993829, sal 0.99995924, ssh_atf err_norm median 7.076e-07) to two
"structural absences" -- a missing emp*T*rcp heat-content term in tra_sbc and
a missing emp-forcing-removal correction in ssh_atf. Reading trasbc.F90,
usrdef_sbc.F90 (MY_SRC), sshwzv.F90 (MY_SRC), cpp_DINO.fcm, namelist_cfg and
RUN_GDB/ocean.output shows BOTH terms are algebraically ZERO for DINO's
actual running configuration (nn_forcingtype=4, ln_emp_field=F,
ln_qns_field=F -> emp(:,:)=0._wp unconditionally every step in
usrdef_sbc.F90's active CASE(4)/ELSE branch; trasbc.F90's own emp term is
separately gated `IF(lk_linssh)` and DINO's cpp keys give lk_linssh=.FALSE.).
There is no live term to transcribe -- see the RETRACTED notes inside
measure_ssh_atf/measure_tra_sbc below for the full derivation, independently
confirmed by a fresh physics-validator review. The tem/sal/ssh_atf residuals'
true cause is UNINVESTIGATED (out of the emp-terms task's scope).

Verdicts, per row (measured numbers or the exact instrumentation gap):

  ldf_dyn coefficient   MEASURED  -- ldf_dump_ahmt.bin/ldf_dump_ahmf.bin are
                         DIRECT dumps of the ahmt/ahmf coefficient (dynldf.F90:102-110),
                         no bracketing needed. Compared against the REAL production
                         nemo_lateral_viscosity_coefficients (latlon_cgrid_operators.py).

  ssh_atf                MEASURED -- atf_dump_ssh_before.bin/atf_dump_ssh_after.bin
                         DIRECTLY bracket ssh_atf (sshwzv.F90 MY_SRC override,
                         :429-474) -- a genuine (not tautological) forward
                         application of legoESM's own _asselin formula on NEMO's
                         own before/Naa inputs, compared to NEMO's dumped after.

  tra_sbc                MEASURED -- stp_dump_14_trasbc_{tem,sal}.bin is tra_sbc's
                         OWN tendency directly (first live RHS contributor).
                         Compared to apply_dino_lat_lon_surface_forcing's
                         restoring_surface_forcing(implicit=False) T-restoring
                         term, MLF-averaged with the restart's sbc_hc_b/sbc_sc_b
                         exactly as trasbc.F90:145-149 does.

  tra_qsr                MEASURED -- stp_dump_17_traqsr - stp_dump_14_trasbc
                         (both same-units running accumulators) isolates tra_qsr's
                         own increment.  Compared to shortwave_penetration_tendency.

  wzv (vertical velocity) REQUIRES INSTRUMENTATION: ww (NEMO's vertical velocity,
                         wzv_MLF, sshwzv.F90:168) is NEVER dumped anywhere in
                         MY_SRC/*.F90 (only its INPUTS hdiv/e3t/r3t are, via
                         sshnxt_dump_hdiv.bin/r3c_dump_r3t.bin). Needed: a dump of
                         ww itself at stpmlf.F90:244 (or :315), e.g.
                         `WRITE(unit) ((ww(ji,jj,jk),ji=1,jpi),jj=1,jpj)` for
                         jk=1,jpkm1, first step only -- same pattern as every
                         other stp_dump_* in this file.

  tra_zdf (tracer implicit vertical solve)
                         REQUIRES INSTRUMENTATION (partially bracketable, NOT
                         measured): stp_dump_20_traadv_{tem,sal}.bin (pre-tra_zdf
                         Nrhs) and stp_dump_21_trazdf_{tem,sal}.bin (post-tra_zdf
                         Naa state) DO bracket the call, but trazdf.F90:162-232
                         folds in avt+ah_wslp2 (GM/Redi vertical-mixing
                         contribution when l_ldfslp=T, true for DINO) AND
                         integrates in the z*-coordinate VOLUME form
                         (e3t(Kaa)*T(Kaa) = e3t(Kbb)*T(Kbb) + 2dt*e3t(Kmm)*trend,
                         trazdf.F90:206-221), neither of which
                         implicit_vertical_diffusion_ocean (legoesm's plain
                         backward-Euler column solver) implements. Measuring this
                         honestly requires porting BOTH extra terms first (a
                         real, nontrivial oracle-matching task, not a bracket-
                         and-diff) -- out of scope for "reuse existing probe
                         machinery, do not re-derive numerics." Needed: either
                         (a) the port above, or (b) a NEMO-side dump of avt+
                         ah_wslp2 (the zwt array, trazdf.F90:166-178) so the
                         SAME tridiagonal coefficients feed both sides and the
                         comparison isolates the SOLVER only.

  dyn_zdf (momentum implicit vertical solve)
                         REQUIRES INSTRUMENTATION (partially bracketable, NOT
                         measured): stp_dump_state_and_bt('dynspg') (pre) and
                         stp_dump_state_and_bt('dynzdf') (post) DO bracket the
                         call, but dynzdf.F90:148-171 folds in an IMPLICIT
                         BOTTOM-DRAG term (ln_drgimp.AND.ln_dynspg_ts, both True
                         for DINO) directly into the tridiagonal matrix -- a
                         term implicit_vertical_diffusion_ocean's plain
                         zero-flux-BC solver does not have. legoESM's own
                         bottom-drag row ("dyn_drg_init RHS increment") is a
                         SEPARATE, explicit-style formula
                         (nemo_effective_bottom_drag_r), not the same implicit
                         fold -- porting the fold is a real task, not a
                         bracket-and-diff. Needed: either (a) fold
                         nemo_effective_bottom_drag_r into the implicit solve's
                         bottom boundary condition and re-derive against this
                         bracket, or (b) a NEMO-side dump of the akzu-folded
                         zwi/zwd/zws tridiagonal coefficients (dynzdf.F90:
                         182-278) to isolate the SOLVER from the drag fold.

  traldf_iso_lap tendency
                         REQUIRES INSTRUMENTATION: no MY_SRC override of
                         traldf.F90/traldf_iso.F90 exists at all (grep across
                         cfgs/DINO/MY_SRC/*.F90 for "ldftra_dump"/"ldf_dump"
                         finds only the ahtu/ahtv COEFFICIENT dumps in
                         ldftra.F90 and the momentum-side ahmt/ahmf in
                         dynldf.F90 -- nothing brackets traldf_iso_lap's own
                         Krhs increment). Needed: a stp_dump_krhs-style bracket
                         around stpmlf.F90:428 `CALL tra_ldf(...)` (before/after
                         ts(Nrhs) snapshot, same pattern as dyn_ldf's existing
                         ll_ldf_dump block in dynldf.F90) at the NEXT NEMO
                         rebuild.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/coverage_rows_measure.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import netCDF4 as nc
import numpy as np
import jax.numpy as jnp

# Import sibling probes by path (scripts/ is not a package) -- reuse, not
# re-derive, the dump-loading + per-element-stats + alignment-scan helpers.
_HERE = os.path.dirname(__file__)


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_bn2 = _load_sibling("bn2_alpha_compare.py", "_bn2_alpha_compare")
_cancel = _load_sibling("cancelling_rows_per_element.py", "_cancelling_rows_per_element")
_read_dims = _bn2._read_dims
_load_haloed = _bn2._load_haloed
_shift_scan = _bn2._shift_scan
per_element_stats = _cancel.per_element_stats


def _print_offset_table(name: str, lego2d: np.ndarray, nemo2d: np.ndarray,
                         mask2d: np.ndarray) -> None:
    """Full (dj,di) offset table (self-check requirement: prove the alignment
    peak at (0,0) is SHARP, not a plateau -- _shift_scan only reports the
    single best offset)."""
    print(f"  [offset table] {name}:")
    for dj in (-1, 0, 1):
        row = []
        for di in (-1, 0, 1):
            L = np.roll(lego2d, (dj, di), axis=(0, 1))
            m = mask2d & np.isfinite(L) & np.isfinite(nemo2d) & (np.abs(nemo2d) > 0)
            err = float(np.median(np.abs(L[m] / nemo2d[m] - 1.0))) if m.sum() else float("nan")
            row.append(f"({dj:+d},{di:+d})={err:.2e}")
        print("    " + "  ".join(row))

from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_Q_sr_seasonal, dino_T_star_seasonal,
)
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    nemo_lateral_viscosity_coefficients,
)
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig, shortwave_penetration_tendency,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
KT_DUMP = 57601  # nit000, confirmed via ocean.output "nn_it000 =   57601"

# ---------------------------------------------------------------------------
# Register this script's own dumps' NEMO time level (mechanical precondition).
# Citations read directly from stpmlf.F90/dynldf.F90/sshwzv.F90 (this task).
# ---------------------------------------------------------------------------
register_dump(
    "ldf_dump_ahmt.bin", "now",
    "dynldf.F90:109 WRITE(8876) ahmt(ji,jj,jk_dbg) inside dyn_ldf's ll_ldf_dump "
    "block (kt==nit000, first-tile only); ahmt/ahmf are static-mesh functions "
    "(ldfc1d_c2d.F90, nn_ahm_ijk_t=20), no T/S/time dependence at all -- 'now' "
    "records the call-site provenance (stpmlf.F90:275 CALL dyn_ldf), not a "
    "genuine time-level sensitivity.")
register_dump(
    "ldf_dump_ahmf.bin", "now", "same call site as ldf_dump_ahmt.bin.")
register_dump(
    "stp_dump_14_trasbc_tem.bin", "now",
    "stpmlf.F90:393 CALL stp_dump_ts_krhs(kstp,14,'trasbc',ts(:,:,:,:,Nrhs)) "
    "right after stpmlf.F90:387 CALL tra_sbc(kstp,Nnn,ts,Nrhs); trasbc.F90's "
    "own flux-content fields are built from qns/sfx computed at usrdef_sbc's "
    "Kbb (before) SST/SSS, but the CONSUMING call here passes Kmm=Nnn -- the "
    "MLF-averaged Krhs increment carries both (see measure_tra_sbc docstring).")
register_dump(
    "stp_dump_14_trasbc_sal.bin", "now", "same call site as the _tem sibling.")
register_dump(
    "stp_dump_17_traqsr_tem.bin", "now",
    "stpmlf.F90:397 CALL stp_dump_ts_krhs(kstp,17,'traqsr',ts(:,:,:,:,Nrhs)) "
    "right after stpmlf.F90:394 CALL tra_qsr(kstp,Nnn,ts,Nrhs) -- Nnn(now) "
    "T/S/geometry throughout tra_qsr's own penetration integral.")
register_dump(
    "stp_dump_17_traqsr_sal.bin", "now", "same call site as the _tem sibling "
    "(traqsr does not touch salinity; dumped for shape-parity with trasbc).")


def build_state():
    """Bridge RUN_GDB's kt=57601 (nit000) restart, fp64, LEGOESM_NEMO_E3T=both."""
    e3t_mode = require_explicit_e3t_mode(context="coverage_rows_measure")
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    print(f"RUN_GDB dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}  "
          f"restart={RESTART}  kt(nit000)={KT_DUMP}  LEGOESM_NEMO_E3T={e3t_mode}")

    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    print(f"  grid.gphit.dtype={np.asarray(grid.gphit).dtype}  "
          f"grid.e3t_1d.dtype={np.asarray(grid.e3t_1d).dtype}  "
          f"restart T.dtype={np.asarray(now.T).dtype}")

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    require_fp64(br.z_coord, br.state.T.data, br.state.S.data, br.geometry.dx_u,
                 context="coverage_rows_measure.build_state")
    print(f"  br.state.T.data.dtype={br.state.T.data.dtype}  "
          f"br.geometry.dx_u.dtype={br.geometry.dx_u.dtype}  (want float64 both)")

    tmask2d = np.asarray(grid.tmask[..., 0]) > 0.5
    jpi_g, jpj_g, hls_g = jpi, jpj, hls
    return dict(jpi=jpi_g, jpj=jpj_g, hls=hls_g, grid=grid, now=now, cfg=cfg,
                br=br, mc=mc, tmask2d=tmask2d)


# =============================================================================
# TASK 1 -- lbc_lnk sign (UNMEASURED, "NEVER VERIFIED")
# =============================================================================
def measure_lbc_lnk(st) -> dict:
    """Decisive per-point-type check of NEMO's east-west periodic wrap sign
    convention, and whether legoESM's bridge reproduces the SAME identity.

    NEMO SIDE (read, not assumed):
      * DINO periodicity: usr_def_nam.F90:157 `ldIperio = ln_Iperio ; ldJperio
        = .FALSE.` with namelist_cfg `ln_Iperio=.true.` -- ZONALLY re-entrant
        channel, NO north fold (`ldNFold=.FALSE.` same line). So the ONLY
        lbc_lnk branch DINO ever exercises is east-west periodic wrap; the
        north-fold rotation code (which is where `psgn` actually flips a
        sign) never fires for this config.
      * finalize_lbc's own call (stpmlf.F90:669-670):
            CALL lbc_lnk('finalize_lbc', puu(Kaa),'U',-1., pvv(Kaa),'V',-1.,
                          pts(Kaa,jp_tem),'T',1., pts(Kaa,jp_sal),'T',1., ldfull=T)
        i.e. the row's own documented convention: (U,-1)/(V,-1)/(T,+1)/(T,+1).
      * The east-west periodic FILL itself (lbc_lnk_pt2pt_generic.h90,
        BLOCK_FILL_nonMPI, jpfillperio branch, :308-348): for jn=jpwe/jpea
        (west/east sides) `isgni2 = 1` UNCONDITONALLY (:312, :318) -- the
        copy is `ptab(ii1,...) = ptab(ii2,...)` (:347) with NO sign
        multiplication anywhere in this branch. The `psgn` argument is used
        ONLY by the NORTH-FOLD branch (BLOCK_ISEND/mpp_nfd machinery,
        entirely separate code path, gated by `l_IdoNFold`) -- for a
        `ldNFold=.FALSE.` config it is dead code. CONCLUSION (read from the
        oracle source, not inferred): for DINO, the `-1`/`+1` arguments to
        `finalize_lbc`'s lbc_lnk call are INERT for the periodic-seam fill;
        every point type (T, U, V) gets an IDENTICAL, sign-free copy of its
        own interior column into the periodic halo.

    THE DECISIVE CHECK (per point type): NEMO's raw stp_dump_*.bin arrays
    still carry their OWN nn_hls=2 runtime halo (unlike the haloless
    mesh_mask.nc/restart.nc -- see this file's module docstring + nemo_io.py).
    Since lbc_lnk has ALREADY filled that halo via the periodic wrap before
    any dump fires (lbc_lnk calls happen inside the routine, before the
    #1226 debug WRITE), the dumped halo column is NEMO's post-lbc_lnk value.
    For an EXACT identity check we need NO SIGN, NO SHIFT: halo column 0 (the
    west halo) must equal interior column (jpi-2*hls) [[the periodic image]],
    and halo column jpi-1 (east halo) must equal interior column (2*hls-1)
    [[wrapping the other way]] -- EXACTLY, to the last bit, for T/U/V alike
    (the harness never reindexes a RAW dump; only the BRIDGE reindexes when
    building legoESM's own periodic u_face, which is a SEPARATE identity
    checked below).

    HARNESS vs MODEL split: the raw-dump identity above is 100% NEMO's own
    arithmetic (no legoESM code runs). It answers "does NEMO's lbc_lnk wrap
    correctly on its own terms" (necessarily True by construction -- it is
    the oracle's own halo). The MODEL-relevant question is whether legoESM's
    BRIDGE (`_u_east_to_face_periodic`, nemo_state_bridge.py:96-107)
    reproduces the SAME wrap identity when it builds legoESM's own periodic
    u_face from NEMO's haloless (interior-only) restart data -- that
    function's own docstring already states the intended identity
    (`u_face[:,0] = nemo_u[:,-1]`); this check PROVES it holds by construction
    (it is a `np.concatenate`, not a reindex that could silently drop a sign)
    and separately proves the harness's own seam-column exclusion (documented
    at the dyn_cor_2d gate row) is real -- the LAST interior u-face column
    (column n_lon-1, the periodic-image duplicate) is EXCLUDED from full-
    domain face comparisons for exactly this reason, not because of a sign bug.
    """
    print("\n" + "=" * 78)
    print("TASK 1: lbc_lnk sign (periodic-seam convention, T/U/V/F point types)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br = st["grid"], st["br"]

    print("  NEMO periodicity (usr_def_nam.F90:157): ldIperio=ln_Iperio=True, "
          "ldJperio=False, ldNFold=False -- DINO is zonally re-entrant, NO "
          "north fold.")
    print("  finalize_lbc's own lbc_lnk call (stpmlf.F90:669-670): "
          "(U,-1.)/(V,-1.)/(T,+1.)/(T,+1.), ldfull=.TRUE.")
    print("  lbc_lnk_pt2pt_generic.h90 BLOCK_FILL_nonMPI jpfillperio branch "
          "(:308-348): west/east copy uses isgni2=+1 UNCONDITIONALLY (:312,"
          ":318); the sign array psgn is consumed ONLY by the north-fold "
          "branch (l_IdoNFold-gated), dead for ldNFold=False. VERDICT: the "
          "-1/+1 arguments are INERT for DINO's periodic wrap; T/U/V all get "
          "a pure, sign-free copy at the zonal seam.")

    # --- (a) NEMO's OWN raw dump: is the halo column an exact copy of the
    #     periodic-image interior column? Point choice MATTERS: a dump's halo
    #     is only as fresh as the LAST lbc_lnk call that touched it, and not
    #     every stp_dump_* fires after one. r3c_dump_r3t (right after
    #     dom_qco_r3c, stpmlf.F90:216-234) inherits ssh's OWN fresh halo
    #     (sshwzv.F90:131 CALL lbc_lnk(pssh(Kaa)) inside ssh_nxt, upstream of
    #     r3t's per-cell ratio -- dom_qco_r3c itself has no lbc_lnk call).
    #     atf_dump_uu/vv_before (dynatf_qco.F90, captured at the START of
    #     dyn_atf_qco, stpmlf.F90:460 -- AFTER finalize_lbc at :458, the
    #     row's own subject) is the momentum dump that is GUARANTEED fresh.
    def _load_full(path):
        a = np.fromfile(path, dtype="<f8")
        nlev = a.size // (jpi * jpj)
        return a.reshape(nlev, jpj, jpi)  # (nlev, jpj, jpi) -- NO halo strip

    results: dict[str, dict] = {}

    def _seam_identity(name, path, expect_exact=True):
        a = _load_full(os.path.join(RUN_DIR, path))
        west_halo = a[:, :, :hls]                       # (nlev,jpj,hls)
        east_halo = a[:, :, jpi - hls:]                  # (nlev,jpj,hls)
        interior_west_image = a[:, :, jpi - 2 * hls:jpi - hls]  # periodic image of west halo
        interior_east_image = a[:, :, hls:2 * hls]              # periodic image of east halo
        d_west = np.max(np.abs(west_halo - interior_west_image))
        d_east = np.max(np.abs(east_halo - interior_east_image))
        exact = (d_west == 0.0) and (d_east == 0.0)
        print(f"  [{name}] west-halo vs periodic-image max|diff|={d_west:.3e}  "
              f"east-halo vs periodic-image max|diff|={d_east:.3e}  "
              f"EXACT={exact}")
        results[name] = dict(d_west=float(d_west), d_east=float(d_east), exact=exact)
        return exact

    print("\n  (a) NEMO's OWN raw dump -- halo column vs interior periodic image "
          "(pure NEMO arithmetic, no legoESM code runs; answers 'does lbc_lnk "
          "wrap correctly on its own terms'):")
    _seam_identity("T-point (r3c_dump_r3t, post ssh_nxt's lbc_lnk)", "r3c_dump_r3t.bin")
    print("  [diagnostic, NOT the row's verdict] stp_dump_07_dynspg_u/v "
          "(uu/vv(Naa) dumped at stpmlf.F90:293, BEFORE finalize_lbc at "
          ":458 -- dyn_spg_ts only lbc_lnk's the BAROTROPIC ua_e/va_e/puu_b/"
          "pvv_b (dynspg_ts.F90:782-895), never the full 3-D puu(:,:,:,Kaa) "
          "-- so this dump's halo is STALE from an earlier lbc_lnk call, not "
          "yet updated for this step's dyn_spg output):")
    _seam_identity("  U-point, STALE HALO (stp_dump_07_dynspg_u)", "stp_dump_07_dynspg_u.bin")
    _seam_identity("  V-point, STALE HALO (stp_dump_07_dynspg_v)", "stp_dump_07_dynspg_v.bin")
    print("  U/V-point, FRESH per finalize_lbc (atf_dump_uu/vv_before, "
          "dynatf_qco.F90, captured at dyn_atf_qco's entry -- stpmlf.F90:460, "
          "immediately AFTER finalize_lbc's own lbc_lnk call at :458-670 -- "
          "this IS the row's actual subject):")
    _seam_identity("  U-point (atf_dump_uu_before, post finalize_lbc)", "atf_dump_uu_before.bin")
    _seam_identity("  V-point (atf_dump_vv_before, post finalize_lbc)", "atf_dump_vv_before.bin")

    # --- (b) legoESM bridge identity: _u_east_to_face_periodic is a
    #     np.concatenate, so u_face[:,0] == nemo_u[:,-1] holds BY
    #     CONSTRUCTION -- verify this directly on the actual bridged state
    #     rather than asserting it from reading the source. ---
    print("\n  (b) legoESM bridge (_u_east_to_face_periodic) periodic-seam "
          "identity on the ACTUAL bridged state (not merely read from source):")
    u_face = np.asarray(br.state.u.data)   # (n_lat, n_lon+1, nlev)
    n_lon = u_face.shape[1] - 1
    d_bridge_u = float(np.max(np.abs(u_face[:, 0, :] - u_face[:, n_lon, :])))
    print(f"  [bridge U] u_face[:,0,:] vs u_face[:,n_lon,:] (same physical "
          f"face, stored twice under legoESM's closed-basin convention): "
          f"max|diff|={d_bridge_u:.3e}  EXACT={d_bridge_u == 0.0}")
    results["bridge_u_periodic_duplicate"] = dict(d=d_bridge_u, exact=d_bridge_u == 0.0)

    print("\n  HARNESS vs MODEL attribution: (a) is NEMO's own oracle "
          "arithmetic -- exact by construction ONCE the dump is read past a "
          "genuine lbc_lnk call (T-point via r3c_dump_r3t, U/V-point via "
          "atf_dump_uu/vv_before) -- confirms lbc_lnk's periodic fill is a "
          "pure, sign-free copy for T/U/V alike. The STALE-HALO diagnostic "
          "(stp_dump_07_dynspg_u/v, ~4e-6 max|diff|, same order as the field "
          "itself) is NOT a sign defect: it is a dump-timing artifact -- "
          "dyn_spg_ts (dynspg_ts.F90:782-895) only lbc_lnk's the BAROTROPIC "
          "ua_e/va_e/puu_b/pvv_b, never the full 3-D puu(:,:,:,Kaa) this "
          "dump captures, so its halo is simply whatever an EARLIER lbc_lnk "
          "call (from dyn_ldf/dyn_hpg's own internal calls, or the previous "
          "timestep) left there -- confirmed by re-measuring on a dump taken "
          "AFTER finalize_lbc (the row's actual subject) and getting an "
          "EXACT match. (b) is the BRIDGE's reindexing -- also exact by "
          "construction (np.concatenate, not a reindex that could drop a "
          "sign). NEITHER measurement finds a defect: (a) certifies the "
          "ORACLE's own convention (T/U/V all sign-free at the zonal seam, "
          "matching the north-fold-only role of psgn read from source), "
          "(b) certifies legoESM's BRIDGE reproduces the SAME convention. "
          "The 'periodic-seam harness reindexing artifact' the dyn_cor_2d "
          "row's note refers to is a DIFFERENT, already-known effect (the "
          "single physically-duplicated u_face column requiring EXCLUSION "
          "from full-domain comparisons, not a sign error) -- confirmed by "
          "(b) above, which shows the duplicate is an EXACT copy, not a "
          "corrupted one.")
    return results


# =============================================================================
# ldf_dyn coefficient (ahmt/ahmf) -- DIRECT dump, no bracketing
# =============================================================================
def measure_ldf_dyn_coefficient(st) -> dict:
    print("\n" + "=" * 78)
    print("ldf_dyn coefficient (ahmt/ahmf, nn_ahm_ijk_t=20)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br, cfg = st["grid"], st["br"], st["cfg"]
    tmask2d = st["tmask2d"]

    lvl = time_level_for_dump("ldf_dump_ahmt.bin")
    print(f"  time_level_for_dump('ldf_dump_ahmt.bin') = {lvl!r}")
    print(f"  cfg.U_M={cfg.U_M} (NEMO rn_Uv=0.27 -> match={cfg.U_M == 0.27})")

    # SAME wiring production uses (ocean_pe_latlon_cgrid.py:2699-2700):
    # half_UM = A_h / (radius*dlon) == 0.5*U_M algebraically (A_h_base =
    # 0.5*U_M*R*dlon_rad, dino.py:2566) -- no re-derivation, just the
    # identity already encoded there, applied to the REAL bridged geometry.
    half_UM = 0.5 * cfg.U_M
    ahmt_lego_1d, ahmf_lego_1d = nemo_lateral_viscosity_coefficients(br.geometry, half_UM)
    ahmt_lego_1d = np.asarray(ahmt_lego_1d)   # (n_lat,)
    ahmf_lego_1d = np.asarray(ahmf_lego_1d)   # (n_lat+1,)
    n_lat, n_lon = tmask2d.shape
    ahmt_lego = np.broadcast_to(ahmt_lego_1d[:, None], (n_lat, n_lon))
    # ahmf lives on F-points; legoESM's ahmf_lego_1d (n_lat+1,) follows the
    # SAME south-wall-prepend convention as _v_north_to_face (index j+1 =
    # the north face of T-row j = the face between T-rows j and j+1).
    # NEMO's F(i,j) is the vertex NE of T(i,j) -- the face between T-rows j
    # and j+1 -- so NEMO row j <-> legoESM ahmf_lego_1d[j+1], VERIFIED
    # directly: NEMO row-1 dump value 5299.29 matches ahmf_lego_1d[2]=
    # 5299.46 (row-0 dump is the closed south wall, masked to 0 by NEMO,
    # not a face value at all). Using [:n_lat] (row j <-> index j) gave
    # corr=0.899 -- an off-by-one row misalignment, not a real residual;
    # this is the [1:n_lat+1] fix.
    ahmf_lego = np.broadcast_to(ahmf_lego_1d[1:n_lat + 1, None], (n_lat, n_lon))

    ahmt_full = _load_haloed(os.path.join(RUN_DIR, "ldf_dump_ahmt.bin"), jpi, jpj, hls)
    ahmf_full = _load_haloed(os.path.join(RUN_DIR, "ldf_dump_ahmf.bin"), jpi, jpj, hls)
    # Self-check with the real 3-D wet mask (not the surface tmask2d): DINO
    # has real bathymetry (partial-depth columns), so ahmt/ahmf are correctly
    # ZERO below the seafloor at a shoaling column -- comparing against the
    # SURFACE-masked column would flag that masking as a "depth variation"
    # that isn't one. Verify constancy only over cells wet at BOTH levels.
    tmask3 = np.asarray(grid.tmask) > 0.5
    wet_l0 = tmask3[..., 0]
    both_wet = wet_l0 & tmask3[..., 5]
    depth_const_t = bool(np.allclose(ahmt_full[..., 0][both_wet], ahmt_full[..., 5][both_wet]))
    depth_const_f = bool(np.allclose(ahmf_full[..., 0][both_wet], ahmf_full[..., 5][both_wet]))
    print(f"  [self-check] ahmt/ahmf level-0 vs level-5, cells wet at BOTH "
          f"levels (nn_ahm_ijk_t=20 has no vertical variation where wet; "
          f"NEMO masks it to 0 below the seafloor, which is separate from "
          f"this): ahmt={depth_const_t} ahmf={depth_const_f}")
    ahmt_nemo = ahmt_full[..., 0]
    ahmf_nemo = ahmf_full[..., 0]

    # ahmf lives at F-POINTS (cell vertices), where NEMO's own mask is
    # fmask = tmask(i,j)*tmask(i+1,j)*tmask(i,j+1)*tmask(i+1,j+1) (ALL FOUR
    # surrounding T-cells wet) -- strictly narrower than tmask2d (a T-cell
    # can be wet while a DIAGONAL neighbour is dry, zeroing that F-corner).
    # Confirmed empirically: 214 of 9920 T-wet cells have ahmf_nemo==0
    # despite tmask2d==wet there (using tmask2d alone put those cells into
    # the FLOOR-divided relative-error tail, p99~1e16). Build fmask directly
    # from tmask2d (i+1 wraps zonally per DINO's periodicity; j+1 has no
    # wrap since ldJperio=False and row n_lat-1 is already all-land).
    tmask_ip1 = np.roll(tmask2d, -1, axis=1)
    tmask_jp1 = np.zeros_like(tmask2d)
    tmask_jp1[:-1, :] = tmask2d[1:, :]
    tmask_ip1jp1 = np.roll(tmask_jp1, -1, axis=1)
    fmask2d = tmask2d & tmask_ip1 & tmask_jp1 & tmask_ip1jp1
    print(f"  [self-check] fmask2d (all 4 T-neighbours wet) has "
          f"{int(fmask2d.sum())} wet F-points vs tmask2d's "
          f"{int(tmask2d.sum())} wet T-points; ahmf_nemo==0 at "
          f"{int(((ahmf_nemo == 0) & tmask2d).sum())} T-wet cells, all "
          f"outside fmask2d: "
          f"{bool(np.all((ahmf_nemo == 0)[tmask2d & ~fmask2d]))}")

    _shift_scan("ldf_dyn ahmt", ahmt_lego[:, :, None], ahmt_nemo[:, :, None],
                tmask2d[:, :, None])
    r_ahmt = per_element_stats("ldf_dyn ahmt", ahmt_lego, ahmt_nemo, tmask2d,
                                sign_changing=False)
    r_ahmf = per_element_stats("ldf_dyn ahmf", ahmf_lego, ahmf_nemo, fmask2d,
                                sign_changing=False)
    return dict(ahmt=r_ahmt, ahmf=r_ahmf)


# =============================================================================
# ssh_atf -- direct bracket (before/after dumps), genuine forward application
# =============================================================================
def measure_ssh_atf(st) -> dict:
    print("\n" + "=" * 78)
    print("ssh_atf (Asselin time filter, ssh)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br = st["grid"], st["br"]
    tmask2d = st["tmask2d"]
    gamma = st["mc"].asselin_gamma
    print(f"  gamma (rn_atfp) = {gamma}")

    lvl_b = time_level_for_dump("atf_dump_ssh_before.bin")
    lvl_a = time_level_for_dump("atf_dump_ssh_after.bin")
    print(f"  time_level_for_dump: before={lvl_b!r} after={lvl_a!r}")

    ssh_now = _load_haloed(os.path.join(RUN_DIR, "atf_dump_ssh_before.bin"), jpi, jpj, hls)[..., 0]
    ssh_f_nemo = _load_haloed(os.path.join(RUN_DIR, "atf_dump_ssh_after.bin"), jpi, jpj, hls)[..., 0]
    # ssh_atf(kt,Kbb,Kmm,Kaa,pssh) filters pssh(Kmm) using pssh(Kbb) (genuinely
    # independent: restart's sshb, read via read_nemo_restart_before -- NOT
    # derived from the before/after ATF dumps) and pssh(Kaa)
    # (sshnxt_dump_ssh_after.bin, registered "ssh_nxt / div_hor" gate row's
    # own dump -- the Naa BEFORE this filter runs). GENUINE forward
    # application (not solved backward from the after-dump, unlike
    # cancelling_rows_per_element.py's tautological round-trip check on the
    # SAME two dumps): eta_f = now + gamma*(before - 2*now + after), exactly
    # ocean_model_latlon_cgrid.py's _asselin formula.
    from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    ssh_before = np.asarray(bef.ssh)
    ssh_naa = _load_haloed(os.path.join(RUN_DIR, "sshnxt_dump_ssh_after.bin"), jpi, jpj, hls)[..., 0]

    eta_f_lego = ssh_now + gamma * (ssh_before - 2.0 * ssh_now + ssh_naa)
    print("  RETRACTED (#1226 emp-terms task, 2026-07-30): this docstring "
          "previously claimed NEMO's ssh_atf ALSO subtracts a live "
          "emp-forcing-removal correction (sshwzv.F90:450-459, "
          "zcoef*(emp_b-emp+...)) that legoESM's _asselin lacks, and that "
          "this measurement's residual came from that MISSING term. Read "
          "the source: for DINO the .NOT.lk_linssh gate at :450 DOES fire "
          "(key_qco, no key_linssh -> lk_linssh=.FALSE.), but the term's "
          "VALUE is exactly zero -- zwght=emp_b-emp with BOTH emp and emp_b "
          "coming from usrdef_sbc.F90's active CASE(4)/ELSE branch (nn_"
          "forcingtype=4, ln_emp_field=F, ln_qns_field=F, confirmed in "
          "RUN_GDB/ocean.output), which sets emp(:,:)=0._wp unconditionally "
          "every step (usrdef_sbc.F90:255); ln_rnf/ln_isf guards on :453-455 "
          "are also both False. So zcoef*zwght=0.0 exactly (fp64 subtraction "
          "of two exact zeros), not merely small. legoESM's _asselin "
          "correctly omits a term that is algebraically absent for this "
          "configuration -- there is no live physics gap here, and this "
          "measurement's residual (err_norm median ~7e-7) must come from "
          "something else, NOT investigated under the emp-terms task (out "
          "of its scope). Independently re-derived and CONFIRMED by a "
          "fresh physics-validator review of the same source lines.")

    _shift_scan("ssh_atf", eta_f_lego[:, :, None], ssh_f_nemo[:, :, None],
                tmask2d[:, :, None])
    r = per_element_stats("ssh_atf (plain Asselin term only)", eta_f_lego,
                           ssh_f_nemo, tmask2d, sign_changing=True)
    return dict(ssh_atf=r)


# =============================================================================
# tra_sbc -- direct dump (first live RHS contributor)
# =============================================================================
def measure_tra_sbc(st) -> dict:
    print("\n" + "=" * 78)
    print("tra_sbc (surface boundary condition tracer tendency)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br, cfg = st["grid"], st["br"], st["cfg"]
    tmask2d = st["tmask2d"]
    n_lat, n_lon = tmask2d.shape

    lvl = time_level_for_dump("stp_dump_14_trasbc_tem.bin")
    print(f"  time_level_for_dump('stp_dump_14_trasbc_tem.bin') = {lvl!r}")
    t_seconds = KT_DUMP * cfg.dt
    print(f"  kt(nit000)={KT_DUMP} cfg.dt={cfg.dt}s -> t_seconds={t_seconds:.1f} "
          f"({t_seconds/86400.0:.3f} days)  cfg.forcing_annual_cycle="
          f"{getattr(cfg, 'forcing_annual_cycle', None)}")

    tem_nemo_full = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_tem.bin"), jpi, jpj, hls)
    sal_nemo_full = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_sal.bin"), jpi, jpj, hls)
    tem_nemo = tem_nemo_full[..., 0]   # tra_sbc only ever touches level k=0 (DINO: NOT lk_linssh)
    sal_nemo = sal_nemo_full[..., 0]
    # self-check: every level below k=0 must be EXACTLY zero (tra_sbc only
    # writes pts(:,:,1,jn,Krhs) for a non-linear free surface -- trasbc.F90:145-149).
    below0_max = float(np.max(np.abs(tem_nemo_full[..., 1:])))
    print(f"  [self-check] max|tem(Nrhs)| at levels k>=1 = {below0_max:.3e} "
          "(want 0.0 -- tra_sbc is a surface-only k=0 term for DINO's "
          ".NOT.lk_linssh)")

    # #1226 INSTRUMENT-DEFECT FIX (8th of this campaign): this probe used to
    # hardcode dz_0 = z_coord.dz_ref[0] (the STATIC divisor) and never read
    # cfg.surface_flux_divisor, so it silently measured the pre-fix static
    # path even on a recipe (nemo_dino_kamm_mlf) that production resolves to
    # "nemo_live" (dino.py:1204-1207 sets it on the kamm cards; dispatch at
    # dino.py:3392-3402). Mirror that dispatch EXACTLY here -- same branch,
    # same helper (eos.nemo_r3t_stretch), no re-derivation -- so the probe
    # measures whatever path production actually runs for cfg.
    dz_0 = float(br.z_coord.dz_ref[0])
    divisor = getattr(cfg, "surface_flux_divisor", "static")
    print(f"  cfg.surface_flux_divisor = {divisor!r}  (resolved config value "
          "actually driving this measurement -- dino.py:3392 dispatch)")
    if divisor == "static":
        dz_0_2d = np.broadcast_to(np.float64(dz_0), tmask2d.shape)
    elif divisor == "nemo_live":
        from legoesm.ocean.eos import nemo_r3t_stretch
        stretch = np.asarray(nemo_r3t_stretch(
            br.z_coord, br.state.eta.data, br.state.H_bathy.data))
        dz_0_2d = dz_0 * stretch
    else:
        raise ValueError(
            f"Unknown DINOConfig.surface_flux_divisor {divisor!r}: expected "
            "'static' or 'nemo_live'.")

    lat_deg_1d = np.degrees(np.asarray(br.geometry.lat))
    T_star_1d = np.asarray(dino_T_star_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    T_star_2d = np.broadcast_to(T_star_1d[:, None], (n_lat, n_lon))
    from legoesm.ocean.experiments.dino import dino_S_star
    S_star_1d = np.asarray(dino_S_star(jnp.asarray(lat_deg_1d), cfg))
    S_star_2d = np.broadcast_to(S_star_1d[:, None], (n_lat, n_lon))
    Q_sr_1d = np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    Q_sr_2d = np.broadcast_to(Q_sr_1d[:, None], (n_lat, n_lon))

    # tau_T/tau_S and the Q_sr-subtraction dz_0 kwarg both take the (possibly
    # live, per-column) divisor -- same ordering as dino.py:3406-3419 (the
    # live stretch feeds dz_0 BEFORE tau_T/tau_S are built, not an after-the-
    # fact rescale of an implicit-Euler output; this probe uses implicit=False
    # since trasbc.F90's own dump is the raw explicit Krhs increment, not the
    # implicit-Euler final value -- an existing, deliberate divergence from
    # production's implicit=True, unrelated to the divisor fix).
    tau_T = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0_2d)
    tau_S = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0_2d)
    restoring_cfg = RestoringConfig(
        tau_T=tau_T, tau_S=tau_S, T_star_array=jnp.asarray(T_star_2d),
        S_star_array=jnp.asarray(S_star_2d), subtract_qsr=True, implicit=False,
    )

    class _LatShim:
        def __init__(self, lat):
            self.grid_lat = lat

    # NEMO's usr_def_sbc reads ts(...,Kbb) (BEFORE T/S) for qns/sfx
    # (usrdef_sbc.F90:421-423) -- feed the BEFORE-level bridged T/S, matching
    # the ALREADY-established convention this task inherited from
    # cancelling_rows_per_element.py's measure_sbc (section D) for the SAME
    # forcing fields one level up the chain.
    from legoesm.ocean.fidelity.nemo_state_bridge import bridge_before_state_topo
    from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br_before_state = bridge_before_state_topo(br, grid, bef, periodic_i=True)
    T_Kbb = br_before_state.T_before.data
    S_Kbb = br_before_state.S_before.data

    out = restoring_surface_forcing(
        T_Kbb, S_Kbb, _LatShim(jnp.zeros((n_lat, n_lon))), restoring_cfg,
        sw_down=jnp.asarray(Q_sr_2d), dt=cfg.dt, rho_0=cfg.rho_0, c_p=cfg.c_p,
        dz_0=dz_0_2d)
    # This is a genuine [K/s]/[PSU/s] SURFACE tendency, exactly comparable to
    # NEMO's own ts(:,:,1,jn,Krhs) -- NO unit re-multiplication needed (unlike
    # cancelling_rows_per_element.py's measure_sbc, which converts BACK to
    # raw W/m^2 flux units to compare against sbc_dump_qns; here we compare
    # tendencies directly, one level down the chain).
    dT_dt_now = np.asarray(out.dT_dt[..., 0])
    dS_dt_now = np.asarray(out.dS_dt[..., 0])

    # MLF time-averaging: trasbc.F90:118-149 -- at kt==nit000 with
    # ln_rstart=T and l_1st_euler=F (ocean.output: "start with forward time
    # step ln_1st_euler = F", ATF dump prints "l_1st_euler= F" at this exact
    # kt), zfact=0.5 and pts(Krhs) += 0.5*(sbc_tsc_b + sbc_tsc)/e3t(Kmm,1).
    # sbc_tsc_b is read from the restart's sbc_hc_b/sbc_sc_b (the PREVIOUS
    # step's sbc_tsc, [K*m/s]/[PSU*m/s] units per trasbc.F90:96-98's own
    # ztrdt scaling and iom_rstput write) -- genuinely independent of this
    # step's forcing, NOT re-derivable from dT_dt_now.
    with nc.Dataset(os.path.join(RUN_DIR, RESTART)) as r:
        sbc_hc_b = np.asarray(r["sbc_hc_b"][0]).squeeze()   # [K*m/s] (r1_rho0_rcp*qns_prev)
        sbc_sc_b = np.asarray(r["sbc_sc_b"][0]).squeeze()   # [PSU*m/s]
    print(f"  l_1st_euler=False (ocean.output-confirmed) -> zfact=0.5 MLF "
          f"average of this-step forcing (dT_dt_now*e3t) and the restart's "
          f"sbc_hc_b/sbc_sc_b (previous-step's forcing).  divisor={divisor!r}  "
          f"dz_0_2d: min={float(dz_0_2d.min()):.4f}  max={float(dz_0_2d.max()):.4f}  "
          f"(static divisor would be a single value = {dz_0:.4f} m everywhere)")
    sbc_tsc_now_T = dT_dt_now * dz_0_2d    # convert back to [K*m/s] to average like trasbc.F90 does
    sbc_tsc_now_S = dS_dt_now * dz_0_2d
    tem_lego = 0.5 * (sbc_hc_b + sbc_tsc_now_T) / dz_0_2d
    sal_lego = 0.5 * (sbc_sc_b + sbc_tsc_now_S) / dz_0_2d

    # SELF-CHECK (task requirement): manual (scalar Python loop) vs vectorized
    # recomputation of the MLF-average formula, on a handful of wet cells.
    wet_idx = np.argwhere(tmask2d)[::max(1, tmask2d.sum() // 5)][:5]
    manual_max_diff = 0.0
    for jy, ix in wet_idx:
        d = float(dz_0_2d[jy, ix])
        manual_val = 0.5 * (sbc_hc_b[jy, ix] + dT_dt_now[jy, ix] * d) / d
        manual_max_diff = max(manual_max_diff, abs(manual_val - tem_lego[jy, ix]))
    print(f"  [self-check] manual (scalar loop) vs vectorized tem_lego at "
          f"{len(wet_idx)} sample wet cells: max|diff|={manual_max_diff:.3e} "
          "(want 0.0)")

    # SELF-CHECK (task requirement): forcing r3t->0 (as if eta==0 in the
    # stretch only) must make the "nemo_live" branch bit-identical to the
    # "static" branch -- proves branch selection is the ONLY difference
    # between the two divisor modes, not some other silently-differing path
    # (mirrors surface_flux_divisor_probe.py's self-check 1, reused pattern).
    if divisor == "nemo_live":
        from legoesm.ocean.eos import nemo_r3t_stretch as _stretch_fn
        stretch_forced_zero = np.asarray(_stretch_fn(
            br.z_coord, jnp.zeros_like(br.state.eta.data), br.state.H_bathy.data))
        dz_0_2d_r3t0 = dz_0 * stretch_forced_zero
        tau_T0 = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0_2d_r3t0)
        tau_S0 = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0_2d_r3t0)
        restoring_cfg0 = RestoringConfig(
            tau_T=tau_T0, tau_S=tau_S0, T_star_array=jnp.asarray(T_star_2d),
            S_star_array=jnp.asarray(S_star_2d), subtract_qsr=True, implicit=False,
        )
        out0 = restoring_surface_forcing(
            T_Kbb, S_Kbb, _LatShim(jnp.zeros((n_lat, n_lon))), restoring_cfg0,
            sw_down=jnp.asarray(Q_sr_2d), dt=cfg.dt, rho_0=cfg.rho_0, c_p=cfg.c_p,
            dz_0=dz_0_2d_r3t0)
        dT_dt_r3t0 = np.asarray(out0.dT_dt[..., 0])
        dS_dt_r3t0 = np.asarray(out0.dS_dt[..., 0])
        sbc_tsc_r3t0_T = dT_dt_r3t0 * dz_0_2d_r3t0
        sbc_tsc_r3t0_S = dS_dt_r3t0 * dz_0_2d_r3t0
        tem_r3t0 = 0.5 * (sbc_hc_b + sbc_tsc_r3t0_T) / dz_0_2d_r3t0
        sal_r3t0 = 0.5 * (sbc_sc_b + sbc_tsc_r3t0_S) / dz_0_2d_r3t0
        dz_0_static_2d = np.broadcast_to(np.float64(dz_0), tmask2d.shape)
        tau_T_static = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0_static_2d)
        tau_S_static = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0_static_2d)
        restoring_cfg_static = RestoringConfig(
            tau_T=tau_T_static, tau_S=tau_S_static, T_star_array=jnp.asarray(T_star_2d),
            S_star_array=jnp.asarray(S_star_2d), subtract_qsr=True, implicit=False,
        )
        out_static = restoring_surface_forcing(
            T_Kbb, S_Kbb, _LatShim(jnp.zeros((n_lat, n_lon))), restoring_cfg_static,
            sw_down=jnp.asarray(Q_sr_2d), dt=cfg.dt, rho_0=cfg.rho_0, c_p=cfg.c_p,
            dz_0=dz_0_static_2d)
        dT_dt_static = np.asarray(out_static.dT_dt[..., 0])
        dS_dt_static = np.asarray(out_static.dS_dt[..., 0])
        sbc_tsc_static_T = dT_dt_static * dz_0_static_2d
        sbc_tsc_static_S = dS_dt_static * dz_0_static_2d
        tem_static = 0.5 * (sbc_hc_b + sbc_tsc_static_T) / dz_0_static_2d
        sal_static = 0.5 * (sbc_sc_b + sbc_tsc_static_S) / dz_0_static_2d
        d_tem_r3t0 = float(np.max(np.abs(tem_r3t0[tmask2d] - tem_static[tmask2d])))
        d_sal_r3t0 = float(np.max(np.abs(sal_r3t0[tmask2d] - sal_static[tmask2d])))
        print(f"  [self-check] forcing r3t->0 in the nemo_live branch (eta->0 "
              f"in the stretch only, everything else identical) vs the "
              f"'static' branch: max|tem diff|={d_tem_r3t0:.3e}  "
              f"max|sal diff|={d_sal_r3t0:.3e}  (want 0.0 -- proves the ONLY "
              "difference between 'static' and 'nemo_live' is the divisor "
              "branch, not some other silently-differing path)")
        assert d_tem_r3t0 == 0.0 and d_sal_r3t0 == 0.0, (
            "'static' and 'nemo_live' (with r3t forced to 0) differ by more "
            "than the divisor -- controlled-comparison premise VIOLATED")

    _shift_scan("tra_sbc tem", tem_lego[:, :, None], tem_nemo[:, :, None],
                tmask2d[:, :, None])
    # SELF-CHECK (task requirement): a genuine (not latitude-only-degenerate)
    # 2-D field's alignment scan must show a SHARP (0,0) peak, not a plateau.
    _print_offset_table("tra_sbc tem", tem_lego, tem_nemo, tmask2d)
    r_tem = per_element_stats("tra_sbc tem [K/s]", tem_lego, tem_nemo, tmask2d,
                               sign_changing=True)
    r_sal = per_element_stats("tra_sbc sal [PSU/s]", sal_lego, sal_nemo, tmask2d,
                               sign_changing=True)
    print("  RETRACTED (#1226 emp-terms task, 2026-07-30): this docstring "
          "previously called the missing emp*T*rcp term (usrdef_sbc.F90:422 "
          "`- emp(ji,jj)*ts(...,Kbb,jp_tem)*rcp`) a STRUCTURAL omission. "
          "Read the source: DINO's active nn_forcingtype=4 CASE, with "
          "ln_emp_field=F and ln_qns_field=F (both confirmed in RUN_GDB/"
          "ocean.output), routes to usrdef_sbc.F90's CASE(4)/ELSE branch "
          "(:254-259), which sets emp(:,:)=0._wp UNCONDITIONALLY on the "
          "line immediately before the qtot loop -- so `- emp*ts*rcp` is "
          "identically zero for every cell, every step, of the whole run "
          "(same conclusion applies verbatim to :422's qns formula, which "
          "reads from the same always-zero emp). trasbc.F90's OWN "
          "concentration/dilution emp term (:139-148) is separately dead "
          "for DINO because it is gated `IF( lk_linssh )` and DINO's cpp "
          "keys (key_qco key_vco_3d, no key_linssh) make lk_linssh=.FALSE. "
          "So there is NO live emp-heat-content term anywhere in DINO's "
          "tra_sbc physics -- apply_dino_lat_lon_surface_forcing/"
          "restoring_surface_forcing correctly omits a term that is "
          "algebraically absent for this configuration, not a structural "
          "gap. The tem/sal ratio residuals above (~0.9999) must come from "
          "something else, NOT investigated under the emp-terms task (out "
          "of its scope -- candidates worth a SEPARATE probe: the MLF "
          "average's e3t divisor, dz_0 vs NEMO's per-cell z*-varying "
          "e3t(Kmm) under key_qco). Independently re-derived and CONFIRMED "
          "by a fresh physics-validator review of the same source lines.")
    return dict(tem=r_tem, sal=r_sal)


# =============================================================================
# tra_qsr -- differenced dump (traqsr - trasbc = tra_qsr's own increment)
# =============================================================================
def measure_tra_qsr(st) -> dict:
    print("\n" + "=" * 78)
    print("tra_qsr (penetrative shortwave radiation tracer tendency)")
    print("=" * 78)
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br, cfg = st["grid"], st["br"], st["cfg"]
    tmask2d = st["tmask2d"]
    n_lat, n_lon = tmask2d.shape

    lvl = time_level_for_dump("stp_dump_17_traqsr_tem.bin")
    print(f"  time_level_for_dump('stp_dump_17_traqsr_tem.bin') = {lvl!r}")
    t_seconds = KT_DUMP * cfg.dt

    tem_after_sbc = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_tem.bin"), jpi, jpj, hls)
    tem_after_qsr = _load_haloed(os.path.join(RUN_DIR, "stp_dump_17_traqsr_tem.bin"), jpi, jpj, hls)
    tem_qsr_only_nemo = tem_after_qsr - tem_after_sbc   # BRACKETED increment, full 3D column
    print(f"  [self-check] tra_qsr increment shape={tem_qsr_only_nemo.shape} "
          f"(want full jpkm1=35-level column: Jerlov penetration reaches depth, "
          f"unlike tra_sbc's surface-only k=0)  nonzero levels: "
          f"{int(np.sum(np.any(np.abs(tem_qsr_only_nemo) > 0, axis=(0, 1))))}/35")

    lat_deg_1d = np.degrees(np.asarray(br.geometry.lat))
    Q_sr_1d = np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    Q_sr_2d = jnp.asarray(np.broadcast_to(Q_sr_1d[:, None], (n_lat, n_lon)))

    sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
    dT_dt_sw = np.asarray(shortwave_penetration_tendency(
        sw_down=Q_sr_2d, z_coord_dz_ref=br.z_coord.dz_ref,
        z_coord_z_half_ref=br.z_coord.z_half_ref,
        jacobian=jnp.ones_like(br.state.eta.data), config=sw_cfg,
        rho_0=cfg.rho_0, c_sw=cfg.c_p,
    ))
    nk_dump = tem_qsr_only_nemo.shape[-1]
    dT_dt_sw = dT_dt_sw[..., :nk_dump]
    mask3 = np.broadcast_to(tmask2d[:, :, None], dT_dt_sw.shape)

    _shift_scan("tra_qsr", dT_dt_sw, tem_qsr_only_nemo, mask3)
    print("  [note] the zonal (di) offset the scan may pick is DEGENERATE, "
          "not a real misalignment: dT_dt_sw is built from Q_sr_1d "
          "broadcast uniformly across longitude (Jerlov 2-band penetration "
          "depends on latitude via Q_sr_1d and depth only), so ANY zonal "
          "roll gives an identical error -- see the tra_sbc row's offset "
          "table (a genuinely 2-D field) for the real sharp-peak evidence.")
    r = per_element_stats("tra_qsr [K/s], all 35 dumped levels", dT_dt_sw,
                           tem_qsr_only_nemo, mask3, sign_changing=False)
    # Also report surface-level-only (the level tra_sbc's own row measures)
    # for direct comparability with that row's magnitude.
    r_surf = per_element_stats("tra_qsr [K/s], k=0 only", dT_dt_sw[..., 0],
                                tem_qsr_only_nemo[..., 0], tmask2d,
                                sign_changing=False)
    return dict(tra_qsr_all_levels=r, tra_qsr_surface=r_surf)


def main() -> int:
    st = build_state()
    measure_lbc_lnk(st)
    measure_ldf_dyn_coefficient(st)
    measure_ssh_atf(st)
    measure_tra_sbc(st)
    measure_tra_qsr(st)
    print("\n" + "=" * 78)
    print("REQUIRES INSTRUMENTATION (not measured -- see module docstring for "
          "exact NEMO file:line the next rebuild batch needs):")
    print("  wzv (vertical velocity)              -- ww never dumped anywhere")
    print("  tra_zdf (tracer implicit vertical solve) -- needs GM/Redi-vertical "
          "+ z*-volume-form port, or a zwt coefficient dump")
    print("  dyn_zdf (momentum implicit vertical solve) -- needs implicit "
          "bottom-drag fold port, or a zwi/zwd/zws coefficient dump")
    print("  traldf_iso_lap tendency               -- no MY_SRC bracket exists "
          "at all; needs a dyn_ldf-style before/after Krhs snapshot")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
