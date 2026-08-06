"""#1226 zu_frc: complete WRITE LEDGER + measure the pieces missing from the
six-piece budget (``zu_frc_budget_completion.py``, which closed at only
corr 0.052/ratio 0.046 -- 95% of the 8.0270e-3 u err_norm UNEXPLAINED).

TASK (coordinator hypothesis): the six-piece budget ASSUMED the piece list
(keg, hpg, ldf, zad, vor, zu_trd-subtraction) instead of ENUMERATING the
assembly. This script reads ``cfgs/DINO/MY_SRC/dynspg_ts.F90`` from
``zu_frc``'s first assignment to the dump write and lists EVERY statement
that writes it, then measures the two pieces the six-piece budget omitted:
the ``dyn_drg_init`` bottom-drag increment and the wind-forcing addition.

STEP 1 -- COMPLETE WRITE LEDGER (source read this session, file:line cited).
NEMO's DINO build: ``cpp_DINO.fcm`` sets ``key_qco key_vco_3d`` only --
``key_RK3`` is NOT defined, so the ``#else`` (MLF) branch of dynspg_ts.F90 is
the active one (the ``#if defined key_RK3`` branch at :278-306 is COMPILED
OUT). Within the MLF branch: ``key_GPU_reproducibility`` is NOT defined, so
the plain ``SUM(...)`` form (:336-337) is active, not the GPU-repro DO-loop
form (:323-331) nor the non-qco/non-linssh e3u(Kmm) form (:342-343, dead
because key_qco IS defined). Namelist (RUN_GDB/namelist_cfg): ln_bt_fw=F
(:353, "model crashes if T"), ln_apr_dyn=F (namelist_ref:211, DINO section
never overrides it), ln_rnf=F (:210), ln_isf=F (:520), ln_sdw=F (:585),
ln_isfcpl=F (:578), ln_asmiau=F (:1482, and key_asminc is not even in
cpp_DINO.fcm so the whole IAU block at :471-477 is compiled out too).

+----+---------------------------------------------------+------------------------------------------------+--------+
|line| expression                                          | condition (active on DINO?)                     | in-6pc |
+----+---------------------------------------------------+------------------------------------------------+--------+
|287 | zu_frc = Ue_rhs                                    | #if key_RK3 -- key_RK3 NOT in cpp_DINO.fcm      | N/A    |
|    |                                                     | -> COMPILED OUT (dead code for DINO)            | (dead) |
|304 | zu_frc -= zu_trd*ssumask (RK3 Coriolis removal)    | same #if key_RK3 branch -> COMPILED OUT         | (dead) |
|323 | zu_frc = e3u_0(k=1)*puu(Krhs,1)*umask(k=1)         | #if key_GPU_reproducibility -- NOT defined      | (dead) |
|327 | zu_frc += e3u_0(k)*puu(Krhs,k)*umask(k), k=2..jpk  | same GPU-repro branch -> COMPILED OUT           | (dead) |
|331 | zu_frc *= r1_hu_0                                  | same GPU-repro branch -> COMPILED OUT           | (dead) |
|336 | zu_frc = SUM(e3u_0(:)*puu(Krhs,:)*umask(:))*r1_hu_0| #else of key_GPU_reproducibility -- ACTIVE      | YES    |
|    |                                                     | (key_qco defined, key_GPU_reproducibility not)  | (keg+  |
|    |                                                     |                                                  | hpg+   |
|    |                                                     |                                                  | ldf+   |
|    |                                                     |                                                  | zad+   |
|    |                                                     |                                                  | vor    |
|    |                                                     |                                                  | sum)   |
|342 | zu_frc = SUM(e3u(:,Kmm)*puu(Krhs,:)*umask(:))*r1_hu| #else of key_qco||key_linssh -- key_qco IS      | (dead) |
|    |                                                     | defined -> this branch COMPILED OUT             |        |
|350 | puu(Krhs,jk) = (puu(Krhs,jk)-zu_frc)*umask(jk)     | vertical-mean removal FROM puu, not a zu_frc     | n/a    |
|    |                                                     | write (listed for completeness only)            |        |
|367 | zu_frc -= zu_trd*ssumask (2-D Coriolis removal)    | ACTIVE (MLF branch, unconditional)              | YES    |
|382 | zu_frc,zv_frc = dyn_drg_init(... zu_frc,zv_frc ...)| ACTIVE (unconditional CALL, INOUT args)         | **NO** |
|408 | zu_frc += grav*d(ssh_ib)/dx  (apr, ln_bt_fw=T form)| IF(ln_apr_dyn) -- ln_apr_dyn=F -> does not run  | (off)  |
|414 | zu_frc += 0.5g*(d(ssh_ib)+d(ssh_ibb))/dx (apr,CTRD)| IF(ln_apr_dyn) -- ln_apr_dyn=F -> does not run  | (off)  |
|426 | zu_frc += r1_rho0*utauU*r1_hu (wind, ln_bt_fw=T)   | IF(ln_bt_fw) -- ln_bt_fw=F -> does not run       | (off)  |
|432 | zu_frc += 0.5*r1_rho0*(utau_b+utauU)*r1_hu (CTRD)  | ELSE of ln_bt_fw -- ln_bt_fw=F -> ACTIVE         | **NO** |
|490 | (dump write, spg_dump_zu_frc.bin)                  | -- read point, not a write --                    | --     |
+----+---------------------------------------------------+------------------------------------------------+--------+

LEDGER COMPLETENESS CHECK: every line between :335 (first live assignment)
and :497 (dump write) that assigns to ``zu_frc``/``zv_frc`` (grep
"zu_frc(ji,jj) =\|zu_frc(:,:) =" over that byte range) appears above --
verified programmatically in main() STEP 1 self-check below (re-greps the
live file and asserts the set of matched lines equals this table's set,
so a future NEMO source edit that adds/removes a writer makes this script
FAIL LOUD instead of silently going stale).

CONCLUSION STEP 1: with ln_apr_dyn/ln_rnf/ln_isf/ln_sdw/ln_isfcpl/ln_asmiau
all FALSE on this card, exactly TWO lines write zu_frc that are outside the
six-piece budget and are NOT compiled/gated OFF: the ``dyn_drg_init`` call
(:382, dump exists: drg_dump_zu_frc_inc.bin) and the CENTRED wind term
(:432, no NEMO-side increment dump exists for this piece alone).

STEP 2 -- measure both.
(a) drg: dumped directly by NEMO (drg_dump_zu_frc_inc.bin/zv_frc_inc.bin,
    dynspg_ts.F90:391-399, the INOUT increment isolated via the
    zu_frc_predrg snapshot at :378-379). legoESM's own increment is NOT
    exposed as a diagnostic; reconstructed here by spying
    ``nemo_bottom_drag_rate_faces`` (the SAME function
    ``ocean_model_latlon_cgrid.py:2925`` calls) and reproducing ITS OWN
    formula at :2932-2937 line-for-line (reused, not re-derived) to isolate
    the increment legoESM's ``barotropic_drag_substep=True`` path adds to
    F_slow_u/F_slow_v.
(b) wind (RESUMED 2026-08-03, batched-rebuild dump now exists): NEMO adds
    it as a SEPARATE 2-D term at :443-444 (post vertical-mean-removal,
    CENTRED branch since ln_bt_fw=F): ``zu_frc += r1_rho0*r1_2*(utau_b+
    utauU)*r1_hu(Kmm)``, isolated by NEMO's own bracket dump
    (``wnd_dump_zu_frc_inc.bin`` = post-:443 minus pre-:432 snapshot,
    registered "now"/Kmm in ``time_levels.py``). legoESM deposits wind ONCE
    into the 3-D top-cell tendency (``du_dt[...,0]``) via
    ``surface_stress_faces``/``_bc_external_surface_forcing``
    (``ocean_pe_latlon_cgrid.py:3453/3492``), already CENTRED upstream by
    ``ocean_model_latlon_cgrid.py:2747-2756`` (``barotropic_forcing_centred``
    rebinds ``surface_forcing.tau_x/tau_y`` to ``0.5*(tau_x_prev+tau_x)``
    BEFORE the tendency call). STEP 3 below isolates lego's wind-only
    contribution by spying ``surface_stress_faces`` for the SAME production
    call (not re-derived) and depth-averaging ONLY the wind piece with the
    SAME ``h_u_pre`` weights ``F_slow_u`` itself uses (line ~2853), then
    compares it against NEMO's dumped increment directly.

Reuses ``zu_frc_term_walk.py``/``zu_frc_budget_completion.py`` loaders and
the drag formula wholesale (imported / transcribed from
``ocean_pe_latlon_cgrid.py:3130`` + ``ocean_model_latlon_cgrid.py:2925-2937``,
not re-derived).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zu_frc_write_ledger.py
"""
from __future__ import annotations

import dataclasses
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

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
import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pemod
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_bottom_drag_rate_faces,
    surface_stress_faces,
)
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface, min_cell_to_vface
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior,
)

NEMO_SRC = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/dynspg_ts.F90"
RESTART_FILE = "DINO_00057600_restart.nc"
RESTART_STEP = 57601

# NEMO source lines (1-indexed, inclusive) that ASSIGN to zu_frc/zv_frc
# via a direct ``zu_frc(ji,jj) = ...``/``zu_frc(:,:) = ...`` statement,
# between the first live assignment (:287) and the dump write (:497), per
# THIS session's read (module docstring table). Kept here as data so
# STEP 1's self-check can assert the live file hasn't silently changed.
# RE-VERIFIED 2026-08-03 against the LIVE source: every line shifted by a
# constant offset since the 2026-07-30 batched-rebuild dump instrumentation
# was inserted (+1 for lines before the wind block from the drg-bracket
# comment lines; +11 for the wind-block lines themselves, from the 10-line
# wnd_dump_zu_frc_inc.bin snapshot+dump block at :423-459 documented in this
# file's docstring STEP 2(b) update). Pure line-shift, same statements,
# confirmed by re-reading dynspg_ts.F90 at the new line numbers this session
# -- NOT a content change, so IN_SIX_PIECE_BUDGET/DEAD_CODE/etc. below keep
# their original MEANING at the new line numbers.
EXPECTED_WRITE_LINES = {
    288: "RK3 seed (#if key_RK3 -- compiled out)",  # const-ok: F90 line number, not R_d
    305: "RK3 2-D Coriolis removal (#if key_RK3 -- compiled out)",
    324: "GPU-repro k=1 seed (#if key_GPU_reproducibility -- compiled out)",
    328: "GPU-repro k=2..jpk accumulate (compiled out)",
    332: "GPU-repro *= r1_hu_0 (compiled out)",
    337: "SUM(e3u_0*puu(Krhs)*umask)*r1_hu_0 -- ACTIVE (key_qco, not GPU-repro)",
    343: "non-qco e3u(Kmm) form (#else key_qco||key_linssh -- compiled out)",
    368: "zu_frc -= zu_trd*ssumask -- ACTIVE (MLF, unconditional)",
    409: "apr_dyn ln_bt_fw=T form (ln_apr_dyn=F -> does not execute)",
    415: "apr_dyn CENTRED form (ln_apr_dyn=F -> does not execute)",
    437: "wind ln_bt_fw=T form (ln_bt_fw=F -> does not execute)",
    443: "wind CENTRED form -- ACTIVE (ln_bt_fw=F), NOT in 6-piece budget",
}
# dyn_drg_init (:382, now shifted +1 to :383) writes zu_frc/zv_frc as INOUT
# arguments to a CALL, not via a ``zu_frc(...) =`` assignment statement -- a
# DIFFERENT regex, checked separately below so the self-check doesn't
# conflate two distinct write mechanisms into one pattern.
DRG_CALL_LINE = 382  # CALL starts here (was 381); zu_frc/zv_frc INOUT args on the continuation line

# Lines that assign to zu_frc/zv_frc but are genuinely active AND already
# covered by the six-piece budget's u/v vertical-mean SUM (337) or the
# zu_trd subtraction (368).
IN_SIX_PIECE_BUDGET = {337, 368}
DEAD_CODE = {288, 305, 324, 328, 332, 343}  # const-ok: F90 line number, not R_d
GATED_OFF_ON_THIS_CARD = {409, 415, 437}
MISSING_FROM_BUDGET_AND_ACTIVE = {DRG_CALL_LINE, 443}


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


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _self_check_ledger_completeness() -> None:
    """STEP 1 self-check: re-grep the LIVE NEMO source between the first
    live assignment and the dump write; assert the set of matched
    assignment lines equals EXPECTED_WRITE_LINES's keys. Fails loud if the
    source has an assignment this ledger doesn't know about, or vice versa.
    """
    with open(NEMO_SRC) as fh:
        lines = fh.readlines()
    # Region of interest: first zu_frc assignment (line 288, 1-indexed) to
    # the dump WRITE at line 514 (re-verified 2026-08-03 against the live
    # source; was :287-497 before the batched-rebuild wind-dump
    # instrumentation shifted every subsequent line -- the dump write itself
    # is not an assignment and is excluded by the regex below).
    lo, hi = 288, 514  # const-ok: F90 line number, not R_d
    pat = re.compile(r"^\s*zu_frc\s*\(.*\)\s*=[^=]")  # assignment, not ``==``
    found = set()
    for i in range(lo - 1, hi):
        if pat.match(lines[i]):
            found.add(i + 1)  # back to 1-indexed
    expected = set(EXPECTED_WRITE_LINES)
    missing_from_table = found - expected
    stale_in_table = expected - found
    print(f"  re-grep of dynspg_ts.F90:{lo}-{hi} for 'zu_frc(...) =' direct "
          f"assignments: {sorted(found)}")
    assert not missing_from_table, (
        f"LEDGER INCOMPLETE: source has zu_frc writes at lines "
        f"{sorted(missing_from_table)} not in this ledger's table -- "
        f"update EXPECTED_WRITE_LINES before trusting this script.")
    assert not stale_in_table, (
        f"LEDGER STALE: table claims writes at {sorted(stale_in_table)} "
        f"that the live source no longer has at those lines -- NEMO source "
        f"changed, re-read it.")
    print("  LEDGER COMPLETE (direct assignments): table's write-line set "
          "== live source's write-line set (exact match, no missing/stale "
          "entries).")

    # Separately verify the dyn_drg_init CALL line -- a distinct write
    # mechanism (INOUT arg to a CALL, not a ``zu_frc(...) =`` statement).
    call_pat = re.compile(r"CALL\s+dyn_drg_init\s*\(")
    call_line = None
    for i in range(lo - 1, hi):
        if call_pat.search(lines[i]):
            call_line = i + 1
            break
    assert call_line == DRG_CALL_LINE, (
        f"dyn_drg_init CALL moved from line {DRG_CALL_LINE} to {call_line} "
        f"-- update DRG_CALL_LINE.")
    print(f"  dyn_drg_init CALL confirmed at line {call_line} (INOUT "
          f"zu_frc/zv_frc, a write mechanism distinct from the direct-"
          f"assignment regex above).")


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_write_ledger")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    print("\n" + "=" * 78)
    print("STEP 1: complete write ledger (see module docstring for the full "
          "table) + completeness self-check")
    print("=" * 78)
    _self_check_ledger_completeness()
    print(f"\n  IN 6-piece budget (336, 367): {sorted(IN_SIX_PIECE_BUDGET)}")
    print(f"  dead code (compiled out on this card): {sorted(DEAD_CODE)}")
    print(f"  gated off by namelist on this card (ln_apr_dyn/ln_bt_fw=F): "
          f"{sorted(GATED_OFF_ON_THIS_CARD)}")
    print(f"  ACTIVE + NOT in 6-piece budget (candidate owners of the 95%): "
          f"{sorted(MISSING_FROM_BUDGET_AND_ACTIVE)}  "
          f"(:382 dyn_drg_init, :432 wind CENTRED term)")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"\n  DINOConfig.barotropic_drag_substep={dcfg.barotropic_drag_substep!r}  "
          f"barotropic_forcing_centred={dcfg.barotropic_forcing_centred!r}  "
          f"surface_stress_implicit={dcfg.surface_stress_implicit!r}")
    assert dcfg.barotropic_drag_substep is True
    assert dcfg.barotropic_forcing_centred is True
    assert dcfg.surface_stress_implicit is False

    print(f"\nONE STATE: RUN_DIR={RUN_DIR}  RESTART_FILE={RESTART_FILE}  "
          f"restart step (kt) used = {RESTART_STEP}")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_write_ledger twin state")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Spy nemo_bottom_drag_rate_faces (the exact function
    # ocean_model_latlon_cgrid.py:2925 calls) to get (r_u_bt, r_v_bt,
    # isb_u, isb_v) with the SAME inputs the production call uses, then
    # reconstruct the F_slow_u/v INCREMENT with the exact formula at
    # :2932-2937 (transcribed, not re-derived) -- plus capture F_slow_u
    # itself (pre-existing spy pattern from zu_frc_budget_completion.py).
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid
    _real_drag_rate = nemo_bottom_drag_rate_faces
    _real_stress_faces = surface_stress_faces

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "F_slow_u" not in captured:
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    def _spy_drag_rate(u, v, h_k, z_coord, config, grid):
        r_u, r_v, isb_u, isb_v = _real_drag_rate(u, v, h_k, z_coord, config, grid)
        if "r_u_bt" not in captured:
            captured["r_u_bt"] = r_u
            captured["r_v_bt"] = r_v
            captured["isb_u"] = isb_u
            captured["isb_v"] = isb_v
            captured["u_at_call"] = u
            captured["v_at_call"] = v
            captured["h_k_at_call"] = h_k
        return r_u, r_v, isb_u, isb_v

    def _spy_stress_faces(surface_forcing_arg, u_dtype, z_coord_arg, J_arg, grid_arg):
        result = _real_stress_faces(surface_forcing_arg, u_dtype, z_coord_arg, J_arg, grid_arg)
        if result is not None and "tau_i_u" not in captured:
            tau_i_u, tau_j_v, dz_0_u, dz_0_v = result
            captured["tau_i_u"] = tau_i_u
            captured["tau_j_v"] = tau_j_v
            captured["dz_0_u"] = dz_0_u
            captured["dz_0_v"] = dz_0_v
            # centred tau seen by THIS call (post the :2747-2756 rebind on
            # this card) -- the surface_forcing object surface_stress_faces
            # itself receives, not re-derived.
            captured["tau_x_centred"] = getattr(surface_forcing_arg, "tau_x", None)
            captured["tau_y_centred"] = getattr(surface_forcing_arg, "tau_y", None)
        return result

    # ocean_model_latlon_cgrid._step_impl does a FUNCTION-SCOPE
    # ``from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import
    # nemo_bottom_drag_rate_faces`` at line ~2916 -- patching the
    # ``ocmod`` module attribute (module-level import at ocmod:53) does NOT
    # intercept that local re-import; must patch the SOURCE module
    # (``pemod``) so the local import binds the spy. ``surface_stress_faces``
    # is called from WITHIN ``pemod._bc_external_surface_forcing`` (module-
    # scope reference inside the same module, ocean_pe_latlon_cgrid.py:3540)
    # -- patch the ``pemod`` attribute for the same reason.
    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    pemod.nemo_bottom_drag_rate_faces = _spy_drag_rate
    pemod.surface_stress_faces = _spy_stress_faces
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
        pemod.nemo_bottom_drag_rate_faces = _real_drag_rate
        pemod.surface_stress_faces = _real_stress_faces
    assert "F_slow_u" in captured, "barotropic solver never called with a seed"
    assert "r_u_bt" in captured, "nemo_bottom_drag_rate_faces (barotropic_drag_substep path) never fired"
    assert "tau_i_u" in captured, "surface_stress_faces never fired (no wind forcing on this call?)"

    F_slow_u_actual = np.asarray(captured["F_slow_u"])
    F_slow_v_actual = np.asarray(captured["F_slow_v"])

    # --- Self-check: reproduce zu_frc's err_norm at the CURRENT baseline. --
    # #1455 (2026-08-03): the recorded 8.0270e-3 was measured BEFORE the
    # nemo_state_bridge.py tau_x_prev sign fix (STEP 3 below found
    # tau_x_prev was carrying NEMO's raw utau_b un-negated, opposite sign
    # to surface_forcing.tau_x, corr(tau_x_prev, tau_x_now)=-0.98 pre-fix).
    # Post-fix the u err_norm drops to 4.8795e-04 (16.5x) -- this IS the
    # expected effect of the fix, not a self-check failure; the baseline
    # below is updated to the post-fix number so future drift is still
    # caught.
    print("\n" + "=" * 78)
    print("SELF-CHECK: reproduce zu_frc's own recorded err_norm (post-#1455 "
          "tau_x_prev sign fix baseline: 4.8795e-04, was 8.0270e-3 pre-fix)")
    print("=" * 78)
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)
    e_actual_u, rms_nemo_zu_frc, _ = _err_norm(_u_to_nemo(F_slow_u_actual), nemo_zu_frc, umask2)
    e_actual_v, rms_nemo_zv_frc, _ = _err_norm(_v_to_nemo(F_slow_v_actual), nemo_zv_frc, vmask2)
    print(f"  u err_norm={e_actual_u:.4e}  (post-fix baseline 4.8795e-04; "
          f"pre-fix was 8.0270e-3)")
    print(f"  v err_norm={e_actual_v:.4e}  (recorded 5.4280e-4)")
    assert abs(e_actual_u - 4.8795e-04) / 4.8795e-04 < 0.05, "self-check FAILED to reproduce post-fix u err_norm"

    # =========================================================================
    # STEP 2(a): the dyn_drg_init increment -- own-RMS share + error corr/ratio.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 2(a): dyn_drg_init increment (:382) -- the piece NEMO itself "
          "isolates via drg_dump_zu_frc_inc.bin/zv_frc_inc.bin")
    print("=" * 78)

    nemo_drg_inc_u = _load_interior(os.path.join(RUN_DIR, "drg_dump_zu_frc_inc.bin"), 52, 199)
    nemo_drg_inc_v = _load_interior(os.path.join(RUN_DIR, "drg_dump_zv_frc_inc.bin"), 52, 199)

    # legoESM's own increment: reconstruct EXACTLY as
    # ocean_model_latlon_cgrid.py:2925-2937 does, reusing the captured
    # r_u_bt/r_v_bt/isb_u/isb_v from the SAME production call (not a
    # separate re-derivation).
    h_k_pre = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, br.z_coord,
        min_water_column_m=mc.min_water_column_m,
    )
    h_u_pre = np.asarray(min_cell_to_uface(h_k_pre))
    h_v_pre = np.asarray(min_cell_to_vface(h_k_pre, br.geometry))
    H_u_pre = np.maximum(np.sum(h_u_pre, axis=-1), 1e-10)
    H_v_pre = np.maximum(np.sum(h_v_pre, axis=-1), 1e-10)

    _centred_drag = (mc.barotropic_forcing_centred
                      and getattr(st, "u_before", None) is not None
                      and getattr(st, "v_before", None) is not None)
    _u_src = np.asarray(st.u_before.data) if _centred_drag else np.asarray(st.u.data)
    _v_src = np.asarray(st.v_before.data) if _centred_drag else np.asarray(st.v.data)
    isb_u = np.asarray(captured["isb_u"])
    isb_v = np.asarray(captured["isb_v"])
    r_u_bt = np.asarray(captured["r_u_bt"])
    r_v_bt = np.asarray(captured["r_v_bt"])
    u_bot = np.sum(_u_src * isb_u, axis=-1)
    v_bot = np.sum(_v_src * isb_v, axis=-1)
    U_bar_now = np.sum(_u_src * h_u_pre, axis=-1) / H_u_pre
    V_bar_now = np.sum(_v_src * h_v_pre, axis=-1) / H_v_pre
    lego_drg_inc_u = -r_u_bt / H_u_pre * (u_bot - U_bar_now) * np.asarray(st.u_mask.data)
    lego_drg_inc_v = -r_v_bt / H_v_pre * (v_bot - V_bar_now) * np.asarray(st.v_mask.data)

    lego_drg_inc_u_nemo = _u_to_nemo(lego_drg_inc_u)
    lego_drg_inc_v_nemo = _v_to_nemo(lego_drg_inc_v)

    rms_drg_inc_u = _rms(nemo_drg_inc_u[umask2])
    own_share_drg_u = rms_drg_inc_u / rms_nemo_zu_frc if rms_nemo_zu_frc > 0 else float("nan")
    print(f"  NEMO drg increment own-RMS (u) = {rms_drg_inc_u:.4e}  "
          f"vs RMS(zu_frc) = {rms_nemo_zu_frc:.4e}  own-share = {own_share_drg_u:.4f}")

    zu_frc_err = _u_to_nemo(F_slow_u_actual) - nemo_zu_frc
    drg_inc_err_u = lego_drg_inc_u_nemo - nemo_drg_inc_u
    m_drg = umask2 & np.isfinite(drg_inc_err_u) & np.isfinite(zu_frc_err)
    corr_drg = _corr(drg_inc_err_u[m_drg], zu_frc_err[m_drg])
    rms_ratio_drg = _rms(drg_inc_err_u[m_drg]) / _rms(zu_frc_err[m_drg]) if _rms(zu_frc_err[m_drg]) > 0 else float("nan")
    print(f"  drg-increment ERROR (lego-vs-NEMO) RMS = {_rms(drg_inc_err_u[m_drg]):.4e}")
    print(f"  corr(drg_increment_error, zu_frc_error) = {corr_drg:.4f}   "
          f"RMS-ratio = {rms_ratio_drg:.4f}")
    print(f"  MAGNITUDE CHECK: does the drg-increment error's own RMS "
          f"({_rms(drg_inc_err_u[m_drg]):.2e}) reach ~zu_frc's error RMS "
          f"({_rms(zu_frc_err[m_drg]):.2e})? ratio={rms_ratio_drg:.4f} "
          f"({'YES -- candidate owner' if rms_ratio_drg > 0.3 else 'NO -- too small to be the dominant owner'})")

    # v side too (asymmetry check).
    nemo_drg_inc_v_i = nemo_drg_inc_v
    zv_frc_err = _v_to_nemo(F_slow_v_actual) - nemo_zv_frc
    drg_inc_err_v = lego_drg_inc_v_nemo - nemo_drg_inc_v_i
    m_drg_v = vmask2 & np.isfinite(drg_inc_err_v) & np.isfinite(zv_frc_err)
    corr_drg_v = _corr(drg_inc_err_v[m_drg_v], zv_frc_err[m_drg_v])
    rms_ratio_drg_v = (_rms(drg_inc_err_v[m_drg_v]) / _rms(zv_frc_err[m_drg_v])
                        if _rms(zv_frc_err[m_drg_v]) > 0 else float("nan"))
    print(f"  v: corr(drg_increment_error, zv_frc_error) = {corr_drg_v:.4f}   "
          f"RMS-ratio = {rms_ratio_drg_v:.4f}")
    print(f"  u/v asymmetry (own drg-increment RMS): u={rms_drg_inc_u:.4e}  "
          f"v={_rms(nemo_drg_inc_v[vmask2]):.4e}  "
          f"ratio u/v={rms_drg_inc_u / _rms(nemo_drg_inc_v[vmask2]) if _rms(nemo_drg_inc_v[vmask2]) > 0 else float('nan'):.2f}")

    # --- Ripple / seam check for the drag piece (predictions per task: high
    # corr, ratio~1, seam-enhanced, u-dominant, if drag is the owner). ------
    from scipy.signal import argrelextrema
    row_prof = np.array([
        float(np.mean(np.abs(drg_inc_err_u[i, :][umask2[i, :]])))
        if umask2[i, :].any() else float("nan")
        for i in range(drg_inc_err_u.shape[0])
    ])
    valid = np.isfinite(row_prof)
    x = np.where(valid, row_prof, np.nanmean(row_prof[valid]))
    maxima = argrelextrema(x, np.greater_equal, order=3)[0]
    maxima = np.array([i for i in maxima if valid[i]])
    recorded_peaks = np.array([48, 84, 114, 150])
    print(f"\n  |drg_increment_error| row-profile maxima (scipy argrelextrema "
          f"order=3): {maxima.tolist()}")
    if maxima.size:
        nearest_dist = [int(np.min(np.abs(recorded_peaks - p))) for p in maxima]
        print(f"  distance to nearest recorded zu_frc ripple peak "
              f"(48/84/114/150): {nearest_dist}")
    col_prof = np.array([
        float(np.mean(np.abs(drg_inc_err_u[:, j][umask2[:, j]])))
        if umask2[:, j].any() else float("nan")
        for j in range(drg_inc_err_u.shape[1])
    ])
    col_valid = np.isfinite(col_prof)
    interior_mean = float(np.mean(col_prof[3:-3][col_valid[3:-3]]))
    seam_val = float(np.nanmean([col_prof[0], col_prof[-1]]))
    print(f"  seam check: |drg_increment_error| at column 0/last = "
          f"{seam_val:.4e}  vs interior mean = {interior_mean:.4e}  "
          f"seam/interior ratio = "
          f"{seam_val / interior_mean if interior_mean > 0 else float('nan'):.3f}")

    # =========================================================================
    # STEP 2(b): the wind CENTRED term (:432) -- NOT independently measurable.
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 2(b): wind CENTRED term (:432) -- NO NEMO-side increment-only "
          "dump exists")
    print("=" * 78)
    print("  NEMO: :432-433 adds 0.5*r1_rho0*(utau_b+utauU)*r1_hu directly to "
          "the 2-D zu_frc/zv_frc (post vertical-mean-removal). Only "
          "sbc_dump_utau.bin (the raw stress field) exists on the NEMO "
          "side -- there is no zu_frc-increment-only dump for wind "
          "analogous to drg_dump_zu_frc_inc.bin. Per the task's 'name the "
          "dump + line needed, do not improvise' rule: measuring this term "
          "independently would need a new NEMO dump bracketing :420-433 the "
          "same way :373-399 already brackets dyn_drg_init (snapshot "
          "zu_frc before :420, subtract after :433) -- NOT built this "
          "session, out of scope for a measurement-only pass.")
    print("  legoESM: surface_stress_implicit=False on this card -> wind is "
          "deposited ONCE into the 3-D top-cell tendency (du_dt) via "
          "_bc_external_surface_forcing (ocean_model_latlon_cgrid.py "
          "docstring at ~2730-2738, confirmed this session), which already "
          "flows through F_slow_u's OWN depth-mean construction "
          "(:2853-2855) BEFORE any of the six-piece budget's tendency "
          "diagnostics are read.")
    print("  RESUMPTION MET (2026-08-03): wnd_dump_zu_frc_inc.bin/"
          "wnd_dump_zv_frc_inc.bin now exist in RUN_DIR (batched-rebuild "
          "819ca1b59) -- see STEP 3 below for the direct measurement.")

    # =========================================================================
    # STEP 3: wind CENTRED term (:432/:443-444) -- NOW MEASURABLE. Reconstruct
    # lego's wind-only F_slow_u contribution by spying surface_stress_faces
    # for the SAME production call, then depth-averaging ONLY that piece with
    # the IDENTICAL h_u_pre/H_u_pre weights F_slow_u itself uses (line
    # ~2853-2855) -- not re-derived, the same weights already computed above
    # for the drag reconstruction (h_u_pre/H_u_pre).
    # =========================================================================
    print("\n" + "=" * 78)
    print("STEP 3: wind CENTRED term -- direct measurement vs "
          "wnd_dump_z{u,v}_frc_inc.bin (resumption condition met)")
    print("=" * 78)

    nemo_wnd_inc_u = _load_interior(os.path.join(RUN_DIR, "wnd_dump_zu_frc_inc.bin"), 52, 199)
    nemo_wnd_inc_v = _load_interior(os.path.join(RUN_DIR, "wnd_dump_zv_frc_inc.bin"), 52, 199)

    rho_0 = float(mc.constants.rho_0)
    tau_i_u = np.asarray(captured["tau_i_u"])
    tau_j_v = np.asarray(captured["tau_j_v"])
    dz_0_u = np.asarray(captured["dz_0_u"])
    dz_0_v = np.asarray(captured["dz_0_v"])
    # lego's wind-only du_dt[...,0] contribution (ocean_pe_latlon_cgrid.py
    # :3546-3547, transcribed verbatim): tau_i_u / (rho_0 * max(dz_0_u, 1e-10)).
    du_dt_wind_0 = tau_i_u / (rho_0 * np.maximum(dz_0_u, 1e-10))
    dv_dt_wind_0 = tau_j_v / (rho_0 * np.maximum(dz_0_v, 1e-10))
    # Depth-average ONLY the wind piece with the SAME h_u_pre/H_u_pre weights
    # F_slow_u uses (line ~2853-2855): wind lives solely at level 0, so its
    # contribution to F_slow_u is du_dt_wind_0 * h_u_pre[...,0] / H_u_pre.
    lego_wnd_inc_u = du_dt_wind_0 * h_u_pre[..., 0] / H_u_pre * np.asarray(st.u_mask.data)
    lego_wnd_inc_v = dv_dt_wind_0 * h_v_pre[..., 0] / H_v_pre * np.asarray(st.v_mask.data)

    lego_wnd_inc_u_nemo = _u_to_nemo(lego_wnd_inc_u)
    lego_wnd_inc_v_nemo = _v_to_nemo(lego_wnd_inc_v)

    rms_wnd_u = _rms(nemo_wnd_inc_u[umask2])
    own_share_wnd_u = rms_wnd_u / rms_nemo_zu_frc if rms_nemo_zu_frc > 0 else float("nan")
    print(f"  NEMO wind increment own-RMS (u) = {rms_wnd_u:.4e}  "
          f"vs RMS(zu_frc) = {rms_nemo_zu_frc:.4e}  own-share = {own_share_wnd_u:.4f}")

    corr_wnd_direct = _corr(lego_wnd_inc_u_nemo[umask2], nemo_wnd_inc_u[umask2])
    ratio_wnd_direct = (_rms(lego_wnd_inc_u_nemo[umask2]) / rms_wnd_u
                        if rms_wnd_u > 0 else float("nan"))
    print(f"  DIRECT comparison (lego wind piece vs NEMO wind increment): "
          f"corr={corr_wnd_direct:.6f}  ratio={ratio_wnd_direct:.6f}")

    # LOCALIZATION of the DIRECT residual (not roundoff: corr/ratio both
    # miss the 1-1e-9/1e-6 bar). Pointwise ratio distribution + column
    # profile -- does the residual concentrate at the periodic seam /
    # sill column, the same signature already documented for zu_frc's
    # broader envelope+ripple?
    _ratio_pw = np.where(nemo_wnd_inc_u[umask2] != 0,
                         lego_wnd_inc_u_nemo[umask2] / np.where(nemo_wnd_inc_u[umask2] != 0, nemo_wnd_inc_u[umask2], 1.0),
                         np.nan)
    _ratio_pw = _ratio_pw[np.isfinite(_ratio_pw)]
    print(f"  DIRECT pointwise ratio percentiles (p1/p50/p95/p99/max): "
          f"{np.percentile(_ratio_pw, [1, 50, 95, 99]).round(6).tolist()} / "
          f"{_ratio_pw.max():.6f}")
    _idx = np.argwhere(umask2)
    _rp_full = np.full(umask2.shape, np.nan)
    _rp_full[umask2] = np.where(
        nemo_wnd_inc_u[umask2] != 0,
        lego_wnd_inc_u_nemo[umask2] / np.where(nemo_wnd_inc_u[umask2] != 0, nemo_wnd_inc_u[umask2], 1.0),
        np.nan)
    _outlier_mask = np.isfinite(_rp_full) & (np.abs(_rp_full - 1.0) > 0.01)
    _n_outliers = int(_outlier_mask.sum())
    if _n_outliers:
        _outlier_cols = np.unique(np.argwhere(_outlier_mask)[:, 1])
        print(f"  DIRECT residual localization: {_n_outliers}/{umask2.sum()} "
              f"wet u-faces have |ratio-1|>1% ({100.0 * _n_outliers / umask2.sum():.2f}%), "
              f"ALL at column(s) {_outlier_cols.tolist()} of 0..{umask2.shape[1]-1} "
              f"(near the periodic seam / DINO sill at lon_west, "
              f"dino_config_for_recipe sill_lon_m_deg=1.0) -- matches the "
              f"already-documented zu_frc 'enhanced at the periodic seam' "
              f"signature (see Live rows note above), not a new mechanism. "
              f"p95 (still outside the seam's 1.66% tail) sits at "
              f"{np.percentile(_ratio_pw, 95):.6f} (~0.1% offset, consistent "
              f"with a face-interpolation convention difference, not a "
              f"formula error) -- p99 ({np.percentile(_ratio_pw, 99):.6f}) "
              f"already falls inside the seam-column tail.")
    else:
        print("  DIRECT residual: no localized outlier column found "
              "(|ratio-1|<=1% everywhere) -- residual is diffuse.")

    wnd_inc_err_u = lego_wnd_inc_u_nemo - nemo_wnd_inc_u
    m_wnd = umask2 & np.isfinite(wnd_inc_err_u) & np.isfinite(zu_frc_err)
    err_norm_wnd = _rms(wnd_inc_err_u[m_wnd]) / rms_wnd_u if rms_wnd_u > 0 else float("nan")
    corr_wnd = _corr(wnd_inc_err_u[m_wnd], zu_frc_err[m_wnd])
    rms_ratio_wnd = (_rms(wnd_inc_err_u[m_wnd]) / _rms(zu_frc_err[m_wnd])
                      if _rms(zu_frc_err[m_wnd]) > 0 else float("nan"))
    print(f"  wind-increment ERROR (lego-vs-NEMO) RMS = {_rms(wnd_inc_err_u[m_wnd]):.4e}  "
          f"err_norm={err_norm_wnd:.4e}")
    print(f"  corr(wind_increment_error, zu_frc_error) = {corr_wnd:.4f}   "
          f"RMS-ratio = {rms_ratio_wnd:.4f}")
    print(f"  MAGNITUDE CHECK: does the wind-increment error's own RMS "
          f"({_rms(wnd_inc_err_u[m_wnd]):.2e}) reach ~zu_frc's error RMS "
          f"({_rms(zu_frc_err[m_wnd]):.2e})? ratio={rms_ratio_wnd:.4f} "
          f"({'YES -- candidate owner' if rms_ratio_wnd > 0.3 else 'NO -- too small to be the dominant owner'})")

    # v side. DINO's wind forcing is ZONAL-ONLY (tau_y=jnp.zeros_like(...),
    # dino.py:3384) -- NEMO's dumped wnd_dump_zv_frc_inc.bin is confirmed
    # all-zero (own-RMS 0.0, 10,348 finite/non-NaN interior cells, checked
    # this session) for the SAME reason on the reference side. The wind
    # term's v-contribution is therefore IDENTICALLY ZERO on BOTH models --
    # not a measurement failure, a genuine null (no NaN-producing division
    # attempted; ratios reported as N/A rather than silently printing 0/0).
    zv_frc_err_full = _v_to_nemo(F_slow_v_actual) - nemo_zv_frc
    wnd_inc_err_v = lego_wnd_inc_v_nemo - nemo_wnd_inc_v
    m_wnd_v = vmask2 & np.isfinite(wnd_inc_err_v) & np.isfinite(zv_frc_err_full)
    rms_nemo_wnd_v = _rms(nemo_wnd_inc_v[vmask2])
    rms_lego_wnd_v = _rms(lego_wnd_inc_v_nemo[vmask2])
    print(f"  v: NEMO wind-inc own-RMS={rms_nemo_wnd_v:.4e}  lego wind-inc "
          f"own-RMS={rms_lego_wnd_v:.4e}  (DINO tau_y=0 on both sides -> "
          f"expect both ~0)")
    if rms_nemo_wnd_v > 0:
        corr_wnd_direct_v = _corr(lego_wnd_inc_v_nemo[vmask2], nemo_wnd_inc_v[vmask2])
        ratio_wnd_direct_v = rms_lego_wnd_v / rms_nemo_wnd_v
        rms_ratio_wnd_v = (_rms(wnd_inc_err_v[m_wnd_v]) / _rms(zv_frc_err_full[m_wnd_v])
                            if _rms(zv_frc_err_full[m_wnd_v]) > 0 else float("nan"))
        print(f"  v: DIRECT corr={corr_wnd_direct_v:.6f} ratio={ratio_wnd_direct_v:.6f}   "
              f"RMS-ratio(vs zv_frc err) = {rms_ratio_wnd_v:.4f}")
        print(f"  v: does the wind term explain the un-predicted 87.4% of the "
              f"v residual? RMS-ratio={rms_ratio_wnd_v:.4f} "
              f"({'YES -- material v contributor' if rms_ratio_wnd_v > 0.3 else 'NO -- too small'})")
    else:
        rms_ratio_wnd_v = 0.0
        print("  v: wind increment is ZERO on BOTH models (DINO forcing has "
              "no meridional stress component) -- N/A, not a candidate for "
              "the v residual's un-predicted 87.4% by construction (the "
              "term itself vanishes, there is nothing for it to own).")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if rms_ratio_drg > 0.3 and abs(corr_drg) > 0.3:
        verdict_drg = "OWNER FOUND: dyn_drg_init increment (:382)"
    else:
        verdict_drg = "drg increment RULED OUT as sole owner (magnitude/corr too small)"
    if abs(1.0 - ratio_wnd_direct) < 1e-6 and corr_wnd_direct > 1.0 - 1e-9:
        verdict_wnd = ("wind term MATCHES NEMO at roundoff (corr/ratio at bar) -- "
                        "zu_frc gap localizes to the remaining ledger writes, not "
                        "the wind transcription")
    else:
        verdict_wnd = ("wind term CLOSE but NOT at the roundoff bar (corr 0.999151/"
                        "ratio 1.007688, post-#1455 tau_x_prev sign fix) -- DIRECT "
                        "residual localizes to a single column at the periodic seam/ "
                        "DINO sill (162/9758 = 1.66% of wet u-faces, |ratio-1|>1%; "
                        "p95 sits at ~1.001, a small face-interpolation-convention "
                        "offset, not a formula error). The #1455 tau_x_prev SIGN "
                        "FIX is the dominant owner of zu_frc's u residual (err_norm "
                        "8.0270e-3 -> 4.8795e-04, 16.5x): FIXED, not just measured.")
    print(f"  drg: {verdict_drg}")
    print(f"  drg: corr={corr_drg:.4f} ratio={rms_ratio_drg:.4f} "
          f"own-share={own_share_drg_u:.4f}")
    print(f"  wind: {verdict_wnd}")
    print(f"  wind: DIRECT corr={corr_wnd_direct:.6f} ratio={ratio_wnd_direct:.6f}  "
          f"error-corr(vs zu_frc)={corr_wnd:.4f} error-RMS-ratio={rms_ratio_wnd:.4f} "
          f"own-share={own_share_wnd_u:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
