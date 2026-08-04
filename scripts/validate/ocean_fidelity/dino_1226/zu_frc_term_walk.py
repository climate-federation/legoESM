"""#1226 zu_frc/zu_trd term walk: does legoESM lack NEMO's Coriolis removal?

TASK: NEMO's ``dyn_spg_ts`` subtracts the pre-step 2-D barotropic Coriolis
trend from ``zu_frc`` before the substep loop::

    CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )   ! dynspg_ts.F90:364 (MLF branch)
    zu_frc(ji,jj) = zu_frc(ji,jj) - zu_trd(ji,jj) * ssumask(ji,jj)      ! dynspg_ts.F90:367
    zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)     ! dynspg_ts.F90:368

The prior session's ``spg_substep_chain.py`` STAGE 7 item 4 (fix_plan "THE
BAROTROPIC SEED") claimed legoESM has "no zu_trd-equivalent post-hoc
subtraction at all", citing ``ocean_model_latlon_cgrid.py:2836-2844`` (where
``F_slow_u`` is FIRST built).  That citation only reads the *first*
construction of ``F_slow_u`` in that function.  The SAME function, further
down (``ocean_model_latlon_cgrid.py:3226-3313``), has an EXPLICIT block that
does exactly this subtraction, gated by
``config.barotropic_coriolis_split == "live"``::

    _cor_u_sub, _cor_v_sub = barotropic_coriolis_een_pre_step(...)   # :3286-3290
    F_slow_u = (F_slow_u - _cor_u_sub) * state.u_mask.data            # :3291
    F_slow_v = (F_slow_v - _cor_v_sub) * state.v_mask.data            # :3292

and the DINO/NEMO MLF recipe card that ``spg_substep_chain.py`` itself builds
(``dino_config_for_recipe("nemo_dino_kamm_mlf")``,
``experiments/dino.py:1325``) sets ``"barotropic_coriolis_split": "live"``
EXPLICITLY -- confirmed here by instantiating the config and printing the
field (Rule 10), not by reading a docstring:

    >>> from legoesm.ocean.experiments.dino import dino_config_for_recipe
    >>> dino_config_for_recipe("nemo_dino_kamm_mlf").barotropic_coriolis_split
    'live'

So on THIS card the subtraction is NOT absent -- it fires every step.  The
prior claim was a real defect in the PRIOR MEASUREMENT (it never looked past
line 2844 in a 4000+ line function), not a defect in legoESM's physics.

This script proves which is really running by RECONSTRUCTING zu_frc two
ways from the SAME production call and checking which matches NEMO's own
dump ``spg_dump_zu_frc.bin``:

  (i)  ``captured["kw"]["F_slow_u"]`` -- the actual value the production
       barotropic solver receives (WITH the :3291 subtraction already
       applied, since the capture point is downstream of it).
  (ii) The reconstructed NEVER-SUBTRACTED variant, ``F_slow_u + cor_u_sub``,
       built from the SAME ``_cor_u_sub`` the subtraction block itself
       computed (spied via ``barotropic_coriolis_een_pre_step``) -- i.e.
       "undo just the :3291 line, change nothing else".

If (i) is closer to NEMO's dump than (ii): the subtraction is present,
correct, and REQUIRED -- verdict (a)-does-not-apply / the removal already
exists (branch (b) in the task's terms: legoESM does not lack the term).
If (ii) is closer: legoESM's production wiring is NOT actually taking the
"live" branch for this call (a real absence) and the prior finding stands.

Then (if the subtraction is confirmed present and the residual persists):
a per-level correlation between zu_frc's own error field and the depth-mean
of the momentum-row error dumps (dyn_ldf, dyn_vor EEN, dyn_adv ZAD) -- never
run before per fix_plan -- to confirm/refute "the barotropic rows are a
downstream consequence of the momentum rows".

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zu_frc_term_walk.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.core.precision import PrecisionPolicy, set_policy
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

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART_FILE = "DINO_00057600_restart.nc"
DT = 2700.0

# --- Precondition 1: fp64 everywhere (skill Rule 1c) -----------------------
set_policy(PrecisionPolicy.fp64())

# --- Precondition 2: register every dump's NEMO time level (source-cited),
# fail-closed registry (Rule 1d) -- an unregistered dump raises. ------------
register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345,367 zu_frc after the zu_trd subtraction, "
              "snapshotted at :489-501 right before the substep loop starts "
              "(the loop's BEFORE-level entry, ln_bt_fw=F).")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
register_dump("cor2d_dump_zu_trd_substep1.bin", "now",
              "dynspg_ts.F90:760,778 zu_trd computed INSIDE the substep loop "
              "at jn=1 from ua_e/va_e (the evolving substep transport) -- a "
              "DIFFERENT dyn_cor_2D call from the pre-loop one at :364 (same "
              "subroutine, different input velocities/time). Dumped for "
              "context only (confirms the live in-loop Coriolis magnitude), "
              "NOT used in the zu_frc reconstruction below (that uses the "
              ":364 pre-loop call, reproduced via "
              "barotropic_coriolis_een_pre_step on legoESM's own before-level "
              "seed, matching NEMO's puu_b(:,:,Kmm)/pvv_b(:,:,Kmm) inputs at "
              ":364 -- see the docstring above for why Kmm, not Kbb).")
register_dump("cor2d_dump_zv_trd_substep1.bin", "now", "same as zu_trd substep1")
register_dump("vor_dump_du.bin", "now",
              "dynvor.F90:151-155 vor_ene/vor_een(..., puu(:,:,:,Kmm), "
              "pvv(:,:,:,Kmm), puu(:,:,:,Krhs), pvv(:,:,:,Krhs)) -- velocity "
              "INPUT at Kmm (now), tendency INCREMENT written into Krhs "
              "(dumped as the post-minus-pre difference, dynvor.F90:184-193).")
register_dump("vor_dump_dv.bin", "now", "same as vor_dump_du")
register_dump("ldf_dump_du.bin", "now",
              "dynldf.F90:85 dyn_ldf_iso(kt, Kbb, Kmm, puu, pvv, Krhs) -- "
              "DINO uses np_lap_i (rotated laplacian); velocity read at both "
              "Kbb and Kmm inside dyn_ldf_iso but the dump (dynldf.F90:98-108) "
              "is the Krhs INCREMENT, registered 'now' matching vor_dump's "
              "convention for a Kmm-input-dominated tendency (isoneutral "
              "slopes use Kmm T/S).")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")
register_dump("zad_dump_du.bin", "now",
              "dynadv.F90:97 dyn_zad(kt, Kmm, puu, pvv, Krhs) -- vertical "
              "advection, velocity INPUT at Kmm, Krhs INCREMENT dumped "
              "(dynadv.F90:101-103).")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")

for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin",
    "cor2d_dump_zu_trd_substep1.bin", "cor2d_dump_zv_trd_substep1.bin",
    "vor_dump_du.bin", "vor_dump_dv.bin",
    "ldf_dump_du.bin", "ldf_dump_dv.bin",
    "zad_dump_du.bin", "zad_dump_dv.bin",
):
    time_level_for_dump(_name)  # raises if unregistered -- fail loud, not silent


def _load_interior(path: str, ni: int, nj: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f8").reshape(nj, ni)


def _load_full(path: str, jpi: int, jpj: int, hls: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<f8").reshape(jpj, jpi)
    if hls:
        a = a[hls:-hls, hls:-hls]
    return a


def _load_full_3d(path: str, jpi: int, jpj: int, jpkm1: int, hls: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<f8").reshape(jpkm1, jpj, jpi)
    if hls:
        a = a[:, hls:-hls, hls:-hls]
    return np.moveaxis(a, 0, -1)


def _stats(name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> tuple:
    m = mask & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    rms = float(np.sqrt(np.mean(ne ** 2))) if ne.size else float("nan")
    err_norm = float(np.sqrt(np.mean((lo - ne) ** 2))) / rms if rms > 0 else float("nan")
    maxdiff = float(np.max(np.abs(lo - ne))) if lo.size else float("nan")
    near_zero = float(np.mean(np.abs(ne) < 1e-3 * (rms if rms > 0 else 1.0)))
    print(f"  {name:<32s} err_norm={err_norm:.4e}  RMS(nemo)={rms:.4e}  "
          f"max|diff|={maxdiff:.4e}  near_zero_frac={near_zero:.3f}  n={int(m.sum())}")
    return err_norm, lo, ne, m


def _align_scan(name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> None:
    """Self-check (skill Rule required): a genuine match must peak sharply at
    (dj,di)=(0,0). A flat scan or nonzero peak = misalignment, not a result."""
    best = None
    all_errs = {}
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
            all_errs[(dj, di)] = err
            if best is None or err < best[0]:
                best = (err, dj, di)
    print(f"  [align scan] {name}: best (dj,di)={best[1:]} err_norm={best[0]:.4e}  "
          f"(0,0)={all_errs.get((0, 0)):.4e}  "
          "(expect (0,0) with a clear minimum)")


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_term_walk")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both' to match NEMO's real ladder)")

    # --- Rule 10: instantiate + print the config field this whole task
    # hinges on, rather than trusting the docstring/comment.
    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.coriolis_scheme={dcfg.coriolis_scheme!r}  "
          f"barotropic_coriolis_split={dcfg.barotropic_coriolis_split!r}  "
          f"vorticity_scheme={dcfg.vorticity_scheme!r}")
    assert dcfg.coriolis_scheme == "explicit_ab2", (
        "this whole reconstruction assumes explicit_ab2 (Coriolis inside "
        "du_dt, hence inside F_slow_u's depth-mean) -- re-derive if the "
        "recipe default ever changes")
    assert dcfg.barotropic_coriolis_split == "live", (
        "the nemo_dino_kamm_mlf card is expected to set barotropic_coriolis_"
        "split='live' (experiments/dino.py:1325) -- if this changes, the "
        "verdict below must be re-derived, not assumed")

    # --- Build the legoESM twin state from the SAME restart NEMO's dumps came from
    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_term_walk twin state")
    print(f"mc.barotropic_coriolis_split={mc.barotropic_coriolis_split!r}  "
          f"mc.coriolis_scheme={mc.coriolis_scheme!r}")
    assert mc.barotropic_coriolis_split == "live"

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Spy 1: the production barotropic entry point -- capture the
    # ACTUAL F_slow_u/F_slow_v it receives (post-subtraction, since the
    # subtraction runs in ocean_model_latlon_cgrid.py BEFORE this call).
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        is_seeded = kw.get("eta_init") is not None
        if is_seeded and "F_slow_u" not in captured:
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    # --- Spy 1b: model.tendencies() itself -- capture du_dt (summed RHS,
    # used as the ldf/zad magnitude-profile proxy in Part 2).
    _real_tend_1b = model.tendencies

    def _spy_tend_1b(state_arg, *a, **kw):
        result = _real_tend_1b(state_arg, *a, **kw)
        if "du_dt_3d" not in captured:
            captured["du_dt_3d"] = result.du_dt.data
        return result

    model.tendencies = _spy_tend_1b

    # --- Spy 2: barotropic_coriolis_een_pre_step -- capture the EXACT
    # (_cor_u_sub, _cor_v_sub) the :3291-3292 subtraction removes, so the
    # NEVER-SUBTRACTED variant is "add this back", not a re-derivation.
    _real_pre_step = bmod.barotropic_coriolis_een_pre_step

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
        model.tendencies = _real_tend_1b

    assert "F_slow_u" in captured, "barotropic_substeps_latlon_cgrid never called with a seed -- check barotropic_solver"
    assert "du_dt_3d" in captured, "tendencies() never called -- check the du_dt spy"
    assert "cor_u_sub" in captured, (
        "barotropic_coriolis_een_pre_step never called -- the 'live' "
        "subtraction branch did NOT fire for this config/call. This would "
        "itself be evidence for verdict (a) (absence in practice even if "
        "present in code) -- STOP and re-examine the gating condition at "
        "ocean_model_latlon_cgrid.py:3235-3237 before proceeding.")

    F_slow_u_actual = np.asarray(captured["F_slow_u"])
    F_slow_v_actual = np.asarray(captured["F_slow_v"])
    cor_u_sub = np.asarray(captured["cor_u_sub"])
    cor_v_sub = np.asarray(captured["cor_v_sub"])
    # Reconstruct the NEVER-SUBTRACTED variant: undo exactly the :3291/:3292
    # line and nothing else.
    F_slow_u_nosub = F_slow_u_actual + cor_u_sub
    F_slow_v_nosub = F_slow_v_actual + cor_v_sub

    print(f"\n|cor_u_sub| mean={float(np.mean(np.abs(cor_u_sub))):.4e}  "
          f"max={float(np.max(np.abs(cor_u_sub))):.4e}  "
          f"(this IS the pre-step 2D Coriolis dynspg_ts.F90:364 computes and "
          f":367 subtracts -- if it is ~0 everywhere, the subtraction is a "
          f"no-op here and this whole branch is moot)")
    assert float(np.max(np.abs(cor_u_sub))) > 1e-6, (
        "cor_u_sub is numerically zero -- the subtraction branch fired but "
        "removed nothing, which would make the (a)/(b) question vacuous; "
        "investigate before trusting the comparison below")

    # --- NEMO's own dump: the AFTER-subtraction zu_frc it uses ------------
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)

    tmask = np.asarray(g.tmask) > 0.5
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    def _u_to_nemo(a):
        return np.asarray(a)[:, 1:]

    def _v_to_nemo(a):
        return np.asarray(a)[1:, :]

    print("\n=== THE DECISIVE RECONSTRUCTION: does legoESM ADD what NEMO removes? ===")
    print("(i) ACTUAL production F_slow_u (WITH the :3291 subtraction already applied):")
    e_actual_u, _, _, _ = _stats("zu_frc [ACTUAL, with subtraction]",
                                  _u_to_nemo(F_slow_u_actual), nemo_zu_frc, umask2)
    e_actual_v, _, _, _ = _stats("zv_frc [ACTUAL, with subtraction]",
                                  _v_to_nemo(F_slow_v_actual), nemo_zv_frc, vmask2)
    print("(ii) RECONSTRUCTED never-subtracted variant (F_slow_u + cor_u_sub):")
    e_nosub_u, _, _, _ = _stats("zu_frc [NEVER-SUBTRACTED]",
                                 _u_to_nemo(F_slow_u_nosub), nemo_zu_frc, umask2)
    e_nosub_v, _, _, _ = _stats("zv_frc [NEVER-SUBTRACTED]",
                                 _v_to_nemo(F_slow_v_nosub), nemo_zv_frc, vmask2)

    print(f"\nu: actual={e_actual_u:.4e}  never-subtracted={e_nosub_u:.4e}  "
          f"ratio(nosub/actual)={e_nosub_u / e_actual_u:.2f}x")
    print(f"v: actual={e_actual_v:.4e}  never-subtracted={e_nosub_v:.4e}  "
          f"ratio(nosub/actual)={e_nosub_v / e_actual_v:.2f}x")

    verdict_b = e_actual_u < e_nosub_u
    print(f"\nVERDICT: {'(b) the subtraction IS present, wired, and REQUIRED' if verdict_b else '(a) legoESM genuinely lacks/mis-applies the subtraction'} "
          f"-- ACTUAL is {'CLOSER to' if verdict_b else 'FARTHER from'} NEMO's dump than the never-subtracted reconstruction.")

    # --- Self-check 1: alignment scan on the ACTUAL production value must
    # peak sharply at (0,0) -- reproduces the historical self-check from
    # spg_substep_chain.py (8.03e-3 at di=0 vs 0.64-0.98 at +-1).
    print("\n=== SELF-CHECK 1: alignment scan (ACTUAL production F_slow_u) ===")
    _align_scan("zu_frc [ACTUAL]", _u_to_nemo(F_slow_u_actual), nemo_zu_frc, umask2)
    _align_scan("zv_frc [ACTUAL]", _v_to_nemo(F_slow_v_actual), nemo_zv_frc, vmask2)

    # --- Self-check 2: reproduce the historical u/v asymmetry (u ~15x
    # worse than v) on THIS reconstruction, as an instrument-consistency
    # check against the previously recorded numbers (8.03e-3 / 5.43e-4).
    print("\n=== SELF-CHECK 2: reproduce the recorded u/v asymmetry ===")
    print(f"  u err_norm={e_actual_u:.4e} (fix_plan recorded 8.03e-03)")
    print(f"  v err_norm={e_actual_v:.4e} (fix_plan recorded 5.43e-04, ~15x smaller)")
    print(f"  ratio u/v = {e_actual_u / e_actual_v:.1f}x "
          "(expect an order-of-magnitude asymmetry, reproducing the historical finding)")

    # =========================================================================
    # PART 2 (only meaningful if (b) holds, i.e. the residual persists despite
    # the subtraction being genuinely present): per-level correlation of
    # zu_frc's own error field against the depth-mean of the momentum-row
    # error fields (dyn_ldf, dyn_vor EEN, dyn_adv ZAD). fix_plan explicitly
    # flags this was NEVER RUN -- only an RSS magnitude coincidence was noted.
    #
    # TWO DIFFERENT RIGOR LEVELS, stated explicitly (do not conflate them):
    #   dyn_vor EEN: legoESM's ``tendencies_with_diagnostics`` (a public,
    #     already-existing API, ocean_model_latlon_cgrid.py:2594) exposes
    #     this term separately as ``vortcor_u``/``vortcor_v``
    #     (MomentumTendencyDiagnostics, state.py:684-687,
    #     ocean_pe_latlon_cgrid.py:4510-4512). Under the DEFAULT
    #     coriolis_scheme this diagnostic is RELATIVE-vorticity-only (its own
    #     docstring: "the planetary Coriolis f×u is applied in the
    #     forward-backward step function and is NOT included") -- NOT
    #     comparable to NEMO's total (f+zeta) dump on its own. But THIS
    #     card's vorticity_scheme="een_total" changes that: read
    #     ocean_pe_latlon_cgrid.py:4035-4054 -- the separate face-f Matsuno
    #     add is explicitly GATED OFF for "een_total"/"ene_total" ("the
    #     separate face-f add would double-count f") and instead "Folded
    #     into the vortcor diagnostic slot (which then carries (f+ζ)×u,
    #     mirroring Veros...)". So on THIS recipe vortcor_u IS the combined
    #     (f+ζ) term -- the SAME quantity NEMO's ln_dynvor_een/np_CRV
    #     ``vor_dump_du.bin`` carries -- confirmed by reading the source,
    #     not assumed, before using it as a genuine per-level ERROR check.
    #   dyn_ldf / dyn_adv ZAD: legoESM has NO separately-exposed field for
    #     either (only the unrelated WENO Dterm diagnostic exists). Without
    #     one, only a weaker MAGNITUDE-PROFILE proxy is possible (does the
    #     depth profile of "where NEMO's term is large" look like "where
    #     legoESM's summed RHS is large"). Labeled PLAUSIBLE, never upgraded
    #     to CONFIRMED, per Rule 1c-adjacent honesty about instrument limits.
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 2: per-level correlation -- are the barotropic rows DOWNSTREAM")
    print("of the momentum rows (dyn_ldf/dyn_vor/dyn_adv ZAD)?")
    print("=" * 78)

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    # Read-only diagnostic call on the SAME twin state (start-of-step
    # tendency estimate, per its own docstring) -- a public API, not a
    # production monkeypatch.
    assert mc.vorticity_scheme == "een_total", (
        "the vortcor_u==combined-(f+zeta) claim above only holds for "
        "een_total/ene_total (ocean_pe_latlon_cgrid.py:4051-4054) -- "
        "re-derive before trusting the comparison if this ever changes")
    _tend_diag, _diag = model.tendencies_with_diagnostics(
        st, surface_forcing=sf, dt=DT)
    vortcor_u_3d = np.asarray(_diag.vortcor_u.data)  # (n_lat, n_lon+1, n_lev)

    # NEMO's PER-LEVEL momentum-row dumps (full domain, 35 levels, Kmm-input
    # tendency INCREMENTS -- see registrations above).
    nemo_vor_du = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_du = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)

    # --- (A) REAL per-level ERROR correlation: dyn_vor EEN, both sides the
    # SAME term (legoESM's vortcor_u IS its dyn_vor EEN tendency contribution).
    vortcor_face = _u_to_nemo(vortcor_u_3d)  # (n_lat, n_lon, n_lev) NEMO u-columns
    n_lat_c = min(vortcor_face.shape[0], nemo_vor_du.shape[0])
    n_lon_c = min(vortcor_face.shape[1], nemo_vor_du.shape[1])
    n_lev_c = min(vortcor_face.shape[2], nemo_vor_du.shape[2])
    vc = vortcor_face[:n_lat_c, :n_lon_c, :n_lev_c]
    nv = nemo_vor_du[:n_lat_c, :n_lon_c, :n_lev_c]
    umask_c = umask2[:n_lat_c, :n_lon_c]
    err_by_level_vor = np.array([
        float(np.sqrt(np.nanmean((vc[..., k] - nv[..., k])[umask_c] ** 2)))
        for k in range(n_lev_c)
    ])
    rms_by_level_vor_nemo = np.array([
        float(np.sqrt(np.nanmean(nv[..., k][umask_c] ** 2))) for k in range(n_lev_c)
    ])
    print(f"\n(A) dyn_vor EEN per-level ERROR (legoESM vortcor_u vs NEMO vor_dump_du.bin), "
          f"{n_lev_c} levels:")
    print(f"    total err_norm={np.sqrt(np.mean(err_by_level_vor**2))/np.sqrt(np.mean(rms_by_level_vor_nemo**2)):.4e} "
          "(should be near the recorded dyn_vor EEN u row 1.001180-ish DEBT scale)")

    # --- (B) magnitude-profile proxy: dyn_ldf, dyn_adv ZAD (no separately-
    # exposed legoESM field exists for either -- see the note above).
    rms_by_level_ldf = np.sqrt(np.nanmean(nemo_ldf_du ** 2, axis=(0, 1)))
    rms_by_level_zad = np.sqrt(np.nanmean(nemo_zad_du ** 2, axis=(0, 1)))
    rms_momentum_total_no_vor = np.sqrt(rms_by_level_ldf ** 2 + rms_by_level_zad ** 2)

    # zu_frc's own per-level error profile is not directly measurable either
    # (zu_frc is ALREADY a depth-mean by the time it is dumped -- NEMO never
    # dumps the pre-depth-mean 3-D Krhs at this exact point). The best
    # available proxy for "which levels drive zu_frc's error" is legoESM's
    # own 3-D RHS (du_dt) magnitude profile, captured by spy 1b above (the
    # SAME call that produced vortcor_u/F_slow_u -- no extra model.step()).
    du_dt_3d = np.asarray(captured["du_dt_3d"])
    du_dt_face = _u_to_nemo(du_dt_3d)
    n_lat_c2 = min(du_dt_face.shape[0], nemo_ldf_du.shape[0])
    n_lon_c2 = min(du_dt_face.shape[1], nemo_ldf_du.shape[1])
    umask_c2 = umask2[:n_lat_c2, :n_lon_c2]
    rms_by_level_lego_rhs = np.sqrt(np.array([
        np.nanmean(du_dt_face[:n_lat_c2, :n_lon_c2, k][umask_c2] ** 2)
        for k in range(du_dt_face.shape[-1])
    ]))

    n_levels_common = min(len(rms_momentum_total_no_vor), len(rms_by_level_lego_rhs))
    corr_r = float(np.corrcoef(
        rms_momentum_total_no_vor[:n_levels_common],
        rms_by_level_lego_rhs[:n_levels_common])[0, 1])
    print(f"\n(B) dyn_ldf+dyn_adv-ZAD per-level MAGNITUDE-PROFILE proxy "
          f"(NEMO's own RSS profile vs legoESM's SUMMED du_dt RMS profile), "
          f"{n_levels_common} levels: r={corr_r:.4f}")
    print("    (proxy only -- no separately-exposed legoESM ldf/zad field "
          "exists; this tests level-profile similarity, not a term-by-term "
          "error match. See module docstring for why dyn_vor got the "
          "stronger check above and these two did not.)")

    known_rows_rss = float(np.sqrt(3.9e-3 ** 2 + 1.2e-3 ** 2 + 4.9e-3 ** 2 + 1.2e-3 ** 2))
    print(f"\nRSS magnitude check (unchanged from fix_plan, reported again for "
          f"completeness): known momentum-row err_norms RSS = {known_rows_rss:.4e} "
          f"vs zu_frc err_norm = {e_actual_u:.4e}")

    unification_verdict = "PLAUSIBLE (magnitude-consistent, structure weakly tested)"
    if abs(corr_r) < 0.3:
        unification_verdict = f"REFUTED for ldf/zad (weak/no per-level structural correlation, r={corr_r:.2f})"
    elif abs(corr_r) > 0.7:
        unification_verdict = f"SUPPORTED for ldf/zad (strong per-level structural correlation, r={corr_r:.2f})"
    print(f"\nUNIFICATION VERDICT ('barotropic rows are downstream of momentum rows'): "
          f"{unification_verdict}")
    print("LABEL: dyn_vor EEN piece is CONFIRMED-INSTRUMENT (real per-level error, "
          "same term both sides). dyn_ldf/dyn_adv-ZAD piece stays PLAUSIBLE "
          "(magnitude-profile proxy only, not a term-by-term error match -- "
          "would need legoESM to expose ldf/zad as separate diagnostic fields, "
          "which is a transcription-adjacent change out of scope for this "
          "READ-ONLY measurement).")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"zu_frc [ACTUAL, subtraction applied]   u={e_actual_u:.4e}  v={e_actual_v:.4e}")
    print(f"zu_frc [NEVER-SUBTRACTED reconstruction] u={e_nosub_u:.4e}  v={e_nosub_v:.4e}")
    print(f"VERDICT: {'(b) CLOSED -- subtraction present & required, branch not an absence' if verdict_b else '(a) genuine absence confirmed'}")
    print(f"dyn_vor EEN per-level error profile total err_norm="
          f"{np.sqrt(np.mean(err_by_level_vor**2))/np.sqrt(np.mean(rms_by_level_vor_nemo**2)):.4e}")
    print(f"dyn_ldf/ZAD per-level magnitude-profile correlation r={corr_r:.4f}: {unification_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
