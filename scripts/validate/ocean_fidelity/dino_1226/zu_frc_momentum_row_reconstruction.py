"""#1226 zu_frc structural question: are the barotropic rows DOWNSTREAM of the
momentum rows (dyn_ldf, dyn_adv ZAD, dyn_vor EEN, dyn_cor_2d)?

TASK (human): the campaign's two LARGEST DEBT rows are ``dyn_spg_ts puu_b``
(1.3e-2) and ``un_adv`` (8.4e-3) -- both downstream of ``zu_frc``
(``dynspg_ts.F90:335-337``: ``zu_frc = SUM(e3u_0*puu(Krhs)*umask)*r1_hu_0``,
the depth-mean of the 3-D momentum RHS). Several DEBT rows measure PIECES of
that same 3-D RHS: ``dyn_ldf u`` 3.9e-3, ``dyn_adv ZAD`` 4.9e-3, ``dyn_vor
EEN`` ~1.2e-3, ``dyn_cor_2d`` 1.2e-3 (RSS 6.49e-3, same order as zu_frc's
8.03e-3 -- an HINT recorded in fix_plan.md, never verified per-level).

``zu_frc_term_walk.py`` (commit e8040c430) CLOSED the "does legoESM lack
NEMO's zu_trd subtraction" branch ((b): present, wired, required) and ran a
REAL per-level error correlation for ONLY dyn_vor EEN (the one term with an
exposed diagnostic field, ``vortcor_u`` under ``vorticity_scheme=
"een_total"``); dyn_ldf/dyn_adv ZAD fell back to a magnitude-profile proxy
(r=0.49), explicitly labeled PLAUSIBLE, settling nothing.

A SEPARATE, unrelated probe (``momentum_jacobian_probe.py``, commit
1dfd4d9b8, fix_plan.md "2026-07-30e" B) tested a DIFFERENT hypothesis (does
the momentum-Jacobian h_u face-thickness convention explain zu_frc) and got
r=+0.11 -- REFUTED. That is NOT this task's reconstruction (it never summed
the four momentum-row errors NEMO's own zu_frc formula weights), so it does
not settle THIS question; it is cited here only so the two need not be
confused as overlapping runs.

THIS script closes the actual gap: as of this session, ``Ah_lap_u``/
``Ah_lap_v`` (dyn_ldf's NEMO nemo_div_curl -> nemo_ldf_lap_viscosity_cgrid
branch, ocean_pe_latlon_cgrid.py:2671-2733) and ``vertadv_u``/``vertadv_v``
(dyn_adv ZAD's nemo_advective branch, ocean_pe_latlon_cgrid.py:2523-2548,
importing ``nemo_advective_vertical_momentum_advection`` verbatim from
latlon_cgrid_operators.py) are ALREADY exposed on
``MomentumTendencyDiagnostics`` (state.py:722-748) and returned by
``model.tendencies_with_diagnostics()`` (ocean_model_latlon_cgrid.py:2594),
closure-tested to 1e-12 (test_momentum_diagnostics_closure.py). No production
change was needed -- git blame confirms both fields predate this session
(fc99d825b, 2026-07-28) and are untouched by the other lane's uncommitted
diff (which touches only ocean_model_latlon_cgrid.py/tke.py/k_profiles.py/
_shared.py/config.py -- verified via ``git status``/``git diff --stat``
before writing this script).

So Part 2 becomes: pull ``Ah_lap_u``/``vertadv_u`` alongside the ALREADY-
VALIDATED ``vortcor_u`` from the SAME ``tendencies_with_diagnostics()`` call
``zu_frc_term_walk.py`` makes, per-level-error them against NEMO's
``ldf_dump_du.bin``/``zad_dump_du.bin``/``vor_dump_du.bin``, add
``dyn_cor_2d`` (NEMO's ``cor2d_dump_zu_trd_substep1.bin`` -- a 2-D,
already-depth-mean field, see note at use site below), reconstruct
``SUM(e3u_0 * err * umask) * r1_hu_0`` exactly as NEMO forms zu_frc
(dynspg_ts.F90:335-337), and compare that reconstructed error field to
zu_frc's own measured 8.03e-3 (u) / 5.43e-4 (v) error.

Reuses ``zu_frc_term_walk.py``'s loaders/constants/spy machinery WHOLESALE
(imported, not re-derived) -- this is a controlled extension of that walk,
not a new harness.

RESULTS (this session, HEAD at time of run -- see the commit this file was
added in for the exact SHA): the fix_plan.md "RSS 6.49e-3" hint used
``1-corr``-style figures for dyn_ldf/dyn_adv ZAD (2.15e-3 / 8.0e-4, read off
the ``fidelity_bar_gate.py`` corr/ratio ledger), NOT the RMS-normalized
``err_norm`` metric ``zu_frc_term_walk.py`` uses for zu_frc itself. Measuring
dyn_ldf/dyn_adv ZAD with the SAME ``err_norm`` metric (same transform both
sides, per the oracle-fidelity rule) gives u: dyn_ldf=4.49e-2, dyn_adv
ZAD=4.00e-2, dyn_vor EEN=1.67e-3 (RSS=6.02e-2 -- ~7.5x zu_frc's 8.03e-3, NOT
"same order"; the earlier hint compared two different metrics). The
decisive reconstruction (``SUM(e3u_0*err*umask)*r1_hu_0`` on the SIGNED
per-term errors) gives:

    u: corr=0.044  ratio=0.045  -- REFUTED (no structural relationship)
    v: corr=0.985  ratio=0.977  -- CONFIRMED (strong, both corr and RMS match)

i.e. the unification claim is TRUE for v and FALSE for u -- confirming the
task's own warning not to assume u/v symmetry. u's error is NOT explained by
the three momentum rows; it shows a basin-scale wave-like latitude profile
(peaks near lat rows ~50-60/110-125/140-155, decaying toward the polar
row and toward both zonal walls) rather than the flat/uniform-interior +
wall-rise pattern of a boundary artifact -- a genuinely open, unexplained
signal for the next probe, not further chased here (out of this task's
scope: production-diagnostics-and-report, not root-causing zu_frc's u error).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zu_frc_momentum_row_reconstruction.py
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

# --- Reuse wholesale: the SAME loaders/constants zu_frc_term_walk.py built,
# imported (not copy-pasted) so this stays a controlled extension of that
# walk, per the module-reuse rule (pre-impl grep found this file first).
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior, _load_full, _load_full_3d,
)

register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345,367 zu_frc after the zu_trd subtraction.")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
register_dump("ldf_dump_du.bin", "now", "dynldf.F90:85 dyn_ldf_iso Krhs increment.")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")
register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment.")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")
register_dump("vor_dump_du.bin", "now", "dynvor.F90:151-155 vor_een Krhs increment.")
register_dump("vor_dump_dv.bin", "now", "same as vor_dump_du")
register_dump("cor2d_dump_zu_trd_substep1.bin", "now",
              "dynspg_ts.F90:760,778 zu_trd inside the substep loop at jn=1 -- "
              "a 2-D, ALREADY depth-mean field (there is no 3-D dyn_cor_2d "
              "term; the barotropic Coriolis is computed once on the "
              "depth-averaged transport, dynspg_ts.F90:364). Used here only "
              "as the depth-UNIFORM dyn_cor_2d contribution to the "
              "reconstructed zu_frc sum (added identically at every level's "
              "weight, matching how NEMO's own zu_frc build folds in a "
              "single 2-D correction alongside the depth-summed 3-D terms) "
              "-- NOT run through a per-level correlation (no per-level "
              "signal exists for a term that is 2-D by construction).")
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
    """RMS error at each level over the wet u/v faces -- same construction
    as zu_frc_term_walk.py's dyn_vor EEN block, factored so all four rows
    use the identical metric (Rule: same transform on both sides)."""
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


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_momentum_row_reconstruction")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vorticity_scheme={dcfg.vorticity_scheme!r}  "
          f"lateral_viscosity_operator={dcfg.lateral_viscosity_operator!r}  "
          f"vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vorticity_scheme == "een_total"
    assert dcfg.lateral_viscosity_operator == "nemo_div_curl", (
        "Ah_lap_u/v is only NEMO's dyn_ldf_lev_lap under 'nemo_div_curl' -- "
        "re-derive the ldf per-level comparison if this recipe field changes")
    assert dcfg.vertical_momentum_scheme == "nemo_advective", (
        "vertadv_u/v is only NEMO's dynzad advective form under "
        "'nemo_advective' -- re-derive if this recipe field changes")

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_momentum_row_reconstruction")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    # --- Capture the ACTUAL production F_slow_u/F_slow_v (zu_frc, WITH the
    # zu_trd subtraction already applied -- the SAME spy zu_frc_term_walk.py
    # uses) from a single model.step(), and the full momentum diagnostic
    # breakdown from tendencies_with_diagnostics() on the SAME twin state.
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
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
    assert "F_slow_u" in captured, "barotropic solver never called with a seed"

    F_slow_u_actual = np.asarray(captured["F_slow_u"])
    F_slow_v_actual = np.asarray(captured["F_slow_v"])

    _tend_diag, diag = model.tendencies_with_diagnostics(st, surface_forcing=sf, dt=DT)
    ah_lap_u_3d = np.asarray(diag.Ah_lap_u.data)
    ah_lap_v_3d = np.asarray(diag.Ah_lap_v.data)
    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)
    vortcor_u_3d = np.asarray(diag.vortcor_u.data)
    vortcor_v_3d = np.asarray(diag.vortcor_v.data)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    e3u_0 = np.asarray(g.e3u_0) if hasattr(g, "e3u_0") else None
    e3v_0 = np.asarray(g.e3v_0) if hasattr(g, "e3v_0") else None
    hu_0 = np.asarray(g.hu_0) if hasattr(g, "hu_0") else None
    hv_0 = np.asarray(g.hv_0) if hasattr(g, "hv_0") else None
    if e3u_0 is None or hu_0 is None:
        raise RuntimeError(
            "mesh_mask read did not expose e3u_0/hu_0 -- cannot reconstruct "
            "zu_frc's depth-mean the way NEMO forms it "
            "(SUM(e3u_0*err*umask)*r1_hu_0); STOP rather than approximate "
            "with a plain mean.")

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
    # cor2d_dump_zu_trd_substep1.bin is dumped on the FULL domain incl. halo
    # (56x203, hls=2 -- 11368 doubles = 56*203, NOT the interior-only 52x199
    # spg_dump_zu_frc.bin uses) -- confirmed by file size, not assumed.
    nemo_zu_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zu_trd_substep1.bin"), jpi, jpj, hls)
    nemo_zv_trd = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zv_trd_substep1.bin"), jpi, jpj, hls)

    ah_lap_u_f = _u_to_nemo(ah_lap_u_3d)
    ah_lap_v_f = _v_to_nemo(ah_lap_v_3d)
    vertadv_u_f = _u_to_nemo(vertadv_u_3d)
    vertadv_v_f = _v_to_nemo(vertadv_v_3d)
    vortcor_u_f = _u_to_nemo(vortcor_u_3d)
    vortcor_v_f = _v_to_nemo(vortcor_v_3d)

    n_lev_common = min(nemo_ldf_du.shape[2], nemo_zad_du.shape[2], nemo_vor_du.shape[2],
                        ah_lap_u_f.shape[2], vertadv_u_f.shape[2], vortcor_u_f.shape[2])

    print("\n" + "=" * 78)
    print("PART 2.1: per-level ERROR correlation, each row vs zu_frc's error shape")
    print("=" * 78)

    rows_u = {}
    rows_v = {}
    for label, lego3, nemo3, mask2, side in (
        ("dyn_ldf", ah_lap_u_f, nemo_ldf_du, umask2, "u"),
        ("dyn_adv ZAD", vertadv_u_f, nemo_zad_du, umask2, "u"),
        ("dyn_vor EEN", vortcor_u_f, nemo_vor_du, umask2, "u"),
        ("dyn_ldf", ah_lap_v_f, nemo_ldf_dv, vmask2, "v"),
        ("dyn_adv ZAD", vertadv_v_f, nemo_zad_dv, vmask2, "v"),
        ("dyn_vor EEN", vortcor_v_f, nemo_vor_dv, vmask2, "v"),
    ):
        err3, err_by_lvl, rms_by_lvl, mask_c = _per_level_err(lego3, nemo3, mask2, n_lev_common)
        tot_err_norm = (float(np.sqrt(np.mean(err_by_lvl ** 2)))
                         / float(np.sqrt(np.mean(rms_by_lvl ** 2))))
        print(f"  {label:<14s} {side}: total err_norm={tot_err_norm:.4e}  "
              f"err_by_level(first5)={np.array2string(err_by_lvl[:5], precision=3)}")
        d = rows_u if side == "u" else rows_v
        d[label] = dict(err3=err3, err_by_lvl=err_by_lvl, mask=mask_c, tot=tot_err_norm)

    def _corr_table(rows, side):
        print(f"\n  cross-row per-level ERROR-MAGNITUDE correlation ({side}), "
              "corrcoef of err_by_level across the 3 momentum rows:")
        labels = list(rows.keys())
        for i, li in enumerate(labels):
            for lj in labels[i + 1:]:
                r = float(np.corrcoef(rows[li]["err_by_lvl"], rows[lj]["err_by_lvl"])[0, 1])
                print(f"    corr(err_by_level[{li}], err_by_level[{lj}]) = {r:.4f}")

    _corr_table(rows_u, "u")
    _corr_table(rows_v, "v")

    print("\n" + "=" * 78)
    print("PART 2.2: THE DECISIVE TEST -- reconstruct zu_frc's error as NEMO")
    print("forms zu_frc: SUM(e3u_0 * err * umask) * r1_hu_0  (dynspg_ts.F90:335-337)")
    print("=" * 78)

    def _reconstruct_depth_mean_err(rows, e3_0, h_0, mask_c, side):
        """Sum the per-term 3-D error fields (NOT magnitudes -- signed
        errors, so cancellation is possible and honestly reflected) and
        depth-weight EXACTLY as NEMO's zu_frc formula does."""
        n_lat_c, n_lon_c, n_lev_c = rows[next(iter(rows))]["err3"].shape
        e3_c = e3_0[:n_lat_c, :n_lon_c, :n_lev_c] if e3_0.ndim == 3 else \
            np.broadcast_to(e3_0[None, None, :n_lev_c], (n_lat_c, n_lon_c, n_lev_c))
        h_c = h_0[:n_lat_c, :n_lon_c]
        total_err3 = sum(r["err3"] for r in rows.values())
        depth_sum = np.sum(e3_c * total_err3, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            recon = np.where(h_c > 0, depth_sum / np.where(h_c > 0, h_c, 1.0), 0.0)
        return recon

    e3u_c = e3u_0[..., :n_lev_common] if e3u_0.ndim == 3 else e3u_0[:n_lev_common]
    e3v_c = e3v_0[..., :n_lev_common] if (e3v_0 is not None and e3v_0.ndim == 3) else e3v_0

    recon_u = _reconstruct_depth_mean_err(rows_u, e3u_c, hu_0, umask2, "u")
    recon_v = (_reconstruct_depth_mean_err(rows_v, e3v_c, hv_0, vmask2, "v")
               if e3v_0 is not None and hv_0 is not None else None)

    # zu_frc's OWN measured error (actual production value vs NEMO's dump).
    zu_frc_err = _u_to_nemo(F_slow_u_actual) - nemo_zu_frc
    zv_frc_err = _v_to_nemo(F_slow_v_actual) - nemo_zv_frc

    n_lat_z = min(recon_u.shape[0], zu_frc_err.shape[0])
    n_lon_z = min(recon_u.shape[1], zu_frc_err.shape[1])
    m_u = umask2[:n_lat_z, :n_lon_z]
    ru = recon_u[:n_lat_z, :n_lon_z][m_u]
    zu = zu_frc_err[:n_lat_z, :n_lon_z][m_u]
    corr_u = float(np.corrcoef(ru, zu)[0, 1]) if ru.size > 1 else float("nan")
    rms_recon_u = float(np.sqrt(np.mean(ru ** 2)))
    rms_zu = float(np.sqrt(np.mean(zu ** 2)))
    ratio_u = rms_recon_u / rms_zu if rms_zu > 0 else float("nan")

    print(f"\nu: reconstructed-depth-mean(dyn_ldf+dyn_adv_ZAD+dyn_vor_EEN error) "
          f"RMS={rms_recon_u:.4e}  vs  zu_frc measured error RMS={rms_zu:.4e}  "
          f"ratio={ratio_u:.3f}  corr={corr_u:.4f}")
    print("  (dyn_cor_2d NOT included in the u reconstruction above -- see note: "
          "it is a 2-D depth-mean quantity by construction, added separately below)")

    # dyn_cor_2d: already depth-mean (2-D), so it enters the reconstruction as
    # a flat additive term, not a per-level contributor -- add it and report
    # both with/without so the reader can see its (small, 1.2e-3) contribution
    # without conflating a 2-D term into a "per-level correlation" claim.
    zu_trd_lego = None
    zv_trd_lego = None
    try:
        import legoesm.ocean.dynamics.barotropic_latlon_cgrid as bmod
        captured2 = {}
        _real_pre_step = bmod.barotropic_coriolis_een_pre_step

        def _spy_pre_step(*a, **kw):
            cor_u, cor_v = _real_pre_step(*a, **kw)
            if "cor_u_sub" not in captured2:
                captured2["cor_u_sub"] = cor_u
                captured2["cor_v_sub"] = cor_v
            return cor_u, cor_v

        bmod.barotropic_coriolis_een_pre_step = _spy_pre_step
        try:
            with jax.disable_jit():
                _ = model.step(st, DT, surface_forcing=sf)
        finally:
            bmod.barotropic_coriolis_een_pre_step = _real_pre_step
        zu_trd_lego = _u_to_nemo(np.asarray(captured2["cor_u_sub"]))
        zv_trd_lego = _v_to_nemo(np.asarray(captured2["cor_v_sub"]))
    except Exception as exc:  # noqa: BLE001 -- report, do not silently drop
        print(f"  [dyn_cor_2d capture failed: {exc!r} -- reported without it]")

    if zu_trd_lego is not None:
        cor2d_err_u = zu_trd_lego[:n_lat_z, :n_lon_z] - nemo_zu_trd[:n_lat_z, :n_lon_z]
        ru_with_cor2d = ru + cor2d_err_u[m_u]
        rms_recon_u_with_cor2d = float(np.sqrt(np.mean(ru_with_cor2d ** 2)))
        corr_u_with_cor2d = (float(np.corrcoef(ru_with_cor2d, zu)[0, 1])
                              if ru_with_cor2d.size > 1 else float("nan"))
        ratio_u_with_cor2d = rms_recon_u_with_cor2d / rms_zu if rms_zu > 0 else float("nan")
        print(f"u (+ dyn_cor_2d 2-D term added): reconstructed RMS="
              f"{rms_recon_u_with_cor2d:.4e}  ratio={ratio_u_with_cor2d:.3f}  "
              f"corr={corr_u_with_cor2d:.4f}")

    if recon_v is not None:
        n_lat_zv = min(recon_v.shape[0], zv_frc_err.shape[0])
        n_lon_zv = min(recon_v.shape[1], zv_frc_err.shape[1])
        m_v = vmask2[:n_lat_zv, :n_lon_zv]
        rv = recon_v[:n_lat_zv, :n_lon_zv][m_v]
        zv = zv_frc_err[:n_lat_zv, :n_lon_zv][m_v]
        corr_v = float(np.corrcoef(rv, zv)[0, 1]) if rv.size > 1 else float("nan")
        rms_recon_v = float(np.sqrt(np.mean(rv ** 2)))
        rms_zv = float(np.sqrt(np.mean(zv ** 2)))
        ratio_v = rms_recon_v / rms_zv if rms_zv > 0 else float("nan")
        print(f"v: reconstructed-depth-mean(dyn_ldf+dyn_adv_ZAD+dyn_vor_EEN error) "
              f"RMS={rms_recon_v:.4e}  vs  zv_frc measured error RMS={rms_zv:.4e}  "
              f"ratio={ratio_v:.3f}  corr={corr_v:.4f}")
        if zv_trd_lego is not None:
            cor2d_err_v = zv_trd_lego[:n_lat_zv, :n_lon_zv] - nemo_zv_trd[:n_lat_zv, :n_lon_zv]
            rv_with_cor2d = rv + cor2d_err_v[m_v]
            rms_recon_v_with_cor2d = float(np.sqrt(np.mean(rv_with_cor2d ** 2)))
            corr_v_with_cor2d = (float(np.corrcoef(rv_with_cor2d, zv)[0, 1])
                                  if rv_with_cor2d.size > 1 else float("nan"))
            ratio_v_with_cor2d = rms_recon_v_with_cor2d / rms_zv if rms_zv > 0 else float("nan")
            print(f"v (+ dyn_cor_2d 2-D term added): reconstructed RMS="
                  f"{rms_recon_v_with_cor2d:.4e}  ratio={ratio_v_with_cor2d:.3f}  "
                  f"corr={corr_v_with_cor2d:.4f}")
    else:
        print("\nv: SKIPPED -- mesh_mask did not expose e3v_0/hv_0 for the v-face "
              "depth weighting; report as UNSETTLED for v, do not assume u/v symmetry.")
        corr_v = None
        ratio_v = None

    print("\n" + "=" * 78)
    print("Where does zu_frc's OWN error concentrate (for the next probe if REFUTED)?")
    print("=" * 78)
    print(f"  zu_frc err field: mean={float(np.mean(zu)):.4e}  "
          f"|max|={float(np.max(np.abs(zu))):.4e}  std={float(np.std(zu)):.4e}")
    # Spatial concentration, reported as a real profile (not a flat index list
    # -- a masked 1-D top-10 gives no usable direction for the next probe).
    zu_full = zu_frc_err[:n_lat_z, :n_lon_z]
    col_mean_abs = np.array([
        float(np.mean(np.abs(zu_full[:, j][m_u[:, j]]))) if m_u[:, j].any() else float("nan")
        for j in range(n_lon_z)
    ])
    row_mean_abs = np.array([
        float(np.mean(np.abs(zu_full[i, :][m_u[i, :]]))) if m_u[i, :].any() else float("nan")
        for i in range(n_lat_z)
    ])
    print(f"  |err| by longitude column (0={n_lon_z-1}): "
          f"{np.array2string(col_mean_abs, precision=2, max_line_width=200)}")
    print(f"  |err| by latitude row, every 5th (0..{n_lat_z-1}): "
          f"{np.array2string(row_mean_abs[::5], precision=2, max_line_width=200)}")

    # ponytail: raw numbers only, no baked-in verdict label here -- CLAUDE.md's
    # instrumentation rule ("never let a probe print its own verdict") applies;
    # interpretation belongs in the analysis/report after reading these numbers,
    # not in the tool. See this file's module docstring RESULTS section for the
    # CONFIRMED/REFUTED conclusion actually drawn from them.
    print("\n" + "=" * 78)
    print("SUMMARY (raw numbers -- see module docstring RESULTS for interpretation)")
    print("=" * 78)
    print(f"u: corr={corr_u:.4f}  ratio={ratio_u:.4f}")
    print(f"v: corr={corr_v:.4f}  ratio={ratio_v:.4f}" if corr_v is not None
          else "v: UNSETTLED (no e3v_0/hv_0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
