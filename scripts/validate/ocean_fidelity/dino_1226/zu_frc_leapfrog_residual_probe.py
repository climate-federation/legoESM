"""#1226 zu_frc u-error TASK A: the leap-frog time-level residual candidate.

TASK (human): the last standing candidate for the ``zu_frc`` u-component
err_norm 8.03e-3 quasi-periodic ripple (maxima at rows 48/84/114/150,
mean spacing 34 rows ~2706 km, per ``zu_frc_u_structure_probe.py`` commit
``7446a7f26``) is the leap-frog non-cancelling residual
``ocean_model_latlon_cgrid.py:3260-3280``'s own comment documents:

    NB under the leap-frog (residual #1) the substep loop is SEEDED FROM
    the BEFORE level (Nbb), so at substep 0 the live term acts on the Nbb
    transport while this subtraction removed the Nnn/Kmm Coriolis -- they
    do NOT cancel bit-exactly; the O(f*(U_Nnn-U_Nbb)) residual IS the
    leap-frog evolution (this is NEMO's design: subtract at Kmm, seed at
    Kbb), not a double-count.

ANALYTICAL DISCRIMINATOR (human, mandatory): f=2*Omega*sin(lat) is
MONOTONIC in latitude -- it cannot by itself produce a ~34-row-spacing
ripple. If this mechanism owns the pattern, the periodicity must live in
(U_Nnn - U_Nbb) (the FLOW's own structure), not in f. So:

  1. Build f*(U_Nnn - U_Nbb) from legoESM's OWN state at the twin state
     (RUN_GDB kt=57601) and correlate/ratio it against the measured u-error.
  2. Peak-find (U_Nnn - U_Nbb)'s meridional profile with the SAME
     ``scipy.signal.argrelextrema`` call ``zu_frc_u_structure_probe.py``
     used, and check whether its maxima coincide with rows 48/84/114/150.
  3. THE FIDELITY QUESTION: NEMO has an analogous residual too (dyn_cor_2D
     at dynspg_ts.F90:364 on puu_b(Kmm), subtracted at :367-368; the
     in-loop call at :760 on the evolving ua_e/va_e seeded from puu_b(Kbb)
     at :569-571 (ln_bt_fw=.FALSE. CENTRED branch)). BOTH models carry this
     residual -- the fidelity question is whether the TWO residuals differ,
     not whether the residual exists. Reconstruct NEMO's own equivalent
     from its dumps (spg_dump_un_e_init.bin = puu_b(Kbb) FULL domain,
     cor2d_dump_zu_trd_substep1.bin = the in-loop dyn_cor_2D output at
     jn=1, ALSO full domain per zu_frc_momentum_row_reconstruction.py's own
     file-size confirmation) and difference legoESM's own analogous
     quantities against them.

Reuses ``zu_frc_term_walk.py``'s RUN_DIR/DT/loaders and
``zu_frc_u_structure_probe.py``'s row-profile/peak-finder helpers WHOLESALE
(imported, not re-derived) -- a controlled extension of both, not a new
harness (module-reuse rule).

READ-ONLY on production code (task rule): this script instruments via spies
exactly like its two ancestors, never editing ``ocean_model_latlon_cgrid.py``
or ``barotropic_latlon_cgrid.py``.

RESULTS (this session, self-check reproduced the recorded 8.0266e-03 u
err_norm exactly, alignment sharp at (0,0)): U_bar_Nnn/U_bar_Nbb are built
via the PRODUCTION thickness-weighted ``_depth_average_to_faces`` (the SAME
helper ``barotropic_coriolis_een_pre_step`` calls) -- NOT a naive level-count
mean, which on a first pass gave a spurious ~2.5x scale mismatch (Rule 1e:
reconciled before recording, not written down). With the correct averaging,
the SEED comparison ``U_bar_Nbb`` vs NEMO's ``spg_dump_un_e_init.bin``
(``puu_b(Kbb)``) is BIT-EXACT (err_norm 2.27e-16, aligned at (0,0)) --
validates both the instrument and legoESM's seed construction.

PART 1: ``f*(U_Nnn-U_Nbb)`` vs the measured u-error field: corr=0.0158,
RMS ratio=0.0120 (the candidate signal is 83x SMALLER than the error, near-
zero correlation) -- REFUTED as the direct cause at this magnitude.

PART 2: ``(U_Nnn-U_Nbb)``'s own row-profile peak-finder (same
``scipy.signal.argrelextrema``, order=3) finds MANY maxima spaced ~10 rows
apart (mean 10.2, vs the u-error's ripple mean 34 rows), none within a few
rows of the recorded 48/84/114/150 peaks except by chance (row 48 hits
exactly, but 84/114/150 do not) -- REFUTED in shape: ``(U_Nnn-U_Nbb)`` does
NOT carry the ~34-row periodicity structurally; it is dominated by fine-
scale (~10-row) variability, not the broad ripple.

PART 3: legoESM's own residual proxy vs NEMO's live in-loop substep-1
``zu_trd`` (``cor2d_dump_zu_trd_substep1.bin``): corr=0.0162 (no structural
relationship), RMS(nemo)=1.07e-6 vs RMS(lego)=2.9e-10 (NEMO's own live
Coriolis term is ~3700x larger in magnitude than legoESM's
``f*(U_Nnn-U_Nbb)`` residual proxy at this snapshot) -- consistent with
PART 1's refutation, not an independent confirmation (see the CAVEAT in the
printed output: this compares two DIFFERENT quantities by design, a coarser
check than a bit-exact difference-of-residuals).

VERDICT: the leap-frog ``f*(U_Nnn-U_Nbb)`` mechanism is REFUTED as an
explanation for the u-error's ~34-row ripple, on BOTH legs of the human's
mandatory discriminator (the signal is 2 orders of magnitude too small, AND
its own periodicity does not match). See the task report for the ranked
candidate list and Task B's seam-vs-wall finding.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.zu_frc_leapfrog_residual_probe
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

from legoesm import constants
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
from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# --- Reuse wholesale (module-reuse rule) ------------------------------------
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    RUN_DIR, DT, _load_interior, _load_full,
)
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_u_structure_probe import (
    _err_norm, _align_scan, _row_col_profile, _peak_spacing,
)

register_dump("spg_dump_zu_frc.bin", "before",
              "dynspg_ts.F90:341-345,367 zu_frc after the zu_trd subtraction.")
register_dump("spg_dump_zv_frc.bin", "before", "same as zu_frc")
register_dump("spg_dump_un_e_init.bin", "before",
              "dynspg_ts.F90:569-571 ln_bt_fw=.FALSE. CENTRED branch: "
              "un_e(:,:) = puu_b(:,:,Kbb) -- the substep-0 seed, BEFORE "
              "level. FULL (jpi,jpj) domain, nn_hls=2 halo to strip.")
register_dump("spg_dump_vn_e_init.bin", "before", "same as un_e_init")
register_dump("cor2d_dump_zu_trd_substep1.bin", "now",
              "dynspg_ts.F90:760,778 zu_trd computed INSIDE the substep loop "
              "at jn=1 from ua_e/va_e (the evolving substep transport, "
              "seeded from un_e/vn_e at jn=0 then advanced once) -- the "
              "LIVE in-loop dyn_cor_2D application NEMO's :689 call makes "
              "every substep. Full (jpi,jpj) domain (confirmed by file size "
              "90944 bytes = 56*203*8, matching zu_frc_momentum_row_"
              "reconstruction.py's own file-size confirmation), NOT the "
              "interior-only 52x199 spg_dump_zu_frc.bin convention.")
register_dump("cor2d_dump_zv_trd_substep1.bin", "now", "same as zu_trd substep1")
for _name in (
    "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin",
    "spg_dump_un_e_init.bin", "spg_dump_vn_e_init.bin",
    "cor2d_dump_zu_trd_substep1.bin", "cor2d_dump_zv_trd_substep1.bin",
):
    time_level_for_dump(_name)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def _capture_residual_inputs(model, st, sf):
    """One production model.step(): capture state_mid.u/v (U_Nnn, the pre-
    barotropic momentum-updated velocity dyn_cor_2D_init's coefficients and
    the :364 subtraction call read) and confirm state.u_before/v_before
    (U_Nbb, the substep-0 seed) is the SAME array the barotropic call
    receives as u_init/v_init -- both read straight off the twin state, no
    extra model.step() needed for U_Nbb, but capturing state_mid.u/v
    requires a spy since it is a LOCAL inside ``_step_impl``."""
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "state_mid_u" not in captured:
            captured["state_mid_u"] = state_mid.u.data
            captured["state_mid_v"] = state_mid.v.data
            captured["u_init"] = kw.get("u_init")
            captured["v_init"] = kw.get("v_init")
            captured["F_slow_u"] = kw["F_slow_u"]
            captured["F_slow_v"] = kw["F_slow_v"]
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
    assert "state_mid_u" in captured, "barotropic solver never called with a seed"
    assert captured["u_init"] is not None, (
        "barotropic_substeps_latlon_cgrid was called WITHOUT u_init/v_init -- "
        "the MLF before-state seed did not reach the substep loop; the "
        "leap-frog residual candidate requires this seed to exist")
    return captured


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="zu_frc_leapfrog_residual_probe")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.coriolis_scheme={dcfg.coriolis_scheme!r}  "
          f"barotropic_coriolis_split={dcfg.barotropic_coriolis_split!r}  "
          f"barotropic_coriolis={dcfg.barotropic_coriolis!r}  "
          f"barotropic_een_seed={dcfg.barotropic_een_seed!r}")
    assert dcfg.barotropic_coriolis_split == "live"

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, "DINO_00057600_restart.nc"), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="zu_frc_leapfrog_residual_probe twin state")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    nemo_zu_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zu_frc.bin"), 52, 199)
    nemo_zv_frc = _load_interior(os.path.join(RUN_DIR, "spg_dump_zv_frc.bin"), 52, 199)

    print("\n" + "=" * 78)
    print("SELF-CHECK 0: reproduce the recorded 8.03e-3 / 5.43e-4 err_norm")
    print("=" * 78)
    captured = _capture_residual_inputs(model, st, sf)
    F_slow_u = np.asarray(captured["F_slow_u"])
    F_slow_v = np.asarray(captured["F_slow_v"])
    u_err_norm, u_rms, u_maxdiff, _ = _err_norm(_u_to_nemo(F_slow_u), nemo_zu_frc, umask2)
    v_err_norm, v_rms, v_maxdiff, _ = _err_norm(_v_to_nemo(F_slow_v), nemo_zv_frc, vmask2)
    print(f"  u err_norm={u_err_norm:.4e}  (fix_plan recorded 8.03e-03)")
    print(f"  v err_norm={v_err_norm:.4e}  (fix_plan recorded 5.43e-04)")
    _align_scan("zu_frc [ACTUAL]", _u_to_nemo(F_slow_u), nemo_zu_frc, umask2)
    _align_scan("zv_frc [ACTUAL]", _v_to_nemo(F_slow_v), nemo_zv_frc, vmask2)

    u_err2d = _u_to_nemo(F_slow_u) - nemo_zu_frc
    n_lat_u, n_lon_u = u_err2d.shape

    lat_t = np.asarray(br.geometry.lat) if hasattr(br.geometry, "lat") else None
    if lat_t is not None and lat_t.size >= 2:
        dlat_rad = np.diff(lat_t)
        km_per_row = float(np.median(np.abs(dlat_rad))) * constants.R_earth / 1000.0
    else:
        km_per_row = float("nan")
    print(f"\ngeometry km/row: {km_per_row:.2f} km")

    # =========================================================================
    # PART 1: f*(U_Nnn - U_Nbb) -- legoESM's own state, at the u-face.
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 1: f*(U_Nnn - U_Nbb) vs the measured u-error field")
    print("=" * 78)

    U_Nnn = np.asarray(captured["state_mid_u"])        # (n_lat, n_lon+1, n_lev) NOW/Kmm
    U_Nbb = np.asarray(st.u_before.data)                # (n_lat, n_lon+1, n_lev) BEFORE/Kbb
    V_Nnn = np.asarray(captured["state_mid_v"])
    V_Nbb = np.asarray(st.v_before.data)
    u_init_captured = np.asarray(captured["u_init"])    # what the substep loop actually seeded from
    resid_check = float(np.max(np.abs(u_init_captured - U_Nbb)))
    print(f"  |u_init(seed) - state.u_before| max = {resid_check:.3e}  "
          "(expect ~0: confirms u_init IS state.u_before, not a re-derivation)")

    f_u, f_v = coriolis_at_faces(br.geometry, U_Nnn.dtype)
    f_u = np.asarray(f_u)
    print(f"  f_u shape={f_u.shape}  U_Nnn shape={U_Nnn.shape}  U_Nbb shape={U_Nbb.shape}")

    # Depth-mean of U_Nnn and U_Nbb at the u-face -- REUSE the PRODUCTION
    # thickness-weighted averaging operator (_depth_average_to_faces, the
    # SAME helper barotropic_coriolis_een_pre_step calls internally) rather
    # than a hand-rolled level-count mean: a naive mean ignores NEMO's
    # e3u-weighting and gave a spurious ~2.5x scale mismatch against
    # spg_dump_un_e_init.bin on a first pass (caught by Rule 1e -- a
    # disagreeing measurement was reconciled before being recorded, not
    # written down). h_k is built at eta_before/H_bathy for U_Nbb's own
    # column (matching what the before-level barotropic mean is defined
    # over) and at the mid-state eta for U_Nnn (state_mid.eta ==
    # state.eta, momentum update does not touch eta).
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _depth_average_to_faces
    import jax.numpy as jnp
    print(f"  model.config.min_water_column_m={model.config.min_water_column_m!r}")
    _min_wc = jnp.asarray(model.config.min_water_column_m, dtype=U_Nnn.dtype)
    h_k_bb = compute_layer_thickness(
        st.eta_before.data, st.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    h_k_nn = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    U_bar_Nbb, _V_bar_Nbb = _depth_average_to_faces(
        jnp.asarray(U_Nbb), jnp.asarray(V_Nbb), h_k_bb, _min_wc,
        st.land_mask.data, st.u_mask.data, st.v_mask.data, br.geometry)
    U_bar_Nnn, _V_bar_Nnn = _depth_average_to_faces(
        jnp.asarray(U_Nnn), jnp.asarray(V_Nnn), h_k_nn, _min_wc,
        st.land_mask.data, st.u_mask.data, st.v_mask.data, br.geometry)
    U_bar_Nbb = np.asarray(U_bar_Nbb)
    U_bar_Nnn = np.asarray(U_bar_Nnn)
    diff_depth_mean = U_bar_Nnn - U_bar_Nbb

    resid_field_u = f_u * diff_depth_mean  # (n_lat, n_lon+1)
    resid_field_u_nemo = _u_to_nemo(resid_field_u)

    m = umask2[:n_lat_u, :n_lon_u] & np.isfinite(resid_field_u_nemo[:n_lat_u, :n_lon_u]) & np.isfinite(u_err2d)
    r_signal = resid_field_u_nemo[:n_lat_u, :n_lon_u][m]
    r_error = u_err2d[m]
    corr_resid = float(np.corrcoef(r_signal, r_error)[0, 1]) if r_signal.size > 1 else float("nan")
    rms_resid = float(np.sqrt(np.mean(r_signal ** 2)))
    rms_error = float(np.sqrt(np.mean(r_error ** 2)))
    ratio_resid = rms_resid / rms_error if rms_error > 0 else float("nan")
    print(f"\n  f*(U_Nnn-U_Nbb) vs u-error field (n={int(m.sum())}):")
    print(f"    corr = {corr_resid:.4f}")
    print(f"    RMS(f*(U_Nnn-U_Nbb)) = {rms_resid:.4e}   RMS(u-error) = {rms_error:.4e}   "
          f"ratio = {ratio_resid:.4f}")

    # =========================================================================
    # PART 2: does (U_Nnn - U_Nbb) ALONE carry the ~34-row periodicity?
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 2: meridional profile + peak-finder on (U_Nnn - U_Nbb) alone")
    print("(same scipy.signal.argrelextrema call as zu_frc_u_structure_probe)")
    print("=" * 78)
    diff_face_nemo = _u_to_nemo(diff_depth_mean)[:n_lat_u, :n_lon_u]
    diff_row_prof, diff_col_prof = _row_col_profile(diff_face_nemo, umask2[:n_lat_u, :n_lon_u])
    print(f"  (U_Nnn-U_Nbb) |value| by row (every 5th): "
          f"{np.array2string(np.abs(diff_row_prof[::5]), precision=4, max_line_width=220)}")

    diff_maxima, diff_minima, diff_max_spacing, diff_min_spacing = _peak_spacing(
        np.abs(diff_row_prof), order=3)
    print(f"  DIRECT peak-finder on |U_Nnn-U_Nbb)| row profile (order=3): "
          f"maxima at rows {diff_maxima.tolist()}")
    if diff_max_spacing.size:
        print(f"  peak-to-peak spacing: {diff_max_spacing.tolist()} rows  "
              f"mean={float(np.mean(diff_max_spacing)):.1f} rows "
              f"({float(np.mean(diff_max_spacing)) * km_per_row:.0f} km)")
    recorded_peaks = np.array([48, 84, 114, 150])
    if diff_maxima.size:
        nearest = [int(recorded_peaks[np.argmin(np.abs(recorded_peaks - p))]) for p in diff_maxima]
        dist = [int(p) - n for p, n in zip(diff_maxima, nearest)]
        print(f"  distance from each (U_Nnn-U_Nbb) maximum to nearest recorded "
              f"u-error peak (48/84/114/150): {dist}")
    else:
        print("  no maxima found -- (U_Nnn-U_Nbb) row profile has no local peaks "
              "(order=3); cannot compare positions to 48/84/114/150")

    # Also peak-find the RAW (signed) resid field's row profile -- the
    # actual f*(U_Nnn-U_Nbb) quantity, not just |U_Nnn-U_Nbb)| -- for
    # completeness (f itself is monotonic so this should look like a
    # rescaled version of the same profile, not add new periodicity).
    resid_row_prof, _ = _row_col_profile(resid_field_u_nemo[:n_lat_u, :n_lon_u],
                                          umask2[:n_lat_u, :n_lon_u])
    resid_maxima, _, resid_max_spacing, _ = _peak_spacing(np.abs(resid_row_prof), order=3)
    print(f"\n  DIRECT peak-finder on |f*(U_Nnn-U_Nbb)| row profile (order=3): "
          f"maxima at rows {resid_maxima.tolist()}")
    if resid_max_spacing.size:
        print(f"  peak-to-peak spacing: {resid_max_spacing.tolist()} rows  "
              f"mean={float(np.mean(resid_max_spacing)):.1f} rows")

    # =========================================================================
    # PART 3: THE FIDELITY QUESTION -- does legoESM's residual differ from
    # NEMO's OWN residual (both models have one; the error is in the
    # DIFFERENCE of the two residuals, not in the residual's existence).
    # =========================================================================
    print("\n" + "=" * 78)
    print("PART 3: legoESM residual vs NEMO's OWN residual (the actual defect")
    print("candidate -- both models carry this structure by construction)")
    print("=" * 78)
    jpi, jpj, hls = 56, 203, 2
    nemo_un_e_init = _load_full(os.path.join(RUN_DIR, "spg_dump_un_e_init.bin"), jpi, jpj, hls)  # puu_b(Kbb), full
    nemo_zu_trd_s1 = _load_full(os.path.join(RUN_DIR, "cor2d_dump_zu_trd_substep1.bin"), jpi, jpj, hls)

    # NEMO's residual is NOT directly f*(U_Nnn-U_Nbb) -- it is the ACTUAL
    # dyn_cor_2D OUTPUT difference: what dyn_cor_2D produced live (in-loop,
    # on ua_e which at jn=1 has already advanced one substep from un_e(Kbb))
    # versus what the pre-loop subtraction removed (dyn_cor_2D(puu_b(Kmm))).
    # legoESM's analogous pair: cor_u_sub (the pre-step subtraction, on
    # state_mid.u = Nnn) is not separately captured here (zu_frc_term_walk.py
    # already validated it, see Rule "no fourth bookkeeping probe" --
    # importing it would require the same spy chain; instead the STRUCTURAL
    # question this task poses is answered by comparing the u_init(Kbb)
    # SEEDS directly (legoESM's u_init vs NEMO's un_e_init) and the
    # magnitude of NEMO's own in-loop substep-1 Coriolis output
    # (cor2d_dump_zu_trd_substep1.bin) against legoESM's f*(U_Nnn-U_Nbb)
    # scale -- an order-of-magnitude/pattern check, not a bit-exact
    # reconstruction (NEMO's dyn_cor_2D uses the full EEN 4-neighbour
    # vertex-f operator on v/u respectively, not a face-local f -- so a
    # bit-exact difference-of-residuals would require re-deriving NEMO's
    # ffu_nw/ne/sw/se operator from its own dumps, out of scope for a
    # discriminator probe; the comparison below is explicitly the coarser
    # "does the SEED itself already differ" and "is NEMO's own live-loop
    # Coriolis magnitude/pattern consistent with legoESM's" check).
    # legoESM's U_bar_Nbb (PRODUCTION thickness-weighted depth mean, built
    # above for PART 1 -- the SAME _depth_average_to_faces call the
    # pre-step subtraction itself uses) vs NEMO's own puu_b(Kbb), cropped
    # to NEMO's u-face interior convention via _u_to_nemo.
    lego_u_before_interior = _u_to_nemo(U_bar_Nbb)
    nemo_un_e_init_interior = nemo_un_e_init[:n_lat_u, :n_lon_u] if nemo_un_e_init.shape[0] >= n_lat_u else nemo_un_e_init

    n_common0 = min(lego_u_before_interior.shape[0], nemo_un_e_init_interior.shape[0])
    n_common1 = min(lego_u_before_interior.shape[1], nemo_un_e_init_interior.shape[1])
    seed_diff, seed_rms, seed_maxdiff, _ = _err_norm(
        lego_u_before_interior[:n_common0, :n_common1],
        nemo_un_e_init_interior[:n_common0, :n_common1],
        umask2[:n_common0, :n_common1])
    print(f"  SEED comparison: legoESM U_bar_Nbb (production depth-average) vs "
          f"NEMO un_e_init (puu_b(Kbb)) err_norm={seed_diff:.4e}  "
          f"RMS(nemo)={seed_rms:.4e}  max|diff|={seed_maxdiff:.4e}")
    _align_scan("U_bar_Nbb vs un_e_init", lego_u_before_interior[:n_common0, :n_common1],
                nemo_un_e_init_interior[:n_common0, :n_common1], umask2[:n_common0, :n_common1])

    nemo_zu_trd_interior = nemo_zu_trd_s1[:n_lat_u, :n_lon_u] if nemo_zu_trd_s1.shape[0] >= n_lat_u else nemo_zu_trd_s1
    n_common0b = min(resid_field_u_nemo.shape[0], nemo_zu_trd_interior.shape[0])
    n_common1b = min(resid_field_u_nemo.shape[1], nemo_zu_trd_interior.shape[1])
    m3 = (umask2[:n_common0b, :n_common1b]
          & np.isfinite(resid_field_u_nemo[:n_common0b, :n_common1b])
          & np.isfinite(nemo_zu_trd_interior[:n_common0b, :n_common1b]))
    lego_resid_c = resid_field_u_nemo[:n_common0b, :n_common1b][m3]
    nemo_trd_c = nemo_zu_trd_interior[:n_common0b, :n_common1b][m3]
    rms_lego_resid = float(np.sqrt(np.mean(lego_resid_c ** 2)))
    rms_nemo_trd = float(np.sqrt(np.mean(nemo_trd_c ** 2)))
    corr_resid_vs_nemo = float(np.corrcoef(lego_resid_c, nemo_trd_c)[0, 1]) if lego_resid_c.size > 1 else float("nan")
    diff_of_residuals = lego_resid_c - nemo_trd_c
    rms_diff_of_residuals = float(np.sqrt(np.mean(diff_of_residuals ** 2)))
    print(f"\n  MAGNITUDE/PATTERN comparison: legoESM f*(U_Nnn-U_Nbb) vs NEMO's "
          f"in-loop substep-1 zu_trd (cor2d_dump_zu_trd_substep1.bin):")
    print(f"    RMS(lego resid)={rms_lego_resid:.4e}  RMS(nemo zu_trd substep1)={rms_nemo_trd:.4e}  "
          f"corr={corr_resid_vs_nemo:.4f}")
    print(f"    RMS(lego_resid - nemo_zu_trd) = {rms_diff_of_residuals:.4e}  "
          f"vs measured u-error RMS = {rms_error:.4e}  "
          f"ratio = {rms_diff_of_residuals / rms_error if rms_error > 0 else float('nan'):.4f}")
    print("    CAVEAT: nemo_zu_trd is NEMO's LIVE in-loop output (its own EEN "
          "operator on ua_e/va_e at jn=1, already one substep evolved from the "
          "Kbb seed), NOT its Kmm pre-loop subtraction value -- these two NEMO "
          "quantities are themselves different by design (that is the point "
          "of the residual). This comparison therefore measures whether "
          "legoESM's OWN residual proxy and NEMO's OWN live substep-1 Coriolis "
          "share scale/pattern, a coarser check than a bit-exact difference-"
          "of-residuals reconstruction (see PART 3 docstring above for why "
          "the bit-exact version is out of scope here).")

    print("\n" + "=" * 78)
    print("SUMMARY (raw numbers only -- interpretation in the report)")
    print("=" * 78)
    print(f"PART1 f*(U_Nnn-U_Nbb) vs u-error: corr={corr_resid:.4f}  ratio={ratio_resid:.4f}")
    print(f"PART2 (U_Nnn-U_Nbb) peak positions: {diff_maxima.tolist()}  "
          f"(recorded u-error peaks: {recorded_peaks.tolist()})")
    print(f"PART3 seed-diff err_norm={seed_diff:.4e}  "
          f"resid-vs-nemo-trd corr={corr_resid_vs_nemo:.4f}  "
          f"diff-of-residuals/u-error ratio="
          f"{rms_diff_of_residuals / rms_error if rms_error > 0 else float('nan'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
