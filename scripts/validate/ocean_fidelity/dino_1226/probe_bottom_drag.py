"""#1226 / #1455 QUEUE ITEM 2: rebuild the missing ``dyn_drg_init RHS
increment`` provenance probe.

The gate row's cited script (``probe_bottom_drag.py``) was ``scripts/tmp/``
scratch, deleted before being committed (STALE-TUPLE SWEEP 2026-08-03: "4
rows' cited probes absent from disk"). This is a from-scratch
reconstruction of that measurement, NOT a copy of lost code -- it follows
the sibling ``zu_frc_leapfrog_residual_probe.py``/``zu_frc_term_walk.py``
structure (same RUN_DIR/DT, same ``_err_norm``/``_align_scan`` self-checks)
rather than re-deriving anything.

WHAT THIS MEASURES: NEMO's ``dyn_drg_init`` baroclinic-residual drag
correction to the barotropic forcing (``dynspg_ts.F90:1614-1643``, DINO
branch ``ln_isfcav=F``/``ln_drgice_imp=F``)::

    pCdU_u  = 0.5*(rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj))     (:1616)
    zu_i    = puu(ikbu,Kmm) - puu_b(Kmm)                  (:1627, ln_bt_fw=F -> Kbb residual)
    pu_RHSi += r1_hu(Kmm) * pCdU_u * zu_i                 (:1642)

vs legoESM's analogous ``barotropic_drag_substep`` correction
(``ocean_model_latlon_cgrid.py:3251-3267``,
``F_slow_u -= r_eff_u/H_u_pre * (u_bot - U_bar_now)``, built from
:func:`nemo_bottom_drag_rate_faces` -- the shared NEMO ``zdfdrg``
transcription, single-owner doctrine). Recomputed here INDEPENDENTLY
(read-only spy, no interception of the bare local) against NEMO's own
dumped increment (``drg_dump_zu_frc_inc.bin``/``..zv_frc_inc..``, interior
52x199 convention) and the T-point rate (``drg_dump_rCdU_bot.bin``, full
56x203 domain -- a coarser cross-check that the underlying rate matches
before judging the face increment).

Excludes the periodic-seam column from the interior comparison the same
way ``probe_dyn_cor_2d.py`` does (reported, not silently dropped) since
this row shares the "some rows/columns are a harness reindexing artifact,
not a lego defect" caveat pattern with its sibling row.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.probe_bottom_drag
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_before_state_topo, bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_bottom_drag_rate_faces
from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface, min_cell_to_vface
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe,
    dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)

# Reuse wholesale (module-reuse rule) -- DT (same value every lane) +
# loaders/self-checks every sibling #1226 probe in this package already
# shares. RUN_DIR/RESTART route through dump_lane (#1455 shared selector)
# instead of zu_frc_term_walk's own (still-hardcoded-to-gdb_y5) RUN_DIR --
# this probe must be lane-switchable independently of that (untouchable)
# sibling.
from scripts.validate.ocean_fidelity.dino_1226 import dump_lane
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    DT, _load_interior, _load_full,
)
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_u_structure_probe import (
    _err_norm, _align_scan,
)

RUN_DIR = dump_lane.RUN_DIR
RESTART = dump_lane.RESTART

JPI, JPJ, HLS = 56, 203, 2
SEAM_COL = 49  # already-documented periodic-seam/DINO-sill column (sill_lon_m_deg=1.0)


def _capture_state_mid(model, st, sf):
    """Spy ``state_mid`` (NOW/Kmm, post-slow-tendency momentum) the same way
    ``probe_dyn_cor_2d.py`` does -- the drag correction is built from
    ``state_mid.u/v`` inside ``_step_impl`` before the barotropic solve."""
    captured = {}
    _real_baro = ocmod.barotropic_substeps_latlon_cgrid

    def _spy_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw):
        result = _real_baro(state_mid, dt_s, n_substeps, grid, z_coord, config, **kw)
        if kw.get("eta_init") is not None and "state_mid_u" not in captured:
            captured["state_mid_u"] = state_mid.u.data
            captured["state_mid_v"] = state_mid.v.data
        return result

    ocmod.barotropic_substeps_latlon_cgrid = _spy_baro
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = _real_baro
    assert "state_mid_u" in captured, "barotropic solver never called with a seed"
    return captured


def main() -> int:
    print(dump_lane.banner())
    e3t_mode = require_explicit_e3t_mode(context="probe_bottom_drag")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"barotropic_drag_substep={dcfg.barotropic_drag_substep!r}  "
          f"barotropic_forcing_centred={dcfg.barotropic_forcing_centred!r}  "
          f"bottom_drag_scheme={dcfg.bottom_drag_scheme!r}")
    assert dcfg.barotropic_drag_substep is True

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="probe_bottom_drag twin state")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    captured = _capture_state_mid(model, st, sf)
    U_Nnn = captured["state_mid_u"]
    V_Nnn = captured["state_mid_v"]

    # h_k_pre / h_u_pre / H_u_pre exactly as ocean_model_latlon_cgrid.py's
    # _step_impl builds them (NOW-level thickness -- momentum update never
    # touches eta, see probe_dyn_cor_2d.py's identical construction).
    h_k_pre = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m)
    h_u_pre = min_cell_to_uface(h_k_pre)
    h_v_pre = min_cell_to_vface(h_k_pre, br.geometry)
    H_u_pre = jnp.maximum(jnp.sum(h_u_pre, axis=-1), 1e-10)
    H_v_pre = jnp.maximum(jnp.sum(h_v_pre, axis=-1), 1e-10)

    r_u_bt, r_v_bt, isb_u, isb_v = nemo_bottom_drag_rate_faces(
        st.u.data, st.v.data, h_k_pre, model.z_coord, model.config, br.geometry)

    # barotropic_forcing_centred=True (nemo_dino_kamm_mlf) -> the velocity
    # RESIDUAL uses the BEFORE (Kbb) level (ocean_model_latlon_cgrid.py's
    # _centred_drag branch); geometry/rate stay at NOW (h_k_pre/H_u_pre).
    centred_drag = (
        getattr(mc, "barotropic_forcing_centred", False)
        and getattr(st, "u_before", None) is not None
        and getattr(st, "v_before", None) is not None)
    print(f"centred_drag resolved to: {centred_drag}")
    u_src = st.u_before.data if centred_drag else U_Nnn
    v_src = st.v_before.data if centred_drag else V_Nnn

    u_bot = jnp.sum(u_src * isb_u, axis=-1)
    v_bot = jnp.sum(v_src * isb_v, axis=-1)
    U_bar_now = jnp.sum(u_src * h_u_pre, axis=-1) / H_u_pre
    V_bar_now = jnp.sum(v_src * h_v_pre, axis=-1) / H_v_pre

    zu_frc_inc = -r_u_bt.astype(U_Nnn.dtype) / H_u_pre * (u_bot - U_bar_now)
    zv_frc_inc = -r_v_bt.astype(V_Nnn.dtype) / H_v_pre * (v_bot - V_bar_now)
    zu_frc_inc = np.asarray(zu_frc_inc)
    zv_frc_inc = np.asarray(zv_frc_inc)
    r_u_bt = np.asarray(r_u_bt)
    r_v_bt = np.asarray(r_v_bt)

    def _to_nemo_u(a):
        return np.asarray(a)[:, 1:]

    def _to_nemo_v(a):
        return np.asarray(a)[1:, :]

    nemo_zu_inc = _load_interior(os.path.join(RUN_DIR, "drg_dump_zu_frc_inc.bin"), 52, 199)
    nemo_zv_inc = _load_interior(os.path.join(RUN_DIR, "drg_dump_zv_frc_inc.bin"), 52, 199)
    nemo_rcdu_bot = _load_full(os.path.join(RUN_DIR, "drg_dump_rCdU_bot.bin"), JPI, JPJ, HLS)

    lego_u_inc = _to_nemo_u(zu_frc_inc)
    lego_v_inc = _to_nemo_v(zv_frc_inc)
    n_lat_u = min(lego_u_inc.shape[0], nemo_zu_inc.shape[0])
    n_lon_u = min(lego_u_inc.shape[1], nemo_zu_inc.shape[1])
    n_lat_v = min(lego_v_inc.shape[0], nemo_zv_inc.shape[0])
    n_lon_v = min(lego_v_inc.shape[1], nemo_zv_inc.shape[1])
    lego_u_inc = lego_u_inc[:n_lat_u, :n_lon_u]
    nemo_u_inc = nemo_zu_inc[:n_lat_u, :n_lon_u]
    lego_v_inc = lego_v_inc[:n_lat_v, :n_lon_v]
    nemo_v_inc = nemo_zv_inc[:n_lat_v, :n_lon_v]
    um = umask2[:n_lat_u, :n_lon_u]
    vm = vmask2[:n_lat_v, :n_lon_v]

    print("\n" + "=" * 78)
    print("zdf_drg_nonlin T-point rate (r = Cd|U|, full domain) vs drg_dump_rCdU_bot.bin")
    print("=" * 78)
    # NEMO stores rCdU_bot <= 0 (see nemo_effective_bottom_drag_r docstring
    # sign walk); this helper's own T-point rate is the FACE-averaged
    # r_u_bt/r_v_bt -- de-average back toward the t-point isn't available
    # here (the T-point rate itself lives inside nemo_bottom_drag_rate_faces
    # as a private local), so this block reports the FACE-rate vs the
    # T-point dump only as a magnitude/sign sanity check, not a bar claim
    # (the existing AT-BAR "zdf_drg_nonlin T-point rate" row already owns
    # the T-point comparison via a different, already-committed probe).
    m_full = np.isfinite(nemo_rcdu_bot)
    print(f"  NEMO rCdU_bot: min={np.min(nemo_rcdu_bot[m_full]):.4e} "
          f"max={np.max(nemo_rcdu_bot[m_full]):.4e} "
          f"(expect <= 0, NEMO sign convention)")
    print(f"  lego r_u_bt (face, +convention): min={np.min(r_u_bt):.4e} max={np.max(r_u_bt):.4e}")

    print("\n" + "=" * 78)
    print("dyn_drg_init RHS increment: -r_eff/H*(u_bot-U_bar) vs NEMO drg_dump_z{u,v}_frc_inc.bin")
    print("=" * 78)
    _align_scan("zu_frc_inc", lego_u_inc, nemo_u_inc, um)
    _align_scan("zv_frc_inc", lego_v_inc, nemo_v_inc, vm)

    def _stat(name, lego, nemo, mask, exclude_col=None):
        m = mask.copy()
        if exclude_col is not None and exclude_col < m.shape[1]:
            m[:, exclude_col] = False
        m = m & np.isfinite(lego) & np.isfinite(nemo)
        a, b = lego[m], nemo[m]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")
        rms_b = float(np.sqrt(np.mean(b ** 2))) if b.size else float("nan")
        ratio = float(np.sqrt(np.mean(a ** 2)) / rms_b) if rms_b > 0 else float("nan")
        err_norm, rms, maxdiff, _ = _err_norm(lego, nemo, m)
        print(f"  {name}: n={a.size}  corr={corr:.8f}  abs_ratio={abs(ratio):.6f}  "
              f"err_norm={err_norm:.4e}  rms(nemo)={rms:.4e}  max|diff|={maxdiff:.4e}")
        return corr, ratio

    print("\n-- FULL interior (includes periodic-seam column) --")
    _stat("u", lego_u_inc, nemo_u_inc, um)
    _stat("v", lego_v_inc, nemo_v_inc, vm)

    print(f"\n-- excluding periodic-seam column {SEAM_COL} (harness reindexing artifact) --")
    corr_u, ratio_u = _stat("u", lego_u_inc, nemo_u_inc, um, exclude_col=SEAM_COL)
    corr_v, ratio_v = _stat("v", lego_v_inc, nemo_v_inc, vm, exclude_col=SEAM_COL)

    bar_ok_u = corr_u >= 1.0 - 1e-9 and abs(abs(ratio_u) - 1.0) <= 1e-6
    bar_ok_v = corr_v >= 1.0 - 1e-9 and abs(abs(ratio_v) - 1.0) <= 1e-6
    print("\n" + "=" * 78)
    print(f"dyn_drg_init RHS increment: u corr={corr_u:.6f} abs_ratio={abs(ratio_u):.6f} "
          f"({'AT BAR' if bar_ok_u else 'DEBT'})")
    print(f"dyn_drg_init RHS increment: v corr={corr_v:.6f} abs_ratio={abs(ratio_v):.6f} "
          f"({'AT BAR' if bar_ok_v else 'DEBT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
