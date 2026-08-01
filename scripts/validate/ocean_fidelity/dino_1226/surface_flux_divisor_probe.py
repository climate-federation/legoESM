#!/usr/bin/env python
"""#1226 tra_sbc residual: STATIC dz_ref[0] vs LIVE e3t(:,:,1,Kmm) divisor.

THE LEAD (task brief): trasbc.F90:152-153 divides the MLF-averaged surface
tracer flux by ``e3t(ji,jj,1,Kmm)`` -- under ``key_qco`` (DINO's active
vertical-coordinate key, ``cpp_DINO.fcm``: ``key_qco key_vco_3d``) this is
NOT the static reference thickness but the LIVE z*-stretched value

    e3t(i,j,1,Kmm) = E3t_0(i,j,1) * (1 + r3t(i,j,Kmm))          [domzgr_substitute.h90:139,145]
    r3t(i,j,Kmm)   = ssh(i,j,Kmm) / ht_0(i,j)                    [domqco.F90:160]
    E3t_0(i,j,1)   = e3t_3d(i,j,1)   (key_vco_3d)                [domzgr_substitute.h90:104-112]

legoESM's ``apply_dino_lat_lon_surface_forcing`` (dino.py:3321,3345,3357,3388,
3455) instead divides by the CONSTANT ``z_coord.dz_ref[0]`` == ``E3t_0(:,:,1)``
with NO ``*(1+r3t)`` stretch. This script A/B's the ``coverage_rows_measure.py
measure_tra_sbc`` row (tem ratio 0.99993829, sal ratio 0.99995924) with the
ONLY change being that divisor -- everything else (state build, MLF average,
restart sbc_hc_b/sbc_sc_b, alignment scan, per_element_stats) is the SAME
machinery, imported from that sibling script, not re-derived.

Confirmed NEMO call chain for the divisor and its time level (read, not
assumed):
  * stpmlf.F90:387        CALL tra_sbc( kstp, Nnn, ts, Nrhs )   -- Kmm=Nnn (NOW)
  * trasbc.F90:150-155    pts(...,Krhs) += zfact*(sbc_tsc_b+sbc_tsc) / e3t(:,:,1,Kmm)
  * cpp_DINO.fcm          key_qco key_vco_3d (NOT key_RK3, NOT key_linssh)
  * domzgr_substitute.h90:104-112 (key_vco_3d): E3t_0(i,j,k) = e3t_3d(i,j,k)
  * domzgr_substitute.h90:139     e3t(i,j,k,t) = E3t_0(i,j,k)*(1+r3t(i,j,t)*tmask(i,j,k))
  * domqco.F90:160               pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)
  * domqco.F90:131 (dom_qco_zgr) CALL dom_qco_r3c( ssh(:,:,Kmm), r3t(:,:,Kmm), ... )
    -- so r3t(Kmm) uses ssh AT Kmm, matching Nnn == the restart's ``sshn``
    (read_nemo_restart reads "sshn", nemo_io.py:212) -- the SAME time level
    ``tra_sbc`` is called with (Kmm=Nnn). No time-level substitution risk here
    (Rule 1d): both eta and the tra_sbc dump are at Nnn/Kmm.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/surface_flux_divisor_probe.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import netCDF4 as nc
import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(__file__)


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the EXACT sibling machinery -- state build, dump loaders, alignment
# scan, per-element stats, and the dump time-level registry (importing this
# module runs its module-level register_dump(...) calls as a side effect).
_cov = _load_sibling("coverage_rows_measure.py", "_coverage_rows_measure")

RUN_DIR = _cov.RUN_DIR
RESTART = _cov.RESTART
KT_DUMP = _cov.KT_DUMP
_load_haloed = _cov._load_haloed
_shift_scan = _cov._shift_scan
_print_offset_table = _cov._print_offset_table
per_element_stats = _cov.per_element_stats

from legoesm.ocean.experiments.dino import dino_T_star_seasonal, dino_S_star
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import read_nemo_restart_before
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_before_state_topo
from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import restoring_surface_forcing


class _LatShim:
    def __init__(self, lat):
        self.grid_lat = lat


def _live_r3t_and_e3t0(br, tmask2d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r3t(Kmm) = eta(Kmm)/H_bathy(ht_0), and the LIVE e3t(:,:,1,Kmm).

    Dry columns (H_bathy==0) -> r3t=0 (mirrors nemo_bn2_live_ladders' inert
    convention, eos.py:681-684 -- those cells are masked out of every wet-cell
    stat anyway).
    """
    eta = np.asarray(br.state.eta.data)               # ssh(Kmm), nemo_io.py:212 "sshn"
    H = np.asarray(br.state.H_bathy.data)              # ht_0 (per-column, full-step)
    r3t = np.where(H > 0.0, eta / np.where(H > 0.0, H, 1.0), 0.0)
    e3t0_k1 = np.asarray(br.z_coord.dz_ref)[0]          # E3t_0(i,j,1) == e3t_3d(i,j,0) under key_vco_3d/LEGOESM_NEMO_E3T=both
    e3t0_k1_2d = np.broadcast_to(np.float64(e3t0_k1), tmask2d.shape) if np.ndim(e3t0_k1) == 0 else e3t0_k1
    e3t_live_k1 = e3t0_k1_2d * (1.0 + r3t)               # domzgr_substitute.h90:139, tmask==1 at k=0 for every wet cell
    return r3t, e3t0_k1_2d, e3t_live_k1


def _recompute_tem_sal(st, dz_0_2d) -> tuple[np.ndarray, np.ndarray]:
    """Same MLF-average recomputation as measure_tra_sbc, but with a
    (possibly per-cell) e3t divisor supplied by the caller instead of the
    scalar ``e3t_k0`` that function hardcodes -- the ONE VARIABLE under test."""
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    grid, br, cfg = st["grid"], st["br"], st["cfg"]
    tmask2d = st["tmask2d"]
    n_lat, n_lon = tmask2d.shape

    t_seconds = KT_DUMP * cfg.dt
    lat_deg_1d = np.degrees(np.asarray(br.geometry.lat))
    T_star_1d = np.asarray(dino_T_star_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    T_star_2d = np.broadcast_to(T_star_1d[:, None], (n_lat, n_lon))
    S_star_1d = np.asarray(dino_S_star(jnp.asarray(lat_deg_1d), cfg))
    S_star_2d = np.broadcast_to(S_star_1d[:, None], (n_lat, n_lon))
    from legoesm.ocean.experiments.dino import dino_Q_sr_seasonal
    Q_sr_1d = np.asarray(dino_Q_sr_seasonal(jnp.asarray(lat_deg_1d), t_seconds, cfg))
    Q_sr_2d = np.broadcast_to(Q_sr_1d[:, None], (n_lat, n_lon))

    # tau_T/tau_S: NEMO derives these from the SAME dz_0 the paper's A_theta/A_S
    # flux coefficients assume (config.py tau_from_flux_coefficient docstring:
    # "the way NEMO derives tau_T = rho0*cp*dz_0/A_theta"). The task's ONE
    # VARIABLE is trasbc.F90's per-step divisor of the ALREADY-formed
    # sbc_tsc/sbc_tsc_b flux-content fields (trasbc.F90:152-153), which is
    # downstream of tau_T/tau_S -- so tau_T/tau_S stay built from the static
    # dz_0 in BOTH variants (matching cfg.A_theta's fixed physical meaning);
    # only the FINAL divide-by-e3t step (this function's own dz_0_2d arg) is
    # swapped. This keeps the comparison a true one-variable change.
    dz_0_scalar = float(br.z_coord.dz_ref[0])
    tau_T = tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p, dz_0_scalar)
    tau_S = tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz_0_scalar)
    restoring_cfg = RestoringConfig(
        tau_T=tau_T, tau_S=tau_S, T_star_array=jnp.asarray(T_star_2d),
        S_star_array=jnp.asarray(S_star_2d), subtract_qsr=True, implicit=False,
    )

    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br_before_state = bridge_before_state_topo(br, grid, bef, periodic_i=True)
    T_Kbb = br_before_state.T_before.data
    S_Kbb = br_before_state.S_before.data

    # NOTE: restoring_surface_forcing's OWN internal Q_sr-subtraction divides
    # by the SCALAR dz_0 it is passed (restoring.py:165), which is a config
    # constant tied to tau_T/tau_S's fixed derivation, not the per-step
    # divisor under test. Pass the scalar there (unchanged from
    # measure_tra_sbc) and apply the variant divisor ONLY in the MLF-average
    # step below, which is the literal trasbc.F90:152-153 analogue.
    out = restoring_surface_forcing(
        T_Kbb, S_Kbb, _LatShim(jnp.zeros((n_lat, n_lon))), restoring_cfg,
        sw_down=jnp.asarray(Q_sr_2d), dt=cfg.dt, rho_0=cfg.rho_0, c_p=cfg.c_p,
        dz_0=dz_0_scalar)
    dT_dt_now = np.asarray(out.dT_dt[..., 0])
    dS_dt_now = np.asarray(out.dS_dt[..., 0])

    with nc.Dataset(os.path.join(RUN_DIR, RESTART)) as r:
        sbc_hc_b = np.asarray(r["sbc_hc_b"][0]).squeeze()
        sbc_sc_b = np.asarray(r["sbc_sc_b"][0]).squeeze()

    # trasbc.F90:152-153: pts(Krhs) += zfact*(sbc_tsc_b + sbc_tsc)/e3t(:,:,1,Kmm)
    # sbc_tsc_b is READ from the restart in flux-content [K*m/s]/[PSU*m/s]
    # units (a PREVIOUS-step quantity, independent of this divisor choice --
    # trasbc.F90:118 reads it once via iom_get, never re-divides it before
    # this line). sbc_tsc (THIS step) is dT_dt_now (a rate, [K/s]) computed by
    # legoESM already-divided-by-dz_0_scalar; NEMO's own sbc_tsc (line 136-137)
    # is a pure [K*m/s] FLUX-CONTENT with NO e3t division at all -- the
    # divide-by-e3t happens ONCE, at line 152-153, applied to the SUM. So the
    # bit-faithful reconstruction multiplies dT_dt_now back up by the SAME
    # dz_0_scalar it was divided by, to recover the flux-content, then applies
    # the (possibly live) test divisor exactly once -- matching trasbc.F90's
    # own single-division structure.
    sbc_tsc_now_T = dT_dt_now * dz_0_scalar
    sbc_tsc_now_S = dS_dt_now * dz_0_scalar
    tem = 0.5 * (sbc_hc_b + sbc_tsc_now_T) / dz_0_2d
    sal = 0.5 * (sbc_sc_b + sbc_tsc_now_S) / dz_0_2d
    return tem, sal


def main() -> int:
    st = _cov.build_state()
    tmask2d = st["tmask2d"]
    br = st["br"]

    print("\n" + "=" * 78)
    print("surface_flux_divisor_probe: STATIC dz_ref[0] vs LIVE e3t(:,:,1,Kmm)")
    print("=" * 78)

    lvl = time_level_for_dump("stp_dump_14_trasbc_tem.bin")
    print(f"  time_level_for_dump('stp_dump_14_trasbc_tem.bin') = {lvl!r}  "
          "(Nnn/now -- matches ssh(Kmm) used for r3t below, per "
          "domqco.F90:131 CALL dom_qco_r3c(ssh(:,:,Kmm), r3t(:,:,Kmm), ...))")

    r3t, e3t0_k1_2d, e3t_live_k1 = _live_r3t_and_e3t0(br, tmask2d)

    # ---- Step 1: r3t magnitude distribution + dtype -----------------------
    print(f"\n  dtypes: eta={np.asarray(br.state.eta.data).dtype} "
          f"H_bathy={np.asarray(br.state.H_bathy.data).dtype} "
          f"dz_ref={np.asarray(br.z_coord.dz_ref).dtype}")
    r3t_wet = r3t[tmask2d]
    rel_stretch = np.abs((e3t_live_k1 / e3t0_k1_2d - 1.0)[tmask2d])
    print(f"  r3t = ssh(Kmm)/ht_0 over {int(tmask2d.sum())} wet T-cells: "
          f"median={np.median(r3t_wet):.6e}  p95={np.percentile(np.abs(r3t_wet), 95):.6e}  "
          f"max|r3t|={np.max(np.abs(r3t_wet)):.6e}")
    print(f"  (e3t_live/e3t_0 - 1) == r3t exactly at k=0 (tmask=1): "
          f"median={np.median(rel_stretch):.6e}  p95={np.percentile(rel_stretch, 95):.6e}  "
          f"max={np.max(rel_stretch):.6e}")
    row_residual_tem = 1.0 - 0.99993829
    row_residual_sal = 1.0 - 0.99995924
    print(f"  Row's own after-averaging residual (from task brief): "
          f"tem ratio-1={row_residual_tem:.6e}  sal ratio-1={row_residual_sal:.6e}")
    print("  Arithmetic: the DIRECT per-cell relative divisor error is "
          f"~median(r3t)={np.median(np.abs(r3t_wet)):.3e}, i.e. ~"
          f"{np.median(np.abs(r3t_wet))/max(row_residual_tem, 1e-30):.1f}x the "
          "row's own residual scale BEFORE the MLF 0.5*(now+before) averaging "
          "(which mixes in the PREVIOUS step's already-divided sbc_tsc_b, "
          "diluting a purely-this-step effect by up to 2x) and the aggregate "
          "|x|-ratio metric (which is a sum-of-residuals, not a per-cell "
          "median, and can partially cancel across signed cells).")

    # ---- Step 2 (task item 2): every legoESM surface-flux divisor site ----
    print("\n" + "-" * 78)
    print("legoESM DINO surface-flux divisor sites (grep sweep, every site "
          "that applies a surface flux to the top cell):")
    print("-" * 78)
    rows = [
        ("packages/ocean/legoesm/ocean/experiments/dino.py:3321",
         "apply_dino_lat_lon_surface_forcing: dz_0=z_coord.dz_ref[0] -> "
         "tau_T/tau_S (T/S restoring timescales) + dz_0 kwarg to "
         "restoring_surface_forcing (Q_sr subtraction, T/S restoring surface term)",
         "STATIC", "trasbc.F90:152-153 e3t(:,:,1,Kmm), LIVE under key_qco"),
        ("packages/ocean/legoesm/ocean/experiments/dino.py:3388",
         "dino_top_layer_u_tendency(forcing['tau_u_face'], dz_0, cfg) -- wind "
         "stress into top-layer u tendency (paper eq 7)",
         "STATIC", "dynzdf.F90 surface stress BC divides by e3u(:,:,1,Kmm) "
         "(momentum-side live thickness; same key_qco stretch, u-point r3u)"),
        ("packages/ocean/legoesm/ocean/experiments/dino.py:3455",
         "apply_dino_mpas_surface_forcing: dz_0=z_coord.dz_ref[0] -> "
         "dino_top_layer_T_tendency (MPAS-mesh T restoring)",
         "STATIC", "same trasbc.F90:152-153 counterpart (MPAS DINO recipe is "
         "a separate mesh, not NEMO-bridged, so no live NEMO r3t to compare "
         "against directly -- flagged for completeness, not measured here)"),
        ("packages/ocean/legoesm/ocean/physics/surface_forcing/prescribed.py:93",
         "dz_0 = z_coord.dz_ref[0] * jacobian -- generic 'prescribed' scheme "
         "surface flux-to-tendency (non-DINO recipes' integration.py dispatch)",
         "STATIC (jacobian-scaled, not r3t-stretched)",
         "trasbc.F90:152-153 (same family; this scheme's own oracle is "
         "usually Veros, whose fixed-lid dzt[-1] IS static -- see "
         "flux_feedback.py:33 comment: matches Veros exactly, not NEMO)"),
        ("packages/ocean/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:161",
         "dz_0 = z_coord.dz_ref[0] * jacobian -- bulk-formula scheme surface flux",
         "STATIC (jacobian-scaled)", "same family as prescribed.py"),
        ("packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py:135,138"
         " (_make_external) / :169 (_make_flux_feedback)",
         "h = compute_layer_thickness(eta, H_bathy, z_coord)[..., 0] -- "
         "ALREADY uses the LIVE partial-cell-aware top thickness, per its own "
         "comment '(not dz_ref[0]*J) so the flux-to-tendency conversion is "
         "conservative on shallow top cells'",
         "LIVE (already fixed)", "matches the live-divisor pattern this "
         "probe recommends for dino.py; these two schemes are NOT DINO's "
         "own call path (DINO uses apply_dino_lat_lon_surface_forcing "
         "directly, not the integration.py scheme dispatch)"),
        ("packages/ocean/legoesm/ocean/biogeochemistry/carbon_cycle.py:91",
         "dz_surface = dz_ref[0] -- surface CO2 flux to top-cell tendency",
         "STATIC", "not a NEMO-DINO-active path (BGC off for DINO); flagged "
         "for completeness only"),
        ("packages/ocean/legoesm/ocean/coupler/runoff_apply.py / sss_apply.py",
         "dz_ref[0] used for runoff / SSS-restoring-from-coupler top-layer thickness",
         "STATIC", "coupler paths, not exercised by the bridged NEMO-DINO "
         "fidelity harness (no coupler active here); flagged for completeness"),
    ]
    for site, desc, kind, counterpart in rows:
        print(f"\n  site: {site}")
        print(f"    what: {desc}")
        print(f"    divisor: {kind}")
        print(f"    NEMO counterpart: {counterpart}")

    # ---- Step 3: A/B on the RUN_GDB kt=57601 state ------------------------
    print("\n" + "-" * 78)
    print("A/B: recompute tra_sbc's own tem/sal tendency, STATIC vs LIVE divisor")
    print("-" * 78)

    tem_nemo_full = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_tem.bin"),
                                  st["jpi"], st["jpj"], st["hls"])
    sal_nemo_full = _load_haloed(os.path.join(RUN_DIR, "stp_dump_14_trasbc_sal.bin"),
                                  st["jpi"], st["jpj"], st["hls"])
    tem_nemo = tem_nemo_full[..., 0]
    sal_nemo = sal_nemo_full[..., 0]

    tem_static, sal_static = _recompute_tem_sal(st, e3t0_k1_2d)
    tem_live, sal_live = _recompute_tem_sal(st, e3t_live_k1)

    print("\n  --- VARIANT A: STATIC dz_ref[0] (legoESM's current production divisor) ---")
    _shift_scan("tra_sbc tem [STATIC]", tem_static[:, :, None], tem_nemo[:, :, None],
                tmask2d[:, :, None])
    r_tem_static = per_element_stats("tra_sbc tem [STATIC]", tem_static, tem_nemo,
                                      tmask2d, sign_changing=True)
    r_sal_static = per_element_stats("tra_sbc sal [STATIC]", sal_static, sal_nemo,
                                      tmask2d, sign_changing=True)

    print("\n  --- VARIANT B: LIVE e3t(:,:,1,Kmm) = e3t_0*(1+r3t) ---")
    _shift_scan("tra_sbc tem [LIVE]", tem_live[:, :, None], tem_nemo[:, :, None],
                tmask2d[:, :, None])
    r_tem_live = per_element_stats("tra_sbc tem [LIVE]", tem_live, tem_nemo,
                                    tmask2d, sign_changing=True)
    r_sal_live = per_element_stats("tra_sbc sal [LIVE]", sal_live, sal_nemo,
                                    tmask2d, sign_changing=True)

    # ---- Self-checks -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SELF-CHECKS")
    print("-" * 78)
    # (1) the two variants differ ONLY in the divisor: force r3t->0 (as if
    # eta==0 everywhere) and confirm STATIC/LIVE become bit-identical.
    e3t_live_zero_eta = e3t0_k1_2d * (1.0 + 0.0 * r3t)
    tem_live_zero, sal_live_zero = _recompute_tem_sal(st, e3t_live_zero_eta)
    d_tem = float(np.max(np.abs(tem_live_zero - tem_static)))
    d_sal = float(np.max(np.abs(sal_live_zero - sal_static)))
    print(f"  [self-check 1] forcing r3t->0 (eta->0 in the divisor only): "
          f"max|tem_live0 - tem_static|={d_tem:.3e}  "
          f"max|sal_live0 - sal_static|={d_sal:.3e}  "
          f"(want 0.0 -- proves the ONLY change between variants is the "
          "divisor, not some other silently-differing path)")
    assert d_tem == 0.0 and d_sal == 0.0, (
        "variants differ in more than the divisor when r3t is forced to 0 -- "
        "controlled-comparison premise VIOLATED")

    # (2) manual scalar-loop recomputation of the LIVE divisor at a handful of
    # wet cells, cross-checked against the vectorized e3t_live_k1 array.
    wet_idx = np.argwhere(tmask2d)[::max(1, int(tmask2d.sum()) // 5)][:5]
    manual_max_diff = 0.0
    eta_arr = np.asarray(br.state.eta.data)
    H_arr = np.asarray(br.state.H_bathy.data)
    for jy, ix in wet_idx:
        r3t_manual = eta_arr[jy, ix] / H_arr[jy, ix] if H_arr[jy, ix] > 0 else 0.0
        e3t_manual = float(e3t0_k1_2d[jy, ix]) * (1.0 + r3t_manual)
        manual_max_diff = max(manual_max_diff, abs(e3t_manual - e3t_live_k1[jy, ix]))
    print(f"  [self-check 2] manual (scalar loop) vs vectorized e3t_live_k1 at "
          f"{len(wet_idx)} sample wet cells: max|diff|={manual_max_diff:.3e} "
          "(want 0.0)")
    assert manual_max_diff < 1e-12

    # ---- Verdict ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  tem: STATIC corr={r_tem_static['corr']:.8f} ratio_abs={r_tem_static['ratio_abs']:.8f}"
          f"  ->  LIVE corr={r_tem_live['corr']:.8f} ratio_abs={r_tem_live['ratio_abs']:.8f}")
    print(f"  sal: STATIC corr={r_sal_static['corr']:.8f} ratio_abs={r_sal_static['ratio_abs']:.8f}"
          f"  ->  LIVE corr={r_sal_live['corr']:.8f} ratio_abs={r_sal_live['ratio_abs']:.8f}")

    tol = 1.0e-6
    tem_at_bar = abs(r_tem_live["ratio_abs"] - 1.0) < tol and abs(r_tem_live["corr"] - 1.0) < tol
    sal_at_bar = abs(r_sal_live["ratio_abs"] - 1.0) < tol and abs(r_sal_live["corr"] - 1.0) < tol
    improved_tem = abs(r_tem_live["ratio_abs"] - 1.0) < abs(r_tem_static["ratio_abs"] - 1.0)
    improved_sal = abs(r_sal_live["ratio_abs"] - 1.0) < abs(r_sal_static["ratio_abs"] - 1.0)

    if tem_at_bar and sal_at_bar:
        print("  CONFIRMED: the live e3t(:,:,1,Kmm)=e3t_0*(1+r3t) divisor moves "
              "both tem and sal to ratio/corr ~1.0 (within 1e-6) -- the static "
              "dz_ref[0] divisor is the row's residual cause.")
    elif improved_tem and improved_sal:
        print("  PLAUSIBLE/PARTIAL: the live divisor moves BOTH tem and sal "
              "closer to 1.0 but does not fully close the row to the 1e-6 bar "
              "-- consistent with the lead but not sufficient alone; residual "
              "structure printed above (offset-table / per_element_stats) is "
              "the next probe's starting point.")
    else:
        print("  REFUTED (or not the dominant term): the live divisor does NOT "
              "improve both tem and sal ratio/corr toward 1.0 -- see the "
              "per_element_stats structure above (median|rel| vs p99 vs max, "
              "sign_changing, near0_frac) for what the residual actually looks "
              "like; do not chase this lead further without new evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
