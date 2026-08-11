"""NEMO ORCA1 tendency-matching comparator (campaign step c; run via SLURM).

Consumes the RUN_TRD hourly trend re-run (``keep_run3_1h_*`` /
``ORCA1_1h_*_trd1h_*``: state + avt/avs + ttrd_zdf/strd_zdf + stress) and
compares the legoESM zdftke card against NEMO **per process, at a matched
state** — no time integration, so the IC/spin-up/seasonal confounds of
state-matching are absent. Two column-local stages (no model construction):

* **Stage A — closure test**: run the grid-agnostic TKE kernel
  (:func:`tke_vertical_mixing`, the ORCA1 card in the quasi-steady
  diagnostic mode) on NEMO's OWN hour-``r-1`` state (T/S, U/V averaged to
  T-points, rho from the NEMO S-EOS, e3t thicknesses, taum stress) and
  compare the resulting tracer diffusivity ``K_H`` against NEMO's ``avt``
  for hour ``r``.  Tests the CLOSURE alone (what the SST campaign tuned).
  Mode-B is an approximation to NEMO's prognostic en at equilibrium —
  expect agreement in pattern/magnitude, not bitwise.
* **Stage B — operator test**: one backward-Euler implicit solve of the
  hour-``r-1`` T/S with NEMO's OWN ``avt``/``avs`` (hour ``r``) and compare
  ``(T' - T)/dt`` against ``ttrd_zdf``/``strd_zdf`` for hour ``r``.  Tests
  the implicit-diffusion OPERATOR with the closure removed.

Record alignment (rn_Dt=3600 -> ONE step per hourly record): state record
``r-1`` is the field the step-``r`` physics acted on; trend/avt record ``r``
is what that step produced.  Records 0..11 are live (the RK3 build stops
emitting trends at hour 12 — see omip_nemo_tendency_matching notes), so
pairs r=1..11 are usable.

Caveats carried from the trend-run findings: NEMO ``ttrd_zdf`` under this
key_RK3 build INCLUDES the EVD convective enhancement via ``avt`` (no
separate ttrd_evd), and any isoneutral-K33 implicit contribution rides in
``avt`` too — Stage B uses NEMO's avt so both effects cancel there; Stage A
compares against the same avt, so convection-scheme differences show up as
K disagreement in convecting columns (reported per-region, not hidden).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# The Stage-B implicit solve MUST run in float64 (see _fill); set the JAX
# flag before any jax import so jnp.asarray(float64) is not silently demoted.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


REGIONS = (
    ("arctic", 66.0, 90.0),
    ("nh_mid", 30.0, 66.0),
    ("tropics", -15.0, 15.0),
    ("sh_mid", -66.0, -30.0),
    ("antarctic", -90.0, -66.0),
)


def _fill(a):
    a = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
    # FLOAT64 ALWAYS.  The netCDF variables are float32, and the Stage-B
    # backward-Euler solve has diagonal terms dt*K/(dz*dz_half) ~ 3e5 in EVD
    # columns (avs = 100, dz ~ 1 m); solving that system in float32 loses
    # ~0.1 PSU to roundoff and FABRICATES |dS/dt| ~ 1e-4 PSU/s exactly where
    # EVD fires.  Measured on the worst column (j=153, i=280, rec 1): f32
    # gives +1.25e-4 PSU/s in an already-homogenised block (8000x the
    # gradient-limited bound, max principle violated) while f64 gives -8e-8,
    # the NEMO strd_zdf magnitude.  The '46x EVD-salt mismatch' of job
    # 9208553 was this instrument bug, not a model defect.
    return a.astype(np.float64)


def load_pair(tfile: str, ufile: str, vfile: str, rec: int):
    """State at record ``rec-1`` + target avt/trends at record ``rec``."""
    import netCDF4 as nc

    dsT = nc.Dataset(tfile)
    dsU = nc.Dataset(ufile)
    dsV = nc.Dataset(vfile)
    r0, r1 = rec - 1, rec
    out = {
        "T": _fill(dsT.variables["votemper"][r0]),      # (z, y, x) degC
        "S": _fill(dsT.variables["vosaline"][r0]),
        "e3t": _fill(dsT.variables["e3t"][r0]),          # (z, y, x) m
        "taum": _fill(dsT.variables["taum"][r0]),        # (y, x) N/m2
        "u": _fill(dsU.variables["uoce"][r0]),           # (z, y, x) U-points
        "v": _fill(dsV.variables["voce"][r0]),           # V-points
        "avt": _fill(dsT.variables["avt"][r1]),          # (z+? , y, x) W-points
        "avt_rec0": _fill(dsT.variables["avt"][0]),      # step-8761 avt (A2 target)
        "avs": _fill(dsT.variables["avs"][r1]),
        "ttrd_zdf": _fill(dsT.variables["ttrd_zdf"][r1]),
        "strd_zdf": _fill(dsT.variables["strd_zdf"][r1]),
        "T_r1": _fill(dsT.variables["votemper"][r1]),
        "S_r1": _fill(dsT.variables["vosaline"][r1]),
        # XIOS one_file splits coords per grid: T fields on nav_*_grid_T,
        # avt/avm on the W grid (identical horizontal positions on eORCA1).
        "lat": _fill(dsT.variables["nav_lat_grid_T"][:]),
        "lon": _fill(dsT.variables["nav_lon_grid_T"][:]),
        "mld": _fill(dsT.variables["mldr10_1"][r0]),
    }
    dsT.close(); dsU.close(); dsV.close()
    return out


def u_to_T(u):
    """U-point -> T-point average along x (NEMO C-grid: u(i) sits between
    T(i) and T(i+1); T(i) gets 0.5*(u(i-1)+u(i)))."""
    uT = np.full_like(u, np.nan)
    uu = np.nan_to_num(u, nan=0.0)
    w = np.isfinite(u).astype(u.dtype)
    num = uu + np.roll(uu, 1, axis=-1)
    den = w + np.roll(w, 1, axis=-1)
    with np.errstate(invalid="ignore"):
        uT = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return uT


def v_to_T(v):
    vT = np.full_like(v, np.nan)
    vv = np.nan_to_num(v, nan=0.0)
    w = np.isfinite(v).astype(v.dtype)
    num = vv + np.roll(vv, 1, axis=-2)
    den = w + np.roll(w, 1, axis=-2)
    with np.errstate(invalid="ignore"):
        vT = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return vT


def run_stage_a(d, cfg):
    """Closure test: legoESM TKE K_H(state) vs NEMO avt. Returns (K_H, avt_i)
    both at the legoESM interior interfaces (nlev-1 per column)."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing

    T = d["T"]; S = d["S"]; e3t = d["e3t"]
    z, ny, nx = T.shape
    wet = np.isfinite(T)
    ncol = ny * nx

    def cols(a):        # (z,y,x) -> (ncol, z)
        return np.transpose(a.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T), nan=0.0)
    S_c = np.where(np.isfinite(cols(S)), cols(S), 35.0)
    u_c = np.nan_to_num(cols(u_to_T(d["u"])), nan=0.0)
    v_c = np.nan_to_num(cols(v_to_T(d["v"])), nan=0.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])            # centre-to-centre
    # In-situ density from the NEMO S-EOS at cell-centre depth (the same EOS
    # family the tripole recipe selects; kernel only needs rho for insitu N2).
    from legoesm.ocean.eos import nemo_seos_eos
    zc = np.cumsum(dz_c, axis=1) - 0.5 * dz_c                # depth >0 down [m]
    # nemo_seos_eos takes PRESSURE [Pa]; depth enters as zh = p/(rho0*g).
    p_pa = constants.rho_ocean * constants.g * zc
    rho = np.asarray(nemo_seos_eos(jnp.asarray(T_c), jnp.asarray(S_c),
                                   jnp.asarray(p_pa)))
    taum = np.nan_to_num(d["taum"].reshape(ncol), nan=0.0)
    lat_deg = d["lat"].reshape(ncol)
    # Stress components: the card's Dirichlet BC + etau need |tau| only —
    # feed taum through tau_x with tau_y=0 (|.| identical), plus the explicit
    # taum channel for exactness.
    # nn_mxl=3 needs (dz_ref, jacobian): the kernel builds e3t as their
    # rank-1 product (same approximation the production C-grid path makes).
    # Reference profile = per-level max over columns (full-cell thickness;
    # partial bottom cells see a slightly long |dl/dz| cap — bottom-cell-
    # local, negligible for the upper-ocean K comparison).
    dz_ref_1d = np.nanmax(dz_c, axis=0)
    jac1 = np.ones((ncol,), dtype=dz_c.dtype)
    # Interior interface reference heights (negative-down, surface+bottom
    # dropped) — the lc/etau surface terms read interface depths from these
    # (the k_profiles caller passes z_coord.z_half_ref[1:-1]).
    z_interface = -np.cumsum(dz_ref_1d)[:-1]
    out = tke_vertical_mixing(
        jnp.asarray(u_c), jnp.asarray(v_c), jnp.asarray(T_c),
        jnp.asarray(S_c), jnp.asarray(rho), jnp.asarray(dz_half),
        tke_old=None,
        tau_x_surface=jnp.asarray(taum), tau_y_surface=jnp.zeros_like(jnp.asarray(taum)),
        taum_surface=jnp.asarray(taum),
        dt=86400.0, cfg=cfg, rho_0=constants.rho_ocean, g=constants.g,
        n_iterations=3,
        z_interface=jnp.asarray(z_interface),
        lat_deg=jnp.asarray(lat_deg),
        ice_frac=None,
        dz_ref=jnp.asarray(dz_ref_1d), jacobian=jnp.asarray(jac1),
    )
    K_H = np.asarray(out.K_H).reshape(ncol, z - 1)
    # NEMO avt lives at W-points (levels 1..z-1 are the interior interfaces
    # legoESM computes; level 0 is the surface). Slice to interior.
    avt_i = np.transpose(d["avt"].reshape(z, ncol), (1, 0))[:, 1:]
    wet_i = np.transpose(wet.reshape(z, ncol), (1, 0))
    wet_pair = wet_i[:, :-1] & wet_i[:, 1:]
    return K_H, avt_i, wet_pair, (z, ny, nx)


def run_stage_a2_mode_a(d, rst, cfg_prog):
    """EXACT Mode-A closure test: NEMO's own restart state INCLUDING the
    prognostic ``en`` -> ONE legoESM en-step (dt=3600, n_iterations=1) ->
    K_H vs NEMO's avt of the very next step (hourly record 0). Same state,
    same carry — closure implementation vs closure implementation."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
    from legoesm.ocean.eos import nemo_seos_eos

    T3 = rst["tn"]; S3 = rst["sn"]
    z, ny, nx = T3.shape
    ncol = ny * nx

    def cols(a):
        return np.transpose(a.reshape(z, ncol), (1, 0))

    e3t = d["e3t"]  # rec r0 geometry (ssh drift over 1 h is negligible)
    T_c = np.nan_to_num(cols(T3), nan=0.0)
    S_c = np.where(np.isfinite(cols(S3)), cols(S3), 35.0)
    u_c = np.nan_to_num(cols(u_to_T(rst["un"])), nan=0.0)
    v_c = np.nan_to_num(cols(v_to_T(rst["vn"])), nan=0.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
    zc = np.cumsum(dz_c, axis=1) - 0.5 * dz_c
    rho = np.asarray(nemo_seos_eos(
        jnp.asarray(T_c), jnp.asarray(S_c),
        jnp.asarray(constants.rho_ocean * constants.g * zc)))
    taum = np.nan_to_num(d["taum"].reshape(ncol), nan=0.0)
    lat_deg = d["lat"].reshape(ncol)
    dz_ref_1d = np.nanmax(dz_c, axis=0)
    z_interface = -np.cumsum(dz_ref_1d)[:-1]
    # en on W-levels (0=surface); interior interfaces 1..z-1 seed the carry.
    en_i = np.nan_to_num(cols(rst["en"])[:, 1:], nan=0.0)
    out = tke_vertical_mixing(
        jnp.asarray(u_c), jnp.asarray(v_c), jnp.asarray(T_c),
        jnp.asarray(S_c), jnp.asarray(rho), jnp.asarray(dz_half),
        tke_old=jnp.asarray(en_i),
        tau_x_surface=jnp.asarray(taum),
        tau_y_surface=jnp.zeros_like(jnp.asarray(taum)),
        taum_surface=jnp.asarray(taum),
        dt=3600.0, cfg=cfg_prog, rho_0=constants.rho_ocean, g=constants.g,
        n_iterations=1,
        z_interface=jnp.asarray(z_interface),
        lat_deg=jnp.asarray(lat_deg),
        ice_frac=None,
        dz_ref=jnp.asarray(dz_ref_1d),
        jacobian=jnp.asarray(np.ones((ncol,), dtype=dz_c.dtype)),
    )
    K_H = np.asarray(out.K_H).reshape(ncol, z - 1)
    return K_H


def run_stage_b(d):
    """Operator test: BE solve with NEMO's avt/avs vs ttrd_zdf/strd_zdf."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_ocean,
    )

    T = d["T"]; S = d["S"]; e3t = d["e3t"]
    z, ny, nx = T.shape
    ncol = ny * nx

    def cols(a):
        return np.transpose(a.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T), nan=0.0)
    S_c = np.where(np.isfinite(cols(S)), cols(S), 35.0)
    dz_c = np.where(np.isfinite(cols(e3t)), cols(e3t), 1.0)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
    K_t = np.nan_to_num(cols(d["avt"]), nan=0.0)[:, 1:]      # interior interfaces
    K_s = np.nan_to_num(cols(d["avs"]), nan=0.0)[:, 1:]
    dt = 3600.0
    T_new = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(T_c), jnp.asarray(K_t), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    S_new = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(S_c), jnp.asarray(K_s), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    dT = (T_new - T_c) / dt
    dS = (S_new - S_c) / dt
    ttrd = cols(d["ttrd_zdf"])
    strd = cols(d["strd_zdf"])
    wet_c = cols(np.isfinite(T).astype(float)) > 0.5

    # --- Stage B2: solve on the reconstructed PRE-ZDF state ----------------
    # NEMO's implicit zdf acts on the RHS-UPDATED field (after the step's
    # explicit trends: advection, ldf, sbc, qsr...), not on the saved r-1
    # state -- in convecting columns the surface fluxes create the very
    # gradients zdf then removes, so Stage B on the r-1 state systematically
    # undershoots (measured f64: ours 2-10x smaller, corr ~0.1-0.6).
    # ttrd_tot/strd_tot are enabled in the XIOS defs but EMPTY in every
    # RUN_TRD file (finite count 0, measured 2026-08-11), so the budget
    # cannot be closed through the total trend.  It does not need to be:
    # trazdf is the LAST tracer operator of the step, so the field it acted
    # on is exactly
    #     S_pre = S(r) - dt*strd_zdf
    # (its own trend definition, run backwards from the SAVED post-step
    # state).  Our solve on S_pre vs strd_zdf is the exact-form operator
    # test; remaining error = discretization difference (+ K33-in-avt).
    T_r1 = np.nan_to_num(cols(d["T_r1"]), nan=0.0)
    S_r1 = np.where(np.isfinite(cols(d["S_r1"])), cols(d["S_r1"]), 35.0)
    T_pre = np.where(np.isfinite(ttrd), T_r1 - dt * np.nan_to_num(ttrd), T_c)
    S_pre = np.where(np.isfinite(strd), S_r1 - dt * np.nan_to_num(strd), S_c)
    T2 = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(T_pre), jnp.asarray(K_t), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    S2 = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(S_pre), jnp.asarray(K_s), jnp.asarray(dz_c),
        jnp.asarray(dz_half), dt))
    dT2 = (T2 - T_pre) / dt
    dS2 = (S2 - S_pre) / dt
    return dT, dS, ttrd, strd, wet_c, (z, ny, nx), dT2, dS2


def d2_for_a2(d):
    """Stage A2 uses the RESTART state; geometry/stress must be the values in
    effect DURING step 8761 — d already holds rec r0=0 state-row fields when
    --rec 1, which IS the post-step-8761 snapshot; the 1-hour drift in e3t/
    taum is negligible for K (documented approximation)."""
    return d


def avt_a2(d):
    """A2 target = avt RECORD 1 (not 0): the restart stores no avt, so
    NEMO's first-step avt (rec 0) is an initialization transient — measured
    2026-07-27: calm-column rms(avt) 9-22 m2/s at rec 0 vs 0.003-0.1 at
    rec 1, same columns. Scoring one kernel en-step against rec 1 leaves a
    one-step en lag (small: surface TKE e-folds in minutes-hours)."""
    z = d["avt"].shape[0]
    ncol = d["avt"].shape[1] * d["avt"].shape[2]
    return np.transpose(d["avt"].reshape(z, ncol), (1, 0))[:, 1:]


def region_report(name, ours, theirs, wet, lat_col, top_k=None,
                  evd_cols=None):
    """Per-region metrics; when ``evd_cols`` (bool, ncol) is given each
    region is split calm/evd — NEMO's EVD convection sets avt=rn_evd=10
    m2/s, which the Mode-B card does not model, so mixing the two regimes
    scores the convection SCHEME difference, not the closure."""
    rows = []
    subsets = [("", None)] if evd_cols is None else [
        ("/calm", ~evd_cols), ("/evd", evd_cols)]
    for tag0, lo, hi in REGIONS:
      for suff, colsel in subsets:
        tag = tag0 + suff
        m = wet & np.isfinite(theirs) & (lat_col[:, None] >= lo) & (lat_col[:, None] < hi)
        if colsel is not None:
            m = m & colsel[:, None]
        if top_k is not None:
            mm = np.zeros_like(m); mm[:, :top_k] = True; m = m & mm
        if not m.any():
            continue
        a, b = ours[m], theirs[m]
        denom = float(np.sqrt(np.mean(b ** 2)))
        rows.append({
            "region": tag, "n": int(m.sum()),
            "rms_ours": float(np.sqrt(np.mean(a ** 2))),
            "rms_nemo": denom,
            "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "nrmse": float(np.sqrt(np.mean((a - b) ** 2)) / max(denom, 1e-30)),
            "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 3 else np.nan,
        })
    print(f"--- {name} ---")
    for r in rows:
        print(f"  {r['region']:16s} n={r['n']:8d} rms(le)={r['rms_ours']:.3e} "
              f"rms(nemo)={r['rms_nemo']:.3e} nrmse={r['nrmse']:.3f} "
              f"corr={r['corr']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfile", required=True)
    ap.add_argument("--ufile", required=True)
    ap.add_argument("--vfile", required=True)
    ap.add_argument("--rec", type=int, default=1,
                    help="target record r (state r-1 vs avt/trends r); 1..11")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--restart-npz", default=None,
                    help="rebuild_nemo_restart.py output; enables the EXACT "
                         "Mode-A closure test (Stage A2, forces --rec 1: the "
                         "restart state pairs with avt record 0)")
    args = ap.parse_args()

    from scripts.run.run_omip_core2 import orca1_zdftke_config
    cfg = orca1_zdftke_config(prognostic=False)  # Mode-B quasi-steady for the
    # state-function closure test (NEMO's en is prognostic; Mode-B is its
    # documented equilibrium approximation — see module docstring).

    d = load_pair(args.tfile, args.ufile, args.vfile, args.rec)
    lat_col = d["lat"].reshape(-1)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = {"rec": args.rec}

    K_H, avt_i, wet_pair, shp = run_stage_a(d, cfg)
    # EVD flag: any interface in the column at/above NEMO's rn_evd scale.
    evd_cols = np.nanmax(np.where(wet_pair, avt_i, 0.0), axis=1) > 1.0
    print(f"[evd] convecting columns: {int(evd_cols.sum())} "
          f"({100.0 * evd_cols.mean():.1f}% of columns)")
    result["stage_a"] = region_report(
        "Stage A: closure K_H (legoESM TKE card) vs NEMO avt [m2/s]",
        K_H, avt_i, wet_pair, lat_col, evd_cols=evd_cols)

    if args.restart_npz:
        if args.rec != 1:
            raise SystemExit("--restart-npz requires --rec 1 (the restart "
                             "state pairs with avt record 0)")
        rst = dict(np.load(args.restart_npz))
        cfg_a2 = orca1_zdftke_config()  # prognostic=True card
        K_H2 = run_stage_a2_mode_a(d2_for_a2(d), rst, cfg_a2)
        result["stage_a2_mode_a"] = region_report(
            "Stage A2: MODE-A closure — kernel(restart state + NEMO en, one "
            "3600s en-step) K_H vs NEMO avt rec 0 [m2/s]",
            K_H2, avt_a2(d), wet_pair, lat_col, evd_cols=evd_cols)

    dT, dS, ttrd, strd, wet_c, _, dT2, dS2 = run_stage_b(d)
    result["stage_b_T"] = region_report(
        "Stage B: operator dT_zdf (BE solve w/ NEMO avt) vs ttrd_zdf [K/s]",
        dT, ttrd, wet_c & np.isfinite(ttrd), lat_col, evd_cols=evd_cols)
    result["stage_b_S"] = region_report(
        "Stage B: operator dS_zdf vs strd_zdf [PSU/s]",
        dS, strd, wet_c & np.isfinite(strd), lat_col, evd_cols=evd_cols)
    result["stage_b2_T"] = region_report(
        "Stage B2: operator on PRE-ZDF state (tot-closure) dT vs ttrd_zdf [K/s]",
        dT2, ttrd, wet_c & np.isfinite(ttrd), lat_col, evd_cols=evd_cols)
    result["stage_b2_S"] = region_report(
        "Stage B2: operator on PRE-ZDF state dS vs strd_zdf [PSU/s]",
        dS2, strd, wet_c & np.isfinite(strd), lat_col, evd_cols=evd_cols)

    with open(out_dir / f"tendency_match_rec{args.rec}.json", "w") as f:
        json.dump(result, f, indent=1)

    if not args.skip_maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z, ny, nx = shp
        k10 = 3   # ~10-30 m interface on the 75-level grid
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        panels = [
            ("legoESM K_H (iface k=3)", K_H[:, k10].reshape(ny, nx), 0, 1e-2),
            ("NEMO avt (iface k=3)", avt_i[:, k10].reshape(ny, nx), 0, 1e-2),
            ("dT_zdf legoESM (k=0)", dT[:, 0].reshape(ny, nx), -2e-5, 2e-5),
            ("ttrd_zdf NEMO (k=0)", ttrd[:, 0].reshape(ny, nx), -2e-5, 2e-5),
            ("Stage B T residual (k=0)", (dT - ttrd)[:, 0].reshape(ny, nx),
             -2e-6, 2e-6),
            ("Stage A K residual (iface k=3)",
             (K_H - avt_i)[:, k10].reshape(ny, nx), -5e-3, 5e-3),
        ]
        for ax, (title, fld, vmin, vmax) in zip(axes.flat, panels):
            wm = np.isfinite(d["T"][0])
            fld = np.where(wm, fld, np.nan)
            im = ax.pcolormesh(fld, vmin=vmin, vmax=vmax,
                               cmap="RdBu_r" if vmin < 0 else "viridis")
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"NEMO tendency match — rec {args.rec} "
                     f"(state r-1 vs avt/trends r)", fontsize=11)
        fig.tight_layout()
        png = out_dir / f"tendency_match_rec{args.rec}.png"
        fig.savefig(png, dpi=110)
        print(f"[maps] {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
